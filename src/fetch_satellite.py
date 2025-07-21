import os
import logging
import logging.handlers
import tomli
from datetime import datetime, timedelta, timezone
from sentinelhub import (
    SHConfig, SentinelHubCatalog, SentinelHubRequest,
    MimeType, CRS, BBox, DataCollection
)
import rasterio
import numpy as np
from scipy.ndimage import label
from rasterio.features import shapes
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor
import time

# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)

# Setup logging
log_file = config_data["logging"]["log_file"]
log_level = getattr(logging, config_data["logging"]["log_level"])
max_size = config_data["logging"]["max_size"]
backup_count = config_data["logging"]["backup_count"]
os.makedirs(os.path.dirname(log_file), exist_ok=True)
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler]
)
logger = logging.getLogger(__name__)

# Setup Sentinel Hub credentials
sh_config = SHConfig()
sh_config.sh_client_id = config_data["sentinelhub"]["client_id"]
sh_config.sh_client_secret = config_data["sentinelhub"]["client_secret"]
sh_config.instance_id = config_data["sentinelhub"]["instance_id"]

# Validate SentinelHub credentials
if not (sh_config.sh_client_id and sh_config.sh_client_secret):
    logger.error("SentinelHub credentials missing in config.toml")
    raise ValueError("SentinelHub client_id or client_secret is missing")

# Define bounding box for all of Kenya (from config.toml)
area = config_data["area"]
west, south, east, north = (
    area["west"], area["south"],
    area["east"], area["north"]
)

# Split into 4 tiles
mid_x = (west + east) / 2
mid_y = (south + north) / 2
tiles = [
    BBox((west, south, mid_x, mid_y), crs=CRS.WGS84),
    BBox((mid_x, south, east, mid_y), crs=CRS.WGS84),
    BBox((west, mid_y, mid_x, north), crs=CRS.WGS84),
    BBox((mid_x, mid_y, east, north), crs=CRS.WGS84)
]

# Define data folder for saving TIFFs
data_folder = "data/sentinel"

def validate_tiff(tiff_path, expected_dtype, expected_shape=(1024, 1024)):
    """Validate TIFF file integrity and data range."""
    try:
        with rasterio.open(tiff_path) as src:
            data = src.read(1)
            if data.shape != expected_shape:
                logger.warning(f"TIFF {tiff_path} has unexpected shape: {data.shape}")
                return False
            if data.dtype != expected_dtype:
                logger.warning(f"TIFF {tiff_path} has unexpected dtype: {data.dtype}")
                return False
            if np.all(data == 0):
                logger.warning(f"TIFF {tiff_path} contains all zeros")
                return False
            if "water_mask" in tiff_path and not np.all(np.isin(data, [0, 1])):
                logger.warning(f"Water mask TIFF {tiff_path} contains invalid values")
                return False
            if "turbidity" in tiff_path and np.any(data < -1) or np.any(data > 1):
                logger.warning(f"Turbidity TIFF {tiff_path} contains out-of-range values")
                return False
            if "chlorophyll" in tiff_path and np.any(data < 0) or np.any(data > 100):
                logger.warning(f"Chlorophyll TIFF {tiff_path} contains out-of-range values")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to validate TIFF {tiff_path}: {str(e)}")
        return False

def fetch_water_data_for_tile(bbox, tile_id):
    """Fetch and process water data for a single tile."""
    start_time = time.time()
    logger.info(f"Starting processing for tile {tile_id}")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)

    # Check for cached TIFFs
    tile_folder = os.path.join(data_folder, f"tile_{tile_id}")
    mask_tiff = os.path.join(tile_folder, "water_mask.tif")
    turbidity_tiff = os.path.join(tile_folder, "turbidity.tif")
    chlorophyll_tiff = os.path.join(tile_folder, "chlorophyll.tif")

    tiff_files = [mask_tiff, turbidity_tiff, chlorophyll_tiff]
    if all(os.path.isfile(f) for f in tiff_files) and all(validate_tiff(mask_tiff, np.uint8) and
                                                          validate_tiff(turbidity_tiff, np.float32) and
                                                          validate_tiff(chlorophyll_tiff, np.float32)):
        logger.info(f"Tile {tile_id}: Using cached TIFFs")
    else:
        logger.info(f"Tile {tile_id}: Fetching new Sentinel-2 data")
        try:
            catalog = SentinelHubCatalog(sh_config)
            search = catalog.search(
                collection='sentinel-2-l2a',
                datetime=f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                limit=1,
                bbox=bbox,
                filter=f"eo:cloud_cover < {config_data['sentinelhub']['max_cloud']}"
            )
            results_list = list(search)
            if not results_list:
                logger.warning(f"Tile {tile_id}: No imagery found")
                return [], 0

            # Create data request
            with open("src/evalscript_wbm.js") as f:
                evalscript = f.read()

            os.makedirs(tile_folder, exist_ok=True)
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("water_mask", MimeType.TIFF),
                    SentinelHubRequest.output_response("turbidity", MimeType.TIFF),
                    SentinelHubRequest.output_response("chlorophyll", MimeType.TIFF)
                ],
                bbox=bbox,
                size=(1024, 1024),  # Maintain original resolution
                config=sh_config,
                data_folder=tile_folder
            )
            for attempt in range(3):  # Retry up to 3 times
                try:
                    data = request.get_data(save_data=True)
                    break
                except Exception as e:
                    logger.warning(f"Tile {tile_id}: API request failed (attempt {attempt + 1}): {str(e)}")
                    if attempt == 2:
                        logger.error(f"Tile {tile_id}: Failed after 3 attempts")
                        return [], 0
                    time.sleep(2)

            # Validate API response
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                logger.error(f"Tile {tile_id}: Invalid API response: {type(data)}")
                return [], 0

            data_dict = data[0]
            if isinstance(data_dict.get("water_mask.tif"), np.ndarray):
                logger.info(f"Tile {tile_id}: Saving API response as TIFFs")
                transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, 1024, 1024)
                for output_id, array in data_dict.items():
                    tiff_path = os.path.join(tile_folder, f"{output_id}")
                    dtype = np.uint8 if output_id == "water_mask.tif" else np.float32
                    array = array.astype(dtype)
                    with rasterio.open(
                            tiff_path,
                            'w',
                            driver='GTiff',
                            height=1024,
                            width=1024,
                            count=1,
                            dtype=dtype,
                            crs='EPSG:4326',
                            transform=transform
                    ) as dst:
                        dst.write(array, 1)
        except Exception as e:
            logger.error(f"Tile {tile_id}: Failed to fetch/process data: {str(e)}")
            return [], 0

    # Verify file paths
    for tiff_file in tiff_files:
        if not os.path.isfile(tiff_file):
            logger.error(f"Tile {tile_id}: TIFF file not found: {tiff_file}")
            return [], 0

    # Process TIFFs
    results = []
    try:
        with rasterio.open(mask_tiff) as src:
            mask = src.read(1) == 1
            structure = np.ones((3, 3), dtype=int)
            labeled, n = label(mask, structure=structure)

            with rasterio.open(turbidity_tiff) as turb_src, rasterio.open(chlorophyll_tiff) as chl_src:
                turbidity = turb_src.read(1)
                chlorophyll = chl_src.read(1)

                for water_id in range(1, n + 1):
                    pixels = (labeled == water_id)
                    area_m2 = pixels.sum() * 100  # 10m x 10m pixel = 100m² at 1024x1024
                    geom_json = next(
                        shapes(
                            pixels.astype(np.uint8),
                            mask=pixels,
                            transform=src.transform,
                        )
                    )[0]
                    mean_turbidity = np.mean(turbidity[pixels]) if pixels.any() else 0
                    mean_chlorophyll = np.mean(chlorophyll[pixels]) if pixels.any() else 0

                    # Validate data ranges
                    if mean_turbidity < -1 or mean_turbidity > 1:
                        logger.warning(f"Tile {tile_id}_{water_id}: Invalid turbidity value: {mean_turbidity}")
                        mean_turbidity = max(0, min(mean_turbidity, 1))  # Clamp to [0, 1]
                    if mean_chlorophyll < 0 or mean_chlorophyll > 100:
                        logger.warning(f"Tile {tile_id}_{water_id}: Invalid chlorophyll value: {mean_chlorophyll}")
                        mean_chlorophyll = max(0, min(mean_chlorophyll, 100))  # Clamp to [0, 100]

                    results.append(dict(
                        id=f"{tile_id}_{water_id}",
                        area_m2=area_m2,
                        geometry=geom_json,
                        turbidity=mean_turbidity,
                        chlorophyll=mean_chlorophyll
                    ))

        logger.info(f"Tile {tile_id}: Processed {n} water bodies in {time.time() - start_time:.2f} seconds")
        return results, n
    except Exception as e:
        logger.error(f"Tile {tile_id}: Failed to process TIFFs: {str(e)}")
        return [], 0

def fetch_water_data():
    """Fetch and process water data for all tiles."""
    start_time = time.time()
    logger.info("Starting water data processing")
    all_results = []
    total_water_bodies = 0

    # Process tiles in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        tile_results = list(executor.map(lambda x: fetch_water_data_for_tile(*x),
                                         [(tile_bbox, i) for i, tile_bbox in enumerate(tiles, 1)]))

    for tile_result, tile_n in tile_results:
        all_results.extend(tile_result)
        total_water_bodies += tile_n

    summary = dict(
        total_water_bodies=total_water_bodies,
        total_area_m2=sum(r["area_m2"] for r in all_results),
        avg_turbidity=np.mean([r["turbidity"] for r in all_results]) if all_results else 0,
        avg_chlorophyll=np.mean([r["chlorophyll"] for r in all_results]) if all_results else 0
    )

    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return summary, all_results

if __name__ == "__main__":
    os.makedirs("data/sentinel", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    summary, results = fetch_water_data()
    logger.info("Water data processing complete")