
import os
import tomli
from datetime import datetime, timedelta
import json
from sentinelhub import (
    SHConfig, SentinelHubCatalog, SentinelHubRequest,
    MimeType, CRS, BBox, DataCollection
)
import rasterio
import numpy as np
from scipy.ndimage import label
from rasterio.features import shapes
from rasterio.transform import from_bounds
import time

# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)

# Setup Sentinel Hub credentials
sh_config = SHConfig()
sh_config.sh_client_id = config_data["sentinelhub"]["client_id"]
sh_config.sh_client_secret = config_data["sentinelhub"]["client_secret"]
sh_config.instance_id = config_data["sentinelhub"]["instance_id"]

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

def fetch_water_data_for_tile(bbox, tile_id):
    start_time = time.time()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    # Check for cached TIFFs
    tile_folder = os.path.join(data_folder, f"tile_{tile_id}")
    mask_tiff = os.path.join(tile_folder, "water_mask.tif")
    turbidity_tiff = os.path.join(tile_folder, "turbidity.tif")
    chlorophyll_tiff = os.path.join(tile_folder, "chlorophyll.tif")
    if all(os.path.isfile(f) for f in [mask_tiff, turbidity_tiff, chlorophyll_tiff]):
        print(f"Tile {tile_id}: Using cached TIFFs")
    else:
        catalog = SentinelHubCatalog(sh_config)
        search = catalog.search(
            collection='sentinel-2-l2a',
            datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            limit=1,
            bbox=bbox,
            filter=f"eo:cloud_cover < {config_data['sentinelhub']['max_cloud']}"
        )
        results_list = list(search)
        if not results_list:
            print(f"No imagery found for tile {tile_id}")
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
            size=(1024, 1024),  # Maintain 1024x1024 resolution
            config=sh_config,
            data_folder=tile_folder
        )
        data = request.get_data(save_data=True)
        print(f"Tile {tile_id} get_data output: {data}")

        # Ensure data is a list with a dictionary
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError(f"Tile {tile_id}: Expected a list of dictionaries, got: {type(data)}")

        data_dict = data[0]
        if isinstance(data_dict.get("water_mask.tif"), np.ndarray):
            print(f"Tile {tile_id}: get_data returned arrays; saving as TIFFs")
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

    # Verify file paths
    for tiff_file in [mask_tiff, turbidity_tiff, chlorophyll_tiff]:
        if not os.path.isfile(tiff_file):
            raise FileNotFoundError(f"Tile {tile_id}: TIFF file not found: {tiff_file}")

    # Process TIFFs
    results = []
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
                results.append(dict(
                    id=f"{tile_id}_{water_id}",
                    area_m2=area_m2,
                    geometry=geom_json,
                    turbidity=mean_turbidity,
                    chlorophyll=mean_chlorophyll
                ))

    print(f"Tile {tile_id} processed in {time.time() - start_time:.2f} seconds")
    return results, n

def fetch_water_data():
    start_time = time.time()
    all_results = []
    total_water_bodies = 0

    # Process each tile
    for i, tile_bbox in enumerate(tiles, 1):
        tile_results, tile_n = fetch_water_data_for_tile(tile_bbox, i)
        all_results.extend(tile_results)
        total_water_bodies += tile_n

    global results
    results = all_results
    global summary
    summary = dict(
        total_water_bodies=total_water_bodies,
        total_area_m2=sum(r["area_m2"] for r in results),
        avg_turbidity=np.mean([r["turbidity"] for r in results]) if results else 0,
        avg_chlorophyll=np.mean([r["chlorophyll"] for r in results]) if results else 0,
        water_bodies=results
    )
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return summary

if __name__ == "__main__":
    summary = fetch_water_data()

    # Save to JSON
    os.makedirs("output", exist_ok=True)
    with open("output/water_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Water body data saved to output/water_summary.json")

# Expose summary and results for import
summary = None
results = None

def initialize_module_vars():
    global summary, results
    summary = fetch_water_data()

initialize_module_vars()
