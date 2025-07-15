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

# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)

# Setup Sentinel Hub credentials
sh_config = SHConfig()
sh_config.sh_client_id = config_data["sentinelhub"]["client_id"]
sh_config.sh_client_secret = config_data["sentinelhub"]["client_secret"]
sh_config.instance_id = config_data["sentinelhub"]["instance_id"]

# Define bounding box for Kenya (or your AOI)
area = config_data["area"]
west, south, east, north = (
    area["west"], area["south"],
    area["east"], area["north"]
)
bbox = BBox((west, south, east, north), crs=CRS.WGS84)


def fetch_water_data():
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)

    catalog = SentinelHubCatalog(sh_config)
    search = catalog.search(
        collection='sentinel-2-l2a',
        date=(start_date.isoformat() + "Z", end_date.isoformat() + "Z"),
        limit=1,
        bbox=bbox,
        filter=(
            f"(eo:cloud_cover < {config_data['sentinelhub']['max_cloud']}) AND "
            f"(datetime >= '{start_date.isoformat()}Z' AND datetime <= '{end_date.isoformat()}Z')"
        ),
        sortby=[{'property': 'datetime', 'direction': 'desc'}]
    )
    results_list = list(search)
    if not results_list:
        raise RuntimeError
    ("No satellite imagery found for given area and time range.")

    # latest = results_list[0]  # Unused variable removed

    # Create data request
    with open("src/evalscript_wbm.js") as f:
        evalscript = f.read()

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                "default", MimeType.TIFF
            )
        ],
        bbox=bbox,
        size=(2048, 2048),  # ~10 m/pixel
        config=sh_config
    )
    mask_tiff = request.get_data(save_data=True)[0]

    # Process TIFF
    with rasterio.open(mask_tiff) as src:
        mask = src.read(1) == 1  # Water = 1
        structure = np.ones((3, 3), dtype=int)
        labeled, n = label(mask, structure=structure)

        global results
        results = []
        for water_id in range(1, n + 1):
            pixels = (labeled == water_id)
            area_m2 = pixels.sum() * 100  # 10m x 10m pixel = 100m²
            geom_json = next(
                shapes(
                    pixels.astype(np.uint8),
                    mask=pixels,
                    transform=src.transform,
                )
            )[0]
            results.append(dict(
                id=water_id,
                area_m2=area_m2,
                geometry=geom_json
            ))

        global summary
        summary = dict(
            total_water_bodies=n,
            total_area_m2=sum(r["area_m2"] for r in results),
            water_bodies=results
        )
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
