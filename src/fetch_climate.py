import os
import datetime as dt
import requests
import xarray as xr
import tomli


# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)


imerg_base = config_data["imerg"]["base_url"]
openmeteo_base = config_data["open_meteo"]["base_url"]


area = config_data["area"]
west, south, east, north = (
    area["west"], area["south"],
    area["east"], area["north"]
)


def build_imerg_filename(now):
    # Example filename: 3B-HHR-L.MS.MRG.
    # 3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.V07B.zip

    date_str = now.strftime("%Y%m%d")
    start_time = "S000000"
    end_time = "E235959"
    version = "V07B"
    filename = (
        f"3B-HHR-L.MS.MRG.3IMERG.{date_str}-{start_time}-"
        f"{end_time}.{version}.zip"
    )
    return filename


def download(url, dest):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


now = dt.datetime.utcnow()
file_name = build_imerg_filename(now)
url = f"{imerg_base}/{file_name}"
local_zip = download(url, dest="data/imerg_latest.zip")


# Assuming the zip is extracted manually or by other means to imerg_latest.nc4
# For now, we assume the file exists
ds = xr.open_dataset("imerg_latest.nc4")
global precip
precip = ds['precipitationCal'].sel(
    lon=slice(west, east),
    lat=slice(south, north)
).mean().item()   # mm/hr


params = dict(
    latitude=-0.0236, longitude=37.9062,  # Kenya centroid
    hourly=['temperature_2m', 'relative_humidity_2m',
            'evapotranspiration', 'soil_moisture_0_1cm',
            'wind_speed_10m', 'precipitation']
)
resp = requests.get(openmeteo_base, params=params, timeout=30)
resp.raise_for_status()
weather_json = resp.json()
latest_idx = -1
global hourly
hourly = {k: v[latest_idx] for k, v in weather_json['hourly'].items()}
