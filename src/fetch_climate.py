import os
import datetime as dt
import requests
import xarray as xr
import tomli
import zipfile
from pydap.client import open_url
from pydap.cas.urs import setup_session

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

# Earthdata credentials
EARTHDATA_USERNAME = "alvin390"
EARTHDATA_PASSWORD = "AKariuki@2006"

def build_imerg_filename(date, extension="HDF5"):
    date_str = date.strftime("%Y%m%d")
    start_time = "S000000"
    end_time = "E235959"
    version = "V07B"
    filename = (
        f"3B-HHR-L.MS.MRG.3IMERG.{date_str}-{start_time}-"
        f"{end_time}.{version}.{extension}"
    )
    return filename

def download(url, dest, auth=(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)):
    response = requests.get(url, stream=True, auth=auth, timeout=60)
    response.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return os.path.join(extract_to, zip_ref.namelist()[0])
    except zipfile.BadZipFile:
        return zip_path

def fetch_imerg_data():
    now = dt.datetime.utcnow()
    date = now - dt.timedelta(days=1)
    max_attempts = 14  # Try up to 14 previous days
    attempt = 0
    extensions = ["HDF5", "HDF5.nc4"]

    # Check for cached files
    for i in range(max_attempts):
        for ext in extensions:
            cached_file = f"data/imerg_latest_{date.strftime('%Y%m%d')}.{ext.lower()}"
            if os.path.isfile(cached_file):
                print(f"Using cached IMERG file: {cached_file}")
                return cached_file
        date -= dt.timedelta(days=1)

    # Try HTTPS download
    date = now - dt.timedelta(days=1)
    while attempt < max_attempts:
        for ext in extensions:
            file_name = build_imerg_filename(date, ext)
            year = date.strftime("%Y")
            day_of_year = date.strftime("%j")
            url = f"{imerg_base}/{year}/{day_of_year}/{file_name}"
            print(f"Attempting to download: {url}")
            try:
                local_file = download(url, dest=f"data/imerg_latest_{date.strftime('%Y%m%d')}.{ext.lower()}")
                local_nc4 = extract_zip(local_file, "data")
                return local_nc4
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"File not found for {date.strftime('%Y%m%d')} ({ext}).")
                else:
                    raise
        date -= dt.timedelta(days=1)
        attempt += 1

    # Try OPeNDAP with pydap
    date = now - dt.timedelta(days=1)
    attempt = 0
    session = None
    while attempt < max_attempts:
        file_name = build_imerg_filename(date, "HDF5.nc4")
        cached_file = f"data/imerg_latest_{date.strftime('%Y%m%d')}.hdf5.nc4"
        if os.path.isfile(cached_file):
            print(f"Using cached OPeNDAP file: {cached_file}")
            return cached_file
        year = date.strftime("%Y")
        day_of_year = date.strftime("%j")
        url = f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/{year}/{day_of_year}/{file_name}"
        print(f"Attempting OPeNDAP: {url}")
        try:
            session = setup_session(EARTHDATA_USERNAME, EARTHDATA_PASSWORD, check_url=url)
            dataset = open_url(url, session=session)
            local_file = f"data/imerg_latest_{date.strftime('%Y%m%d')}.hdf5.nc4"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            with open(local_file, "wb") as f:
                f.write(dataset['precipitationCal'][:].data.tobytes())  # Save raw data
            return local_file
        except Exception as e:
            print(f"OPeNDAP failed for {date.strftime('%Y%m%d')}: {str(e)}")
        finally:
            if session is not None:  # Only close session if it was created
                session.close()
        date -= dt.timedelta(days=1)
        attempt += 1

    print("No IMERG data found; proceeding with partial data.")
    return None  # Fallback to allow pipeline to continue

# Process IMERG data
local_nc4 = fetch_imerg_data()
global precip
if local_nc4:
    try:
        ds = xr.open_dataset(local_nc4)
        precip = ds['precipitationCal'].sel(
            lon=slice(west, east),
            lat=slice(south, north)
        ).mean().item()  # mm/hr
    except Exception as e:
        print(f"Failed to process IMERG data: {str(e)}")
        precip = None
else:
    precip = None

# Fetch Open-Meteo data
params = dict(
    latitude=-0.0236, longitude=37.9062,  # Kenya centroid
    hourly=['temperature_2m', 'relative_humidity_2m',
            'evapotranspiration', 'soil_moisture_0_1cm',
            'wind_speed_10m', 'precipitation']
)
try:
    resp = requests.get(openmeteo_base, params=params, timeout=30)
    resp.raise_for_status()
    weather_json = resp.json()
    latest_idx = -1
    global hourly
    hourly = {k: v[latest_idx] for k, v in weather_json['hourly'].items()}
except Exception as e:
    print(f"Failed to fetch Open-Meteo data: {str(e)}")
    hourly = None
