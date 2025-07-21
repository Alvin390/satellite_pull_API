import os
import datetime as dt
import requests
import xarray as xr
import tomli
import numpy as np
import logging
import logging.handlers
from glob import glob

# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)

# Configuration parameters
imerg_base = config_data["imerg"]["base_url"]
imerg_version = config_data["imerg"]["version"]
imerg_file_extension = config_data["imerg"]["file_extension"]
imerg_fallback_days = config_data["imerg"]["fallback_days"]
openmeteo_base = config_data["open_meteo"]["base_url"]
openmeteo_latitude = config_data["open_meteo"]["latitude"]
openmeteo_longitude = config_data["open_meteo"]["longitude"]
openmeteo_hourly_variables = config_data["open_meteo"]["hourly_variables"]
openmeteo_forecast_days = config_data["open_meteo"]["forecast_days"]

area = config_data["area"]
west, south, east, north = (
    area["west"], area["south"],
    area["east"], area["north"]
)

# Earthdata credentials
EARTHDATA_USERNAME = config_data["earthdata"]["username"]
EARTHDATA_PASSWORD = config_data["earthdata"]["password"]

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

def build_imerg_filename(date):
    """Build IMERG filename using config parameters."""
    date_str = date.strftime("%Y%m%d")
    start_time = "S000000"
    end_time = "E235959"
    filename = f"3B-HHR.MS.MRG.3IMERG.{date_str}-{start_time}-{end_time}.{imerg_version}.{imerg_file_extension}"
    return filename

def validate_netcdf(file_path):
    """Validate that a NetCDF file is readable."""
    try:
        with xr.open_dataset(file_path, engine="h5netcdf") as ds:
            if 'precipitationCal' not in ds.variables:
                logger.warning(f"NetCDF file {file_path} missing 'precipitationCal' variable")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to validate NetCDF file {file_path}: {str(e)}")
        return False

def download(url, dest, auth=(EARTHDATA_USERNAME, EARTHDATA_PASSWORD)):
    """Download a file with authentication."""
    try:
        response = requests.get(url, stream=True, auth=auth, timeout=60, headers={'Accept': 'application/octet-stream'})
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded file: {dest}")
        return dest
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Download failed for {url}: {str(e)}")
        return None

def fetch_imerg_data():
    """Fetch IMERG precipitation data for Kenya."""
    logger.info("Starting IMERG data fetch")
    now = dt.datetime.now(dt.UTC)
    date = now - dt.timedelta(days=1)
    max_attempts = imerg_fallback_days
    logger.info("Starting IMERG data fetch")
    today_utc = dt.datetime.now(dt.UTC).date() # Get today's date in UTC

    # Check for cached files
    cached_files = sorted(glob(f"data/imerg_latest_*.{imerg_file_extension}"), key=os.path.getmtime, reverse=True)
    for cached_file in cached_files:
        if validate_netcdf(cached_file):
            logger.info(f"Using cached IMERG file: {cached_file}")
            return cached_file

    # Try HTTPS download
    attempt = 0
    while attempt < max_attempts:
        file_name = build_imerg_filename(date)
        year = date.strftime("%Y")
        day_of_year = date.strftime("%j")
        url = f"{imerg_base}/{year}/{day_of_year}/{file_name}"
        logger.info(f"Attempting HTTPS download: {url}")
        local_file = download(url, dest=f"data/imerg_latest_{date.strftime('%Y%m%d')}.{imerg_file_extension}")
        if local_file and validate_netcdf(local_file):
            return local_file
        date -= dt.timedelta(days=1)
        attempt += 1

    logger.warning("No valid IMERG data found; returning None")
    return None

def fetch_openmeteo_forecast():
    """Fetch 14-day weather forecast from Open-Meteo."""
    logger.info("Starting Open-Meteo forecast fetch")
    try:
        params = dict(
            latitude=openmeteo_latitude,
            longitude=openmeteo_longitude,
            hourly=openmeteo_hourly_variables,
            forecast_days=openmeteo_forecast_days
        )
        resp = requests.get(openmeteo_base, params=params, timeout=30)
        resp.raise_for_status()
        weather_json = resp.json()
        logger.debug(f"Open-Meteo forecast response: {weather_json}")

        # Validate response
        forecast_data = weather_json.get('hourly', {})
        if not forecast_data or 'time' not in forecast_data:
            logger.warning("Invalid Open-Meteo forecast response: missing 'hourly' or 'time'")
            return None

        # Extract forecast data
        times = [dt.datetime.fromisoformat(t.replace('Z', '+00:00')).replace(tzinfo=dt.timezone.utc) for t in forecast_data['time']]
        now = dt.datetime.now(dt.UTC) # This is already timezone-aware UTC
        end_time = now + dt.timedelta(days=openmeteo_forecast_days) # This will also be timezone-aware UTC

        forecast = {key: [] for key in forecast_data if key != 'time'}
        for i, t in enumerate(times):
            if now <= t <= end_time:
                for key in forecast:
                    value = forecast_data[key][i]
                    forecast[key].append(value if value is not None else 0)

        # Summarize as daily values
        daily_forecast = []
        for day in range(openmeteo_forecast_days):
            start_idx = day * 24
            end_idx = (day + 1) * 24
            day_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=day + 1)
            daily_data = {
                'date': day_time.strftime('%Y-%m-%d'),
                'precipitation_total_mm': sum(forecast['precipitation'][start_idx:end_idx]) if forecast['precipitation'] else 0,
                'temperature_mean_c': np.mean(forecast['temperature_2m'][start_idx:end_idx]) if forecast['temperature_2m'] else 0,
                'relative_humidity_mean_percent': np.mean(forecast['relative_humidity_2m'][start_idx:end_idx]) if forecast['relative_humidity_2m'] else 0,
                'evapotranspiration_mean_mm': np.mean(forecast['evapotranspiration'][start_idx:end_idx]) if forecast['evapotranspiration'] else 0,
                'soil_moisture_mean_m3_m3': np.mean(forecast['soil_moisture_0_1cm'][start_idx:end_idx]) if forecast['soil_moisture_0_1cm'] else 0,
                'wind_speed_mean_kmh': np.mean(forecast['wind_speed_10m'][start_idx:end_idx]) if forecast['wind_speed_10m'] else 0
            }
            daily_forecast.append(daily_data)

        logger.info("Successfully fetched Open-Meteo forecast")
        return daily_forecast
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo forecast: {str(e)}")
        return None

def fetch_openmeteo_current():
    """Fetch current weather data from Open-Meteo."""
    logger.info("Starting Open-Meteo current data fetch")
    try:
        params = dict(
            latitude=openmeteo_latitude,
            longitude=openmeteo_longitude,
            hourly=openmeteo_hourly_variables
        )
        resp = requests.get(openmeteo_base, params=params, timeout=30)
        resp.raise_for_status()
        weather_json = resp.json()
        logger.debug(f"Open-Meteo current response: {weather_json}")

        # Validate response
        hourly_data = weather_json.get('hourly', {})
        if not hourly_data or 'time' not in hourly_data:
            logger.warning("Invalid Open-Meteo current response: missing 'hourly' or 'time'")
            return None

        # Select the most recent timestamp
        times = [dt.datetime.fromisoformat(t.replace('Z', '+00:00')) for t in hourly_data['time']]
        now = dt.datetime.now(dt.UTC)
        latest_idx = max(range(len(times)), key=lambda i: times[i] if times[i] <= now else dt.datetime.min)
        hourly = {k: v[latest_idx] if v[latest_idx] is not None else 0 for k, v in hourly_data.items() if k != 'time'}
        hourly['time'] = times[latest_idx].isoformat()

        logger.info("Successfully fetched Open-Meteo current data")
        return hourly
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo current data: {str(e)}")
        return None

def fetch_climate_data():
    """Fetch all climate data (IMERG, Open-Meteo current, and forecast)."""
    logger.info("Starting climate data processing")

    # Fetch IMERG data
    local_nc4 = fetch_imerg_data()
    precip = None
    if local_nc4:
        try:
            with xr.open_dataset(local_nc4, engine="h5netcdf") as ds:
                precip = ds['precipitationCal'].sel(
                    lon=slice(west, east),
                    lat=slice(south, north)
                ).mean().item()  # mm/hr
            logger.info(f"Processed IMERG data: precipitation = {precip} mm/hr")
        except Exception as e:
            logger.error(f"Failed to process IMERG data: {str(e)}")
            precip = None

    # Fetch Open-Meteo data
    forecast = fetch_openmeteo_forecast()
    hourly = fetch_openmeteo_current()

    return precip, hourly, forecast

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    precip, hourly, forecast = fetch_climate_data()
    logger.info("Climate data processing complete")