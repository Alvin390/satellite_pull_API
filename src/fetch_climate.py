import os
import datetime as dt
import requests
import xarray as xr
import tomli
import numpy as np
import logging
import logging.handlers
from glob import glob
import rasterio
import urllib.parse

# Load config from TOML
with open("src/config.toml", "rb") as f:
    config_data = tomli.load(f)

openmeteo_base = config_data["open_meteo"]["base_url"]
openmeteo_archive_url = config_data["open_meteo"]["archive_url"]
openmeteo_latitude = config_data["open_meteo"]["latitude"]
openmeteo_longitude = config_data["open_meteo"]["longitude"]
openmeteo_hourly_variables = config_data["open_meteo"]["hourly_variables"]
openmeteo_forecast_days = config_data["open_meteo"]["forecast_days"]

# Setup logging
log_file = config_data["logging"]["log_file"]
log_level = getattr(logging, config_data["logging"]["log_level"])
max_size = config_data["logging"]["max_size"]
backup_count = config_data["logging"]["backup_count"]
os.makedirs(os.path.dirname(log_file), exist_ok=True)
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
handler2 = logging.StreamHandler()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler, handler2]
)
logger = logging.getLogger(__name__)

def fetch_openmeteo_historical():
    url = openmeteo_archive_url
    params = {
        "latitude": openmeteo_latitude,
        "longitude": openmeteo_longitude,
        "start_date": (dt.datetime(2024, 4, 20, tzinfo=dt.timezone.utc)).strftime("%Y-%m-%d"),
        "end_date": (dt.datetime(2024, 5, 20, tzinfo=dt.timezone.utc)).strftime("%Y-%m-%d"),
        "hourly": openmeteo_hourly_variables
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ClimateFetchBot/1.0)"
    }
    try:
        # Log the exact URL for debugging
        encoded_url = f"{url}?{urllib.parse.urlencode(params)}"
        logger.debug(f"Sending request to: {encoded_url}")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()['hourly']
        for key in data:
            if key != 'time':
                data[key] = [v for v in data[key] if v is not None]
        if not data['precipitation']:
            logger.warning("No valid precipitation data in Open-Meteo historical response")
            return None
        return data
    except Exception as e:
        logger.error(f"Open-Meteo historical fetch failed: {str(e)}")
        return None

def fetch_openmeteo_forecast():
    logger.info("Starting Open-Meteo forecast fetch")
    params = {
        "latitude": openmeteo_latitude,
        "longitude": openmeteo_longitude,
        "hourly": openmeteo_hourly_variables,
        "forecast_days": openmeteo_forecast_days
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ClimateFetchBot/1.0)"
    }
    try:
        # Log the exact URL for debugging
        encoded_url = f"{openmeteo_base}?{urllib.parse.urlencode(params)}"
        logger.debug(f"Sending request to: {encoded_url}")
        resp = requests.get(openmeteo_base, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        weather_json = resp.json()
        logger.debug(f"Open-Meteo forecast response: {weather_json}")

        forecast_data = weather_json.get('hourly', {})
        if not forecast_data or 'time' not in forecast_data:
            logger.warning("Invalid Open-Meteo forecast response: missing 'hourly' or 'time'")
            return None

        times = [dt.datetime.fromisoformat(t.replace('Z', '+00:00')).replace(tzinfo=dt.timezone.utc) for t in forecast_data['time']]
        now = dt.datetime.now(dt.UTC)
        end_time = now + dt.timedelta(days=openmeteo_forecast_days)

        forecast = {key: [] for key in forecast_data if key != 'time'}
        for i, t in enumerate(times):
            if now <= t <= end_time:
                for key in forecast:
                    value = forecast_data[key][i]
                    forecast[key].append(value if value is not None else 0)

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
                'soil_moisture_mean_m3_m3': np.mean(forecast['soil_moisture_0_to_10cm'][start_idx:end_idx]) if forecast['soil_moisture_0_to_10cm'] else 0,
                'wind_speed_mean_kmh': np.mean(forecast['wind_speed_10m'][start_idx:end_idx]) if forecast['wind_speed_10m'] else 0
            }
            daily_forecast.append(daily_data)

        logger.info("Successfully fetched Open-Meteo forecast")
        return daily_forecast
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo forecast: {str(e)}")
        return None

def fetch_openmeteo_current():
    logger.info("Starting Open-Meteo current data fetch")
    params = {
        "latitude": openmeteo_latitude,
        "longitude": openmeteo_longitude,
        "hourly": openmeteo_hourly_variables
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ClimateFetchBot/1.0)"
    }
    try:
        # Log the exact URL for debugging
        encoded_url = f"{openmeteo_base}?{urllib.parse.urlencode(params)}"
        logger.debug(f"Sending request to: {encoded_url}")
        resp = requests.get(openmeteo_base, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        weather_json = resp.json()
        logger.debug(f"Open-Meteo current response: {weather_json}")

        hourly_data = weather_json.get('hourly', {})
        if not hourly_data or 'time' not in hourly_data:
            logger.warning("Invalid Open-Meteo current response: missing 'hourly' or 'time'")
            return None

        times = [dt.datetime.fromisoformat(t.replace('Z', '+00:00')).replace(tzinfo=dt.timezone.utc) for t in hourly_data['time']]
        now = dt.datetime.now(dt.UTC)
        latest_idx = max(range(len(times)), key=lambda i: times[i] if times[i] <= now else dt.datetime.min.replace(tzinfo=dt.timezone.utc))
        hourly = {k: v[latest_idx] if v[latest_idx] is not None else 0 for k, v in hourly_data.items() if k != 'time'}
        hourly['time'] = times[latest_idx].isoformat()

        logger.info("Successfully fetched Open-Meteo current data")
        return hourly
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo current data: {str(e)}")
        return None



def fetch_climate_data():
    logger.info("Starting climate data processing")

    # Fetch Open-Meteo historical data for precipitation
    precip = None
    historical = fetch_openmeteo_historical()
    if historical and 'precipitation' in historical and historical['precipitation']:
        try:
            precip = sum(historical['precipitation']) / len(historical['precipitation'])
            logger.info(f"Processed Open-Meteo historical data: precipitation = {precip} mm/hr")
        except Exception as e:
            logger.error(f"Failed to process Open-Meteo historical data: {str(e)}")

    # Fallback to CHIRPS if Open-Meteo fails
    if precip is None:
        precip = fetch_chirps_data()
        if precip is not None:
            logger.info(f"Used CHIRPS fallback: precipitation = {precip} mm/hr")
        else:
            logger.warning("No precipitation data available; using fallback value 0")
            precip = 0

    # Fetch Open-Meteo current and forecast data
    hourly = fetch_openmeteo_current()
    forecast = fetch_openmeteo_forecast()

    return precip, hourly, forecast



def fetch_chirps_data():
    date = dt.datetime(2024, 10, 31, tzinfo=dt.timezone.utc)  # Start with November 30, 2024
    max_attempts = 30
    attempt = 0
    while attempt < max_attempts:
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/{year}/chirps-v2.0.{year}.{month}.{day}.tif"
        local_file = f"data/chirps_{year}{month}{day}.tif"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(local_file, 'wb') as f:
                    f.write(response.content)
                with rasterio.open(local_file) as src:
                    precip = src.read(1).mean() / 24
                logger.info(f"Processed CHIRPS data for {year}-{month}-{day}: precipitation = {precip} mm/hr")
                return precip
            else:
                logger.warning(f"CHIRPS download failed for {year}-{month}-{day}: status {response.status_code}")
        except Exception as e:
            logger.error(f"CHIRPS fetch error for {year}-{month}-{day}: {str(e)}")
        date -= dt.timedelta(days=1)
        attempt += 1
    logger.warning(f"No CHIRPS data available after {max_attempts} attempts")
    return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    precip, hourly, forecast = fetch_climate_data()
    logger.info("Climate data processing complete")