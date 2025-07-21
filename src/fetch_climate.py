import os
import datetime as dt
import openmeteo_requests
import pandas as pd
from retry_requests import retry
import requests_cache
import numpy as np
import logging
import logging.handlers
import tomli
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
openmeteo_past_days = config_data["open_meteo"]["past_days"]

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

# Setup Open-Meteo client with retries, no caching
retry_session = retry(session=requests_cache.CachedSession('.cache', expire_after=0), retries=3, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_openmeteo_data():
    logger.info("Starting Open-Meteo data fetch")
    params = {
        "latitude": openmeteo_latitude,
        "longitude": openmeteo_longitude,
        "hourly": openmeteo_hourly_variables,
        "minutely_15": openmeteo_hourly_variables,
        "past_days": openmeteo_past_days,
        "forecast_days": openmeteo_forecast_days
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ClimateFetchBot/1.0)"
    }
    try:
        encoded_url = f"{openmeteo_base}?{urllib.parse.urlencode(params)}"
        logger.debug(f"Sending request to: {encoded_url}")
        responses = openmeteo.weather_api(openmeteo_base, params=params)
        response = responses[0]
        logger.debug(f"Open-Meteo response: Coordinates {response.Latitude()}°N {response.Longitude()}°E")

        # Process historical data
        historical = response.Hourly()
        if historical is None or historical.Variables(0) is None:
            logger.warning("No hourly data available in Open-Meteo response")
            return None, None, None
        historical_data = {
            "time": pd.date_range(
                start=pd.to_datetime(historical.Time(), unit="s", utc=True),
                end=pd.to_datetime(historical.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=historical.Interval()),
                inclusive="left"
            ).to_pydatetime().tolist()
        }
        variable_map = {
            "precipitation": 0,
            "temperature_2m": 1,
            "relative_humidity_2m": 2,
            "soil_moisture_0_to_1cm": 3,
            "wind_speed_10m": 4,
            "et0_fao_evapotranspiration": 5
        }
        for key, idx in variable_map.items():
            values = historical.Variables(idx).ValuesAsNumpy()
            if values is None or np.all(np.isnan(values)):
                logger.warning(f"No valid data for {key} in historical response")
                historical_data[key] = [0] * len(historical_data["time"])
            else:
                historical_data[key] = np.where(np.isnan(values), 0, values).tolist()

        if not historical_data["precipitation"] or all(v == 0 for v in historical_data["precipitation"]):
            logger.warning("No valid precipitation data in Open-Meteo historical response")
            historical_data = None
        else:
            logger.info("Successfully fetched Open-Meteo historical data")

        # Process current data (minutely_15, fallback to hourly)
        current_data = None
        minutely = response.Minutely15()
        now = dt.datetime.now(dt.UTC)
        if minutely and minutely.Variables(0):
            minutely_times = pd.date_range(
                start=pd.to_datetime(minutely.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely.Interval()),
                inclusive="left"
            ).to_pydatetime().tolist()
            latest_idx = max(range(len(minutely_times)), key=lambda i: minutely_times[i] if minutely_times[i] <= now else dt.datetime.min.replace(tzinfo=dt.timezone.utc))
            current_data = {
                "time": minutely_times[latest_idx].isoformat(),
                "precipitation": float(minutely.Variables(0).ValuesAsNumpy()[latest_idx] or 0),
                "temperature_2m": float(minutely.Variables(1).ValuesAsNumpy()[latest_idx] or 0),
                "relative_humidity_2m": float(minutely.Variables(2).ValuesAsNumpy()[latest_idx] or 0),
                "soil_moisture_0_to_1cm": float(minutely.Variables(3).ValuesAsNumpy()[latest_idx] or 0),
                "wind_speed_10m": float(minutely.Variables(4).ValuesAsNumpy()[latest_idx] or 0),
                "evapotranspiration": float(minutely.Variables(5).ValuesAsNumpy()[latest_idx] or 0)
            }
            logger.info(f"Successfully fetched Open-Meteo current data (minutely_15) at {current_data['time']}")
        else:
            logger.warning("Minutely_15 data unavailable, falling back to hourly")
            hourly = response.Hourly()
            if hourly and hourly.Variables(0):
                hourly_times = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ).to_pydatetime().tolist()
                latest_idx = max(range(len(hourly_times)), key=lambda i: hourly_times[i] if hourly_times[i] <= now else dt.datetime.min.replace(tzinfo=dt.timezone.utc))
                current_data = {
                    "time": hourly_times[latest_idx].isoformat(),
                    "precipitation": float(hourly.Variables(0).ValuesAsNumpy()[latest_idx] or 0),
                    "temperature_2m": float(hourly.Variables(1).ValuesAsNumpy()[latest_idx] or 0),
                    "relative_humidity_2m": float(hourly.Variables(2).ValuesAsNumpy()[latest_idx] or 0),
                    "soil_moisture_0_to_1cm": float(hourly.Variables(3).ValuesAsNumpy()[latest_idx] or 0),
                    "wind_speed_10m": float(hourly.Variables(4).ValuesAsNumpy()[latest_idx] or 0),
                    "evapotranspiration": float(hourly.Variables(5).ValuesAsNumpy()[latest_idx] or 0)
                }
                logger.info(f"Successfully fetched Open-Meteo current data (hourly) at {current_data['time']}")
            else:
                logger.warning("Hourly data also unavailable")
                current_data = None

        if current_data and any(v is None for v in current_data.values()):
            logger.warning("Invalid Open-Meteo current data: missing values")
            current_data = None

        # Process forecast data
        forecast_data = {
            "time": pd.date_range(
                start=now.replace(hour=0, minute=0, second=0, microsecond=0),
                periods=openmeteo_forecast_days * 24,
                freq="H",
                tz="UTC"
            ).to_pydatetime().tolist()
        }
        for key, idx in variable_map.items():
            values = historical.Variables(idx).ValuesAsNumpy()[-openmeteo_forecast_days * 24:]
            if values is None or np.all(np.isnan(values)):
                logger.warning(f"No valid data for {key} in forecast period")
                forecast_data[key] = [0] * len(forecast_data["time"])
            else:
                forecast_data[key] = np.where(np.isnan(values), 0, values).tolist()

        daily_forecast = []
        for day in range(openmeteo_forecast_days):
            start_idx = day * 24
            end_idx = (day + 1) * 24
            day_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=day + 1)
            daily_data = {
                "date": day_time.strftime("%Y-%m-%d"),
                "precipitation_total_mm": float(sum(forecast_data["precipitation"][start_idx:end_idx]) or 0),
                "temperature_mean_c": float(np.mean(forecast_data["temperature_2m"][start_idx:end_idx]) or 0),
                "relative_humidity_mean_percent": float(np.mean(forecast_data["relative_humidity_2m"][start_idx:end_idx]) or 0),
                "soil_moisture_mean_m3_m3": float(np.mean(forecast_data["soil_moisture_0_to_1cm"][start_idx:end_idx]) or 0),
                "wind_speed_mean_kmh": float(np.mean(forecast_data["wind_speed_10m"][start_idx:end_idx]) or 0),
                "evapotranspiration_mean_mm": float(np.mean(forecast_data["et0_fao_evapotranspiration"][start_idx:end_idx]) or 0)
            }
            daily_forecast.append(daily_data)

        logger.info("Successfully fetched Open-Meteo forecast")
        return historical_data, current_data, daily_forecast
    except Exception as e:
        logger.error(f"Open-Meteo fetch failed: {str(e)}")
        return None, None, None

def fetch_chirps_data():
    date = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
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
                with open(local_file, "wb") as f:
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

def fetch_climate_data():
    logger.info("Starting climate data processing")

    # Fetch Open-Meteo data
    historical, current, forecast = fetch_openmeteo_data()

    # Process precipitation
    precip = None
    if historical and isinstance(historical, dict) and "precipitation" in historical and historical["precipitation"]:
        try:
            precip = float(sum(historical["precipitation"]) / len(historical["precipitation"]) or 0)
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

    return precip, current, forecast

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    precip, hourly, forecast = fetch_climate_data()
    logger.info("Climate data processing complete")