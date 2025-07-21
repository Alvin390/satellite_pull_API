import json
import os
import numpy as np
import datetime
import logging
import logging.handlers
import tomli
from src.fetch_satellite import fetch_water_data
from src.fetch_climate import fetch_climate_data

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

def convert_types(obj):
    """Convert non-serializable types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, (list, dict)):
        return obj
    logger.warning(f"Unexpected type in JSON serialization: {type(obj)}")
    return str(obj)

def validate_data(satellite_data, precip, hourly, forecast):
    valid = True
    if satellite_data is None or not satellite_data[0] or not satellite_data[1]:
        logger.warning("Satellite data is missing or incomplete")
        valid = False
    else:
        summary, results = satellite_data
        required_summary_fields = ["total_water_bodies", "total_area_m2", "avg_turbidity", "avg_chlorophyll"]
        for field in required_summary_fields:
            if field not in summary:
                logger.warning(f"Missing field in satellite summary: {field}")
                valid = False
        if not results:
            logger.warning("No water bodies in satellite results")
            valid = False

    if precip is None:
        logger.warning("Precipitation data is unavailable")
        valid = False
    elif precip == 0:
        logger.info("Using fallback precipitation value of 0")

    if not isinstance(hourly, dict):
        logger.warning(f"Open-Meteo current data is invalid: got {type(hourly)}, expected dict")
        hourly = {"time": datetime.datetime.now(datetime.UTC).isoformat()}
        valid = False
    else:
        required_hourly_fields = ["precipitation", "temperature_2m", "relative_humidity_2m", "soil_moisture_0_to_1cm", "wind_speed_10m", "evapotranspiration", "time"]
        for field in required_hourly_fields:
            if field not in hourly:
                logger.warning(f"Missing field in Open-Meteo current data: {field}, using default 0")
                hourly[field] = 0
                valid = False

    if not isinstance(forecast, list):
        logger.warning(f"Open-Meteo forecast data is invalid: got {type(forecast)}, expected list")
        forecast = []
        valid = False
    else:
        required_forecast_fields = ["date", "precipitation_total_mm", "temperature_mean_c", "relative_humidity_mean_percent", "soil_moisture_mean_m3_m3", "wind_speed_mean_kmh", "evapotranspiration_mean_mm"]
        for day in forecast:
            if not isinstance(day, dict):
                logger.warning(f"Invalid forecast day: got {type(day)}, expected dict")
                valid = False
                continue
            for field in required_forecast_fields:
                if field not in day:
                    logger.warning(f"Missing field in Open-Meteo forecast: {field}, using default 0")
                    day[field] = 0
                    valid = False

    return valid, hourly, forecast

def assemble_json():
    """Assemble and write JSON output."""
    logger.info("Starting JSON assembly")
    try:
        # Fetch satellite data
        satellite_data = fetch_water_data()
        if satellite_data is None:
            logger.error("Failed to fetch satellite data")
            return False
        summary, results = satellite_data

        # Fetch climate data
        precip, hourly, forecast = fetch_climate_data()

        # Validate data and update hourly/forecast if modified
        valid, hourly, forecast = validate_data(satellite_data, precip, hourly, forecast)
        if not valid:
            logger.warning("Data validation failed; writing JSON with available data")

        # Use Open-Meteo current timestamp if available, else current UTC time
        timestamp = hourly.get("time", datetime.datetime.now(datetime.UTC).isoformat())

        # Assemble payload
        payload = dict(
            timestamp=timestamp,
            satellite_water_data=dict(
                summary=summary,
                water_bodies=results
            ),
            climate=dict(
                precipitation_imerg_mm_per_hr=precip if precip is not None else "unavailable",
                open_meteo_current=hourly,
                open_meteo_forecast=forecast
            )
        )

        # Write JSON
        output_file = config_data["output"]["output_file"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(payload, f, indent=2, default=convert_types)
        logger.info(f"Wrote JSON to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to assemble JSON: {str(e)}")
        return False

if __name__ == "__main__":
    success = assemble_json()
    if not success:
        logger.error("JSON assembly failed; check logs for details")