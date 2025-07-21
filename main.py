import logging
import logging.handlers
import tomli
import os
from src.fetch_satellite import fetch_water_data
from src.fetch_climate import fetch_climate_data
from src.assemble_json import assemble_json

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

if __name__ == "__main__":
    logger.info("Starting pipeline")
    try:
        success = assemble_json()
        if success:
            logger.info("Pipeline completed successfully")
        else:
            logger.error("Pipeline failed; check logs for details")
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")