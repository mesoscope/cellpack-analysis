import logging
import logging.config
from pathlib import Path

log_file_path = Path(__file__).parents[1] / "logging.conf"
logging.config.fileConfig(log_file_path, disable_existing_loggers=True)
log = logging.getLogger()
