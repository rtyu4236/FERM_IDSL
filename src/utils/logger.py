import logging
import os
import sys
from datetime import datetime
from config import settings as config

def setup_logger():
    """Sets up a logger for use throughout the project.

    Logs are set to INFO level and above, and are output to both the console and a log file.
    Log files are saved in the format `output/backtest_log_YYYY-MM-DD_HHMMSS.log`.
    """
    # Remove existing handlers from the root logger (to prevent duplicate output)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    os.makedirs(os.path.join(config.OUTPUT_DIR, log_name), exist_ok=True)
    
    log_filename = f"backtest_log_{log_name}.log"
    log_filepath = os.path.join(config.OUTPUT_DIR, log_name, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.info("Logger setup complete.")
    logger.info(f"Log file created at: {log_filepath}")
    # Attach log identifiers/paths as attributes on the logger object
    # Access via `logger.LOG_NAME`, `logger.LOG_DIR`, `logger.LOG_FILEPATH`
    logger.LOG_NAME = log_name
    logger.LOG_DIR = os.path.join(config.OUTPUT_DIR, log_name)
    logger.LOG_FILEPATH = log_filepath
    return logger

# Set up the logger when the module is loaded
logger = setup_logger()