import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = 'logs'
log_file_name = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
max_file_size = 5*1024*1024
backup_count = 3

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),LOG_DIR)

os.makedirs(log_dir,exist_ok=True)
log_file_path = os.path.join(log_dir,log_file_name)

def configure():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file_path,maxBytes=max_file_size,backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure the logger
configure()