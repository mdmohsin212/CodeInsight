import logging
import os
from datetime import datetime

dir = "tmp/logs"
os.makedirs(dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOF_PATH = os.path.join(dir, LOG_FILE)

file_handler = logging.FileHandler(LOF_PATH)
console_handler = logging.StreamHandler()

log_format = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
formetter = logging.Formatter(log_format)

file_handler.setFormatter(formetter)
console_handler.setFormatter(formetter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler],
)