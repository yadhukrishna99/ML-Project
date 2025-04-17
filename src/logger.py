import logging
import os
from datetime import datetime

LOG_FILE = F"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ## Log file will be created in this name
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)  ## Log file will be saved in the current working dir with the name as logs followed by LOG_File name.
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)

logger = logging.getLogger(__name__)