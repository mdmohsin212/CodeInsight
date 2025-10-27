from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging
from pathlib import Path
import sys
import yaml

def load_config(config_path : Path = Path("config/model.yaml")) -> dict:
    try:
        with open(config_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.info(f"Config loaded from {config_path}")
        return config
    
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise ExceptionHandle(e, sys)
    
    except Exception as e:
        logging.error(f"Error loading config from {config_path}")
        raise ExceptionHandle(e, sys)