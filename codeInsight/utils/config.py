from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging
import sys
import yaml

def load_config(path : str):
    try:
        with open(path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.info(f"Config loaded from {path}")
        return config
    
    except Exception as e:
        logging.error(f"Error loading config from {path}")
        raise ExceptionHandle(e, sys)