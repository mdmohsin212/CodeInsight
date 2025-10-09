from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
from codeInsight.utils.config import load_config
from datasets import load_dataset
import os
import sys

def load_dataset():
    try:
        logging.info("Downloading dataset from huggingface")
        config = load_config("config/schema.yaml")
        
        data = load_dataset(config['dataset']['name'])
        train_data = data['train']
        
        subset = train_data.shuffle(seed=42).select(range(20000))
        subset = subset.remove_columns(['input'])
        
        return subset
    
    except Exception as e:
        logging.error("Error loading dataset")
        raise ExceptionHandle(e, sys)