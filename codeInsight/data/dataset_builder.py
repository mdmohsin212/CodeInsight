from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
from datasets import load_dataset
import torch
import sys

class DatasetBuilder:
    def __init__(self, config : dict, tokenizer):
        self.config = config['dataset']
        self.tokenizer = tokenizer
        torch.manual_seed(self.config['shuffle_seed'])
        logging.info("DatasetBuilder initialized")
    
    def _format_example(self, example : dict) -> dict:
        try:
            text = (
                f"{self.config['SYSTEM_PROMPT']}"
                f"{self.config['USER_TOKEN']}{example['instruction']}{self.config['USER_TOKEN']}\n\n"
                f"{self.config['ASSISTANT_TOKEN']}{example['output']}{self.config['USER_TOKEN']}"
            )
            return {"text" : text}
        
        except Exception as e:
            logging.error("Something is wrong in format_example")
            raise ExceptionHandle(e, sys)
    
    def get_dataset(self) -> dict:
        try:
            dataset_name = self.config['name']
            raw_data = load_dataset(dataset_name)
            logging.info(f"Load dataset: {dataset_name}")
            
            for split in raw_data.keys():
                raw_data[split] = raw_data[split].map(self._format_example, num_proc=4)
                raw_data[split] = raw_data[split].remove_columns(['instruction', 'output'])
                    
            raw_data["train"] = raw_data["train"].shuffle(seed=self.config['shuffle_seed'])
            
            logging.info("All datasets processed and tokenized.")
            return raw_data
        
        except Exception as e:
            logging.error(f"Failed during dataset preparation: {e}")
            raise ExceptionHandle(e, sys)