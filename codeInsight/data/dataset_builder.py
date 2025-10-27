from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
from datasets import load_dataset
import torch
import sys

class DatasetBuilder:
    def __init__(self, config : dict, tokenizer):
        self.config = config['dataset']
        self.tokenizer = tokenizer
        logging.info("DatasetBuilder initialized")
    
    def _format_example(self, example : dict) -> dict:
        try:
            prompt_template = self.config['prompt_template']
            text = prompt_template.format(
                instruction=example['instruction'],
                output=example['output']
            )
            return {"text" : text}
        
        except Exception as e:
            logging.error("Something is wrong in format_example")
            raise ExceptionHandle(e, sys)
    
    def _tokenize_batch(self, batch : dict) -> dict:
        return self.tokenizer(
            batch['text'],
            padding=True,
            truncation=True,
            return_tensors=None
        )
    
    def get_tokenized_dataset(self) -> dict:
        try:
            dataset_name = self.config['name']
            raw_data = load_dataset(dataset_name)
            logging.info(f"Load dataset: {dataset_name}")
            
            tokenized_dataset = {}
            
            for split in raw_data.keys():
                formatted_data = raw_data[split].map(
                    self._format_example,
                    remove_columns=['instruction', 'output']
                )
                if split=="train":
                    formatted_data = formatted_data.shuffle(seed=self.config['shuffle_seed'])
                    
                tokenized_dataset[split] = formatted_data
            logging.info("All datasets processed and tokenized.")
            return tokenized_dataset
        
        except Exception as e:
            logging.error(f"Failed during dataset preparation: {e}")
            raise ExceptionHandle(e, sys)