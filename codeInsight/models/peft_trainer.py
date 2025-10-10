import torch
from peft import get_peft_model, LoraConfig
from transformers import PreTrainedModel
from codeInsight.utils.config import load_config
from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
import sys

def peft_trainer(base_model : PreTrainedModel):
    try:
        config = load_config("config/model.yaml")
        
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(base_model, lora_config)
        logging.info('PEFT model loaded')
        logging.info("Total Trainable parameters in the model : {}".format(model.print_trainable_parameters()))
        
        return model
    
    except Exception as e:
        logging.error("Error loading PEFT model")
        raise ExceptionHandle(e, sys)