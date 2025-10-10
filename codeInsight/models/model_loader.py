import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
import sys

def load_model(model_name : str, device : str = None):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading CLIP model {model_name} on {device}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    except Exception as e:
        logging.error(f"Error loading model {model_name}")
        raise ExceptionHandle(e, sys)