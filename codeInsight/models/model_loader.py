import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle
import sys

def load_model_and_tokenizer(config : dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        model_id = config['base_model_id']
        quant_config = config['quantization']
        logging.info(f"Loading base model: {model_id}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=quant_config['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=config['attn_implementation']
        )
        logging.info("Base model loaded successfully with 4-bit quantization.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Tokenizer loaded successfully.")
        
        return model, tokenizer
    
    except Exception as e:
        logging.error("Failed to load model or tokenizer")
        raise ExceptionHandle(e, sys)