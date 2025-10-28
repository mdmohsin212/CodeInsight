import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from codeInsight.utils.config import load_config
from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging

class CodeAssistant:
    def __init__(self, config_path="config/model.yaml"):
        try:
            self.config = load_config(config_path)
            model_repo = self.config['model']['final_model_repo']
            logging.info(f"Initializing CodeAssistant with model from: {model_repo}")
                            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_repo,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.prompt_template = self.config['dataset']['prompt_template']
            logging.info("CodeAssistant initialized successfully.")
            
        except Exception as e:
            logging.error("Failed to initialize CodeAssistant")
            raise ExceptionHandle(e, sys)
    
    def _formet_prompt(self, instruction : str) -> str:
        formatted = self.prompt_template.format(
            instruction=instruction,
            output=""
        )
        return formatted.split("Output:")[0] + "Output:\n"

    def generate(self, instruction : str, max_new_token : int = 1024) -> str:
        try:
            prompt = self._formet_prompt(instruction)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_token,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            logging.info("Response generated successfully.")
            return response
            
        except Exception as e:
            logging.error("Failed during code generation")
            raise ExceptionHandle(e, sys)