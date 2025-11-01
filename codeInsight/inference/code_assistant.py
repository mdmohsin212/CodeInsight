import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from codeInsight.utils.config import load_config
from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging

class CodeAssistant:
    def __init__(self, config_path="config/model.yaml"):
        try:
            self.config = load_config(config_path)
            self.dataset_config = self.config['dataset']
            model_repo = self.config['model']['final_model_repo']
            logging.info(f"Initializing CodeAssistant with model from: {model_repo}")
                            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_repo,
                device_map="auto",
                torch_dtype=torch.bfloat16, 
                trust_remote_code=False
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo
            )
            self.model.eval()
            self.model.config.use_cache = True
            
            logging.info("CodeAssistant initialized successfully.")
            
        except Exception as e:
            logging.error("Failed to initialize CodeAssistant")
            raise ExceptionHandle(e, sys)
    
    def _formet_prompt(self, prompt : str) -> str:
        return f"{self.dataset_config['SYSTEM_PROMPT']}{self.dataset_config['USER_TOKEN']}{prompt}{self.dataset_config['END_TOKEN']}\n\n{self.dataset_config['ASSISTANT_TOKEN']}"

    def generate(self, prompt : str, max_length : int = 512, temperature: float = 0.1, top_p : float =0.80) -> str:
        try:
            input_text = self._formet_prompt(prompt)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(self.dataset_config['END_TOKEN']),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if self.dataset_config['ASSISTANT_TOKEN'] in generated_text:
                generated_code = generated_text.split(self.dataset_config['ASSISTANT_TOKEN'])[1].strip()
                if self.dataset_config['END_TOKEN'] in generated_code:
                    generated_code = generated_code.split(self.dataset_config['END_TOKEN'])[0].strip()
            else:
                generated_code = generated_text
                    
            logging.info("Response generated successfully.")
            return generated_code
            
        except Exception as e:
            logging.error("Failed during code generation")
            raise ExceptionHandle(e, sys)