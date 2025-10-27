import os
import sys
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from codeInsight.utils.config import load_config
from codeInsight.data.dataset_builder import DatasetBuilder
from codeInsight.models.model_loader import load_model_and_tokenizer
from codeInsight.models.peft_trainer import ModelTrainer
from codeInsight.evaluation.evaluator import compute_metrics
from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle

class TrainingPipeline:
    def __init__(self, config_path: str = "config/model.yaml"):
        self.config = load_config(config_path)
        self.wandb_key = os.getenv('WANDB_API_KEY')
        self.gf_token = os.getenv('HF_TOKEN')
        
    def _wandb_login(self):
        try:
            if self.wandb_key:
                wandb.login(key=self.wandb_key)
                wandb.init(project=self.config['wandb']['project_name'])
                logging.info("WandB login successful.")
            else:
                raise ValueError('WANDB_API_KEY is not set')
            
        except Exception as e:
            logging.error("Failed to login to WandB")
            raise ExceptionHandle(e, sys)
    
    def run_training(self):
        try:
            if self.config['training']['report_to'] == "wandb":
                self._wandb_login()
            
            model, tokenizer = load_model_and_tokenizer(self.config['model'])
            
            dataset_builder = DatasetBuilder(self.config, tokenizer)
            tokenized_datasets = dataset_builder.get_tokenized_dataset()
            
            trainer = ModelTrainer(
                model=model,
                tokenizer=tokenizer,
                datasets=tokenized_datasets,
                compute_metrics_fn=compute_metrics,
                config=self.config
            )
            
            trainer.train()
            logging.info("Model Training Successfull")
            trainer.save_apater()
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise ExceptionHandle(e, sys)
    
    def run_merge_and_push(self):
        try:
            model_config = self.config['model']
            paths_config = self.config['paths']
            logging.info("Starting model merge and push process")
            
            torch.cuda.empty_cache()
            logging.info('Cleaned GPU cache')
            
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config['base_model_id'],
                return_dict=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_id'])
            tokenizer.pad_token = tokenizer.eos_token
            
            logging.info(f"Loading adapter from {paths_config['adapter_save_dir']}")
            model_to_merge = PeftModel.from_pretrained(
                base_model, 
                paths_config['adapter_save_dir']
            )
            
            merged_model = model_to_merge.merge_and_unload()
            logging.info("Merge complete.")
            
            repo_id = paths_config['final_model_repo']
            
            logging.info(f"Pushing merged model and tokenizer to Hugging Face Hub: {repo_id}")
            merged_model.push_to_hub(
                repo_id,
                token=self.hf_token,
                check_pr=False
            )
            
            tokenizer.push_to_hub(
                repo_id,
                token=self.hf_token,
                check_pr=False
            )
            
            logging.info("Successfully pushed model and tokenizer to the Hub.")

        except ExceptionHandle as e:
            logging.error("Failed to merge and push model")
            raise ExceptionHandle(e, sys)