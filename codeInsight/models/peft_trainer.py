import sys
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from codeInsight.logger import logging
from codeInsight.exception import ExceptionHandle

class ModelTrainer:
    def __init__(self, model, tokenizer, datasets: dict, compute_metrics_fn, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.compute_metrics_fn = compute_metrics_fn
        self.lora_config = config['lora']
        self.training_config = config['training']
        self.paths_config = config['paths']
        
        self.trainer = self._setup_trainer()
        logging.info("ModelTrainer initialized.")
        
    def _peft_model_setup(self):
        try:
            logging.info('Setting up PEFT LoRA model')
            lora_config = LoraConfig(
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['lora_alpha'],
                target_modules=self.lora_config['target_modules'],
                lora_dropout=self.lora_config['lora_dropout'],
                bias=self.lora_config['bias'],
                task_type=self.lora_config['task_type']
            )
            
            peft_model = get_peft_model(self.model, lora_config)
            logging.info("PEFT model created successfully.")
            peft_model.print_trainable_parameters()
            
            return peft_model
        
        except Exception as e:
            logging.error("Failed to setup PEFT model")
            raise ExceptionHandle(e, sys)

    def _get_training_args(self) -> TrainingArguments:
        try:
            return TrainingArguments(
                output_dir=self.paths_config['output_dir'],
                per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
                gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
                num_train_epochs=self.training_config['num_train_epochs'],
                learning_rate=self.training_config['learning_rate'],
                warmup_ratio=self.training_config['warmup_ratio'],
                warmup_steps=self.training_config['warmup_steps'],
                bf16=self.training_config['bf16'],
                gradient_checkpointing=self.training_config['gradient_checkpointing'],
                fp16=self.training_config['fp16'],
                weight_decay=self.training_config['weight_decay'],
                logging_steps=self.training_config['logging_steps'],
                eval_steps=self.training_config['eval_steps'],
                save_steps=self.training_config['save_steps'],
                evaluation_strategy=self.training_config['eval_strategy'],
                save_strategy=self.training_config['save_strategy'],
                save_total_limit=self.training_config['save_total_limit'],
                load_best_model_at_end=self.training_config['load_best_model_at_end'],
                metric_for_best_model=self.training_config['metric_for_best_model'],
                greater_is_better=self.training_config['greater_is_better'],
                prediction_loss_only=self.training_config['prediction_loss_only'],
                report_to=self.training_config['report_to']
            )
            
        except Exception as e:
            logging.error("Failed to create TrainingArguments")
            raise ExceptionHandle(e, sys)
    
    def _setup_trainer(self) -> SFTTrainer:
        logging.info("Initializing SFTTrainer")
        peft_model = self._peft_model_setup()
        training_args = self._get_training_args()
        
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['val'],
            args=training_args,
            compute_metrics=self.compute_metrics_fn
        )
        logging.info("SFTTrainer initialized successfully.")
        return trainer
    
    def save_apater(self):
        try:
            adapter_path = self.paths_config['adapter_save_dir']
            self.trainer.model.save_pretrained(adapter_path)
            logging.info(f"LoRA adapter saved successfully to {adapter_path}")
            
        except Exception as e:
            logging.error("Failed to save LoRA adapter")
            raise ExceptionHandle(e, sys)