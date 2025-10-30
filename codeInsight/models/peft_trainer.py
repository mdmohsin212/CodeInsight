import sys
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
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
    
    def _get_target_module(self, model) -> list:
        try:
            logging.info('Start Finding LoRA target module')
            candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            present = set()
            for name, module in model.named_modules():
                for cand in candidates:
                    if name.endswith(cand):
                        present.add(cand)
            return list(present) if present else ["q_proj", "v_proj"]
        
        except Exception as e:
            logging.error(f"Something is wrong here")
            raise ExceptionHandle(e, sys)
        
    def _peft_model_setup(self):
        try:
            logging.info('Setting up PEFT LoRA model')
            lora_config = LoraConfig(
                r=self.lora_config['r'],
                lora_alpha=self.lora_config['lora_alpha'],
                target_modules=self._get_target_module(self.model),
                lora_dropout=self.lora_config['lora_dropout'],
                bias=self.lora_config['bias'],
                task_type=self.lora_config['task_type'],
                use_rslora=self.lora_config['use_rslora']
            )
            
            peft_model = get_peft_model(self.model, lora_config)
            logging.info("PEFT model created successfully.")
            peft_model.print_trainable_parameters()
            
            return peft_model
        
        except Exception as e:
            logging.error("Failed to setup PEFT model")
            raise ExceptionHandle(e, sys)

    def _get_training_args(self) -> SFTConfig:
        try:
            return SFTConfig(
                output_dir=self.paths_config['output_dir'],
                per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
                per_device_eval_batch_siz=self.training_config['per_device_eval_batch_size'],
                gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
                num_train_epochs=self.training_config['num_train_epochs'],
                learning_rate=self.training_config['learning_rate'],
                warmup_ratio=self.training_config['warmup_ratio'],
                warmup_steps=self.training_config['warmup_steps'],
                bf16=self.training_config['bf16'],
                tf32=self.training_config['tf32'],
                fp16=self.training_config['fp16'],
                lr_scheduler_type=self.training_config['lr_scheduler_type'],
                optim=self.training_config['optim'],
                gradient_checkpointing=self.training_config['gradient_checkpointing'],
                gradient_checkpointing_kwargs=self.training_config['gradient_checkpointing_kwargs'],
                max_grad_norm=self.training_config['max_grad_norm'],
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
                report_to=self.training_config['report_to'],
                dataloader_num_workers=self.training_config['dataloader_num_workers'],
                max_seq_length=self.training_config['max_seq_length'],
                dataset_text_field=self.training_config['dataset_text_field'],
                label_names=self.training_config['label_names'],
                neftune_noise_alpha=self.training_config['neftune_noise_alpha']
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