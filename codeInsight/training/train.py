import sys
from codeInsight.pipeline.training_pipeline import TrainingPipeline
from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging

def start_training():
    try:
        logging.info("Initializing Training Pipeline...")
        pipeline = TrainingPipeline()
        
        logging.info("Starting Model Training")
        pipeline.run_training()
        
        logging.info("Start Model Merge and Push")
        pipeline.run_merge_and_push()
        
        logging.info("Pipeline Complet")
        
    except Exception as e:
        logging.error("Pipeline failed")
        raise ExceptionHandle(e, sys)


if __name__ == "__main__":
    start_training()