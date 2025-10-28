import sys
from codeInsight.inference.code_assistant import CodeAssistant
from codeInsight.safety.safety_checker import SafetyChecker
from codeInsight.exception import ExceptionHandle
from codeInsight.logger import logging

class PredictionPipeline:
    def __init__(self, config_path : str = "config/model.yaml"):
        try:
            self.assistant = CodeAssistant(config_path)
            self.safety_checker = SafetyChecker()
            logging.info("Prediction Pipeline initialized successfully.")
            
        except Exception as e:
            logging.error("Failed to initialize PredictionPipeline")
            raise ExceptionHandle(e, sys)
    
    def predict(self, instruction : str) -> str:
        try:
            raw_output = self.assistant.generate(instruction)
            safe_output = self.safety_checker.check_outputs(raw_output)
            
            return safe_output
        
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return "An error occurred while processing your request. Please try again."