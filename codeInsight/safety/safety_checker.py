from codeInsight.logger import logging
import re

class SafetyChecker:
    def __init__(self):
        logging.info("SafetyChecker initialized.")
    
    def check_outputs(self, text : str) -> str:
        if not text:
            return "No response Generated"

        refusal_phrases = ["I cannot", "I am unable", "As an AI model", "I'm sorry"]
        if any(phrase.lower() in text.lower() for phrase in refusal_phrases):
            logging.warning(f"Model refusal detected: {text}")
            return "I'm sorry, but I cannot fulfill that request."
        
        bad_word_pattern = r"\b(fuck|shit|bitch|asshole|bastard)\b"
        if re.search(bad_word_pattern, text, re.IGNORECASE):
            logging.warning('Bad word detected')
            return "[Content removed due to inappropriate language]"
        
        pii_pattern = [
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{16}\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ]
        for pattern in pii_pattern:
            if re.search(pattern, text):
                logging.warning("PII detected in model output.")
                return "[Sensitive information removed for privacy]"
        
        hallucination_markers = ["According to a study", "In recent news", "As per research"]
        if any(marker.lower() in text.lower() for marker in hallucination_markers):
            logging.info("Potential hallucination detected.")
            
        
        logging.info("Output passed all safety checks.")
        return text