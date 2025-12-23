import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from codeInsight.utils.config import load_config
import litserve as ls

class LLMApi(ls.LitAPI):
    def setup(self, device, config_path="config/model.yaml"):
        self.config = load_config(config_path)
        self.dataset_config = self.config['dataset']
        model_name = self.config['paths']['final_model_repo']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device != "cpu":
            self.model.to(device)
        self.model.eval()
    
    def _formet_prompt(self, prompt : str) -> str:
        return f"{self.dataset_config['SYSTEM_PROMPT']}{self.dataset_config['USER_TOKEN']}{prompt}{self.dataset_config['END_TOKEN']}\n\n{self.dataset_config['ASSISTANT_TOKEN']}"
    
    def generate(self, prompt : str, max_length : int = 512, temperature: float = 0.2, top_p : float =0.80) -> str:
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
            return {"response": generated_code}
            
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    server = ls.LitServer(LLMApi(), accelerator="auto")
    server.run()