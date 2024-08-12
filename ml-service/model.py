from transformers import AutoProcessor,AutoModelForCausalLM,AutoModelForSpeechSeq2Seq
import torch
from peft import PeftModel

CACHE_DIR = 'models'

class VisionModel:

    def __init__(self):
        self.model_id = "microsoft/Phi-3-vision-128k-instruct" 

    def load_model(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                        device_map="cuda",
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.float16,
                                                        _attn_implementation='flash_attention_2')
        except Exception as e:
            print(e)
            model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                        device_map="cuda",
                                                        trust_remote_code=True,
                                                        torch_dtype=torch.float16,
                                                        _attn_implementation='eager')
        return model

    def load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        return processor
    

class SpeechModel:
    
    def __init__(self):
        self.model_id = "openai/whisper-small" 

    def load_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id, 
                                                          device_map="cuda",
                                                            torch_dtype=torch.float16,
                                                            low_cpu_mem_usage=True,
                                                            use_safetensors=True)
        return model

    def load_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=CACHE_DIR)
        return processor

class SQLModel:

    def __init__(self):
        self.base_model_id = "defog/sqlcoder-7b-2"
        self.adapter_model_id = "manishdighore/intersystems-sql-coder"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="cuda",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model,self.adapter_model_id)
        model = model.to("cuda")
        model.eval()
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        return tokenizer