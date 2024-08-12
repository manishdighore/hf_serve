from model import VisionModel,SpeechModel
from transformers import pipeline
from PIL import Image

class VisionInference:
    def __init__(self):
        model_obj = VisionModel()
        self.model = model_obj.load_model()
        self.processor = model_obj.load_processor()

    def predict(self, image_path: str, instruction: str) -> str:
        image = Image.open(image_path)

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{instruction}"}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 2048,
            "temperature": 0.0
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response
    

class SpeechInference:
    def __init__(self):
        model_obj = SpeechModel()
        self.model = model_obj.load_model()
        self.processor = model_obj.load_processor()
        self.pipeline = pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    return_timestamps=True,
                    # torch_dtype=torch_dtype,
                    # device=device,
                )

    def predict(self, audio_path: str) -> str:
        
        transcription = self.pipeline(audio_path)

        return transcription
    

class SQLInference:
    def __init__(self):
        model_obj = SQLModel()
        self.model = model_obj.load_model()
        self.tokenizer = model_obj.load_tokenizer()

    def predict(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to('cuda'), max_new_tokens=256)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response