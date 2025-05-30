# zeyhyr_llm.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ZephyrWrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "HuggingFaceH4/zephyr-7b-alpha"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        ).to(self.device)

    def generate_smart_headline(self, text):
        prompt = f"Generate a smart and catchy headline for the following text:\n\n{text}\n\nHeadline:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return headline.split("Headline:")[-1].strip()
