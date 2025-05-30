# phi2_llm.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class Phi2Wrapper:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "microsoft/phi-2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )

    def generate_smart_headline(self, text):
        prompt = f"Generate a smart and catchy headline for the following text:\n\n{text}\n\nHeadline:"
        outputs = self.generator(
            prompt,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        headline = outputs[0]['generated_text']
        return headline.split("Headline:")[-1].strip()
