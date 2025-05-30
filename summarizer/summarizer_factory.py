from transformers import pipeline
import torch

def load_summarizer(model_name: str):
    device = 0 if torch.cuda.is_available() else -1

    if model_name == "bart":
        return pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    elif model_name == "pegasus":
        return pipeline("summarization", model="google/pegasus-xsum", device=device)
    elif model_name == "falcon":
        return pipeline("summarization", model="tiiuae/falcon-7b-instruct", device=device)
    elif model_name == "minilm":
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    else:
        raise ValueError(f"Unsupported summarizer model: {model_name}")
