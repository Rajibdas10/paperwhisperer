from transformers import AutoModel, AutoTokenizer

model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model cache path:", model.base_model_prefix)
