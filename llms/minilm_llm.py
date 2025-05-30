from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

class MiniLMWrapper:
    def __init__(self):
        model_name = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Add a basic summarizer to simulate headline generation
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",device = 0 if torch.cuda.is_available() else -1)

    def answer_question(self, context, question):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        input_ids = inputs["input_ids"][0][answer_start:answer_end]
        answer = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        return answer.strip()

    def generate_headline(self, text):
        """Fakes a 'smart headline' by summarizing the first few sentences."""
        result = self.summarizer(text[:512], max_length=15, min_length=5, do_sample=False)
        return result[0]["summary_text"]
