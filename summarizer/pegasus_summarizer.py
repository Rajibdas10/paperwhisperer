from transformers import pipeline
from typing import List

class PegasusSummarizer:
    def __init__(self):
        self.pipeline = pipeline("summarization", model="google/pegasus-xsum")

    def summarize(self, text: str, max_chunk_tokens: int = 1024) -> str:
        chunks = [text[i:i+max_chunk_tokens] for i in range(0, len(text), max_chunk_tokens)]
        summaries: List[str] = []
        
        for chunk in chunks:
            summary = self.pipeline(chunk, max_length=128, min_length=30, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
            
        return "\n\n".join(summaries)
