import re
from transformers import GPT2TokenizerFast
import streamlit as st

# Cache the tokenizer to speed up app reloads
@st.cache_resource
def get_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer = get_tokenizer()

# Simple custom sentence tokenizer that doesn't rely on NLTK's punkt
def custom_sent_tokenize(text):
    """A simple sentence tokenizer that handles common English sentence endings."""
    # This regex splits on periods, question marks, or exclamation points
    # followed by a space and an uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Further refine splits for cases with quotations, parentheses, etc.
    result = []
    for sentence in sentences:
        # Split any longer sentences at newlines to handle paragraph breaks
        parts = sentence.split('\n')
        for part in parts:
            if part.strip():  # only add non-empty parts
                result.append(part.strip())
    
    return result

def split_info_semantic_chunks(text, max_tokens=400):
    # Use our custom tokenizer instead of NLTK's
    sentences = custom_sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        tokens = tokenizer.encode(current_chunk + " " + sentence if current_chunk else sentence)
        if len(tokens) <= max_tokens:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def clean_chunk(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.*$', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()