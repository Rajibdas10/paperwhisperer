from transformers import pipeline
import streamlit as st
import math

# Load your summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # example model

# Constants
MAX_INPUT_TOKENS = 1024   # Depends on model (BART: 1024 tokens, T5-base: 512 tokens, etc.)
SAFE_MARGIN = 50          # Always leave a margin to avoid hidden tokenizer overflow
CHUNK_SIZE = MAX_INPUT_TOKENS - SAFE_MARGIN

def split_text(text, chunk_size):
    """Splits text into manageable chunks without breaking words."""
    words = text.split()
    chunks = []
    current_chunk = []

    current_length = 0
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length <= chunk_size:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_text(text):
    """Handles large text summarization automatically."""
    chunks = split_text(text, CHUNK_SIZE)

    st.write(f"🔵 Splitting into {len(chunks)} chunks...")  # Optional: Remove if you want

    summaries = []
    for idx, chunk in enumerate(chunks):
        st.write(f"🟢 Summarizing chunk {idx+1}/{len(chunks)}...")  # Optional
        summary = summarizer(
            chunk,
            max_length=130,  # adjust based on your needs
            min_length=30,
            do_sample=False
        )[0]["summary_text"]
        summaries.append(summary)

    final_summary = "\n\n".join(summaries)
    return final_summary

# Usage Example inside your app.py
if "pdf_text" in st.session_state and st.session_state["pdf_text"]:
    final_summary = summarize_text(st.session_state["pdf_text"])
    st.write("✅ **Summary:**")
    st.write(final_summary)
