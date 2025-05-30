import re
import string 
import textstat
import nltk
from collections import Counter
from utils.difficulty_embedding import get_semantic_difficulty_score
from nltk.corpus import words

english_vocab = set(words.words())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

def get_jargon_density(text):
    words_list = text.split()
    total = len(words_list)
    non_dict_words = [w for w in words_list if w.lower() not in english_vocab]
    jargon_ratio = round(len(non_dict_words)/total, 3) if total > 0 else 0
    return jargon_ratio

def get_vocabulary_richness(text):
    words = text.split()
    total_words = len(words)
    unique_words = len(set(words))
    return round(unique_words/total_words, 3) if total_words > 0 else 0

def estimate_difficulty(text):
    cleaned_text = clean_text(text)
    
    sentences = re.split(r'[.!?]', text)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.strip())> 0]
    avg_sentence_length = round(sum(sentence_lengths) / len(sentence_lengths), 2)
    
    vocab_richness = get_vocabulary_richness(cleaned_text)    
    flesch_score = textstat.flesch_reading_ease(text)
    jargon_density = get_jargon_density(cleaned_text)
    semantic_result = get_semantic_difficulty_score(text)
    
    final_score = {
        "flesch_score": flesch_score,
        "avg_sentence_length": avg_sentence_length,
        "vocab_richness": vocab_richness,
        "jargon_density": jargon_density,
        "semantic_score": semantic_result["semantic_score"],
        "difficulty_level": semantic_result["semantic_level"]  # final decision based on transformer logic
    }
    
    return final_score
