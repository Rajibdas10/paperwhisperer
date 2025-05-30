from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)

def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
