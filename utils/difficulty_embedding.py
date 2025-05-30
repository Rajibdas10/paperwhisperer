from sentence_transformers import SentenceTransformer, util

# Load once and reuse
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Reference anchor texts for difficulty
easy_ref = "This paper is written in very simple language with short sentences and clear structure."
hard_ref = "This paper contains complex terminology, long sentences, and requires deep domain knowledge."

easy_embed = model.encode(easy_ref, convert_to_tensor=True)
hard_embed = model.encode(hard_ref, convert_to_tensor=True)

def get_semantic_difficulty_score(text):
    paper_embed = model.encode(text, convert_to_tensor=True)
    
    sim_easy = util.cos_sim(paper_embed, easy_embed).item()
    sim_hard = util.cos_sim(paper_embed, hard_embed).item()
    
    # Higher means harder
    difficulty_score = round(sim_hard - sim_easy, 3)
    
    if difficulty_score > 0.1:
        label = "Hard"
    elif difficulty_score > -0.1:
        label = "Moderate"
    else:
        label = "Easy"
    
    return {
        "semantic_score": difficulty_score,
        "semantic_level": label
    }
