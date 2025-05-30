import ollama

def generate_cheat_sheet(text,model='mistral'):
    
    prompt = f"""
    
    You are a expert summarizer and note-maker.Your task is to extract a cheat sheet from the following research paper text.
    
    Focus on:
    1. Key definition and terminologies
    2. Important equation or algorithm
    3. Core findings and contributions
    4. Summarized methodology 
    5. Any limitations or future work (if mentioned)
    
    Paper Text:
    {text}
    
    Return the cheat sheet in bullet points, categorized clearly.
    """
    
    response = ollama.chat(
        model = model,
        messages =[
            {"role": "user","content": prompt}
        ]
    )
    
    return response['message']['content']