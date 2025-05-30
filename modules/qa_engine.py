from modules.chunking import split_info_semantic_chunks as chunk_text
from utils.embedding import get_embedding, cosine_similarity
import re
import torch
import gc
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

class MemoryOptimizer:
    """Helper class to manage memory usage for low-resource environments"""
    @staticmethod
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def set_gpu_memory_limits():
        if torch.cuda.is_available():
            # Limit GPU memory usage to 75% of available memory
            torch.cuda.set_per_process_memory_fraction(0.75)

class DistilBERTQAModel:
    """Enhanced QA model using DistilBERT fine-tuned on SQuAD"""
    def __init__(self):
        # Initialize with memory optimization for your hardware
        MemoryOptimizer.set_gpu_memory_limits()
        
        # Load model with memory-optimized settings
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
        
        # Use half precision to save memory
        self.model = DistilBertForQuestionAnswering.from_pretrained(
            'distilbert-base-uncased-distilled-squad',
            torch_dtype=torch.float16
        )
        
        # Move to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Set maximum input length
        self.max_length = 384  # Standard for DistilBERT
        
    def answer_question(self, context, question):
        """Get answer from context using DistilBERT"""
        # Clear GPU memory before processing
        MemoryOptimizer.clear_gpu_memory()
        
        try:
            # Handle empty context
            if not context or not context.strip():
                return ""
                
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation="only_second",
                stride=128,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            
            # Get the most likely answer span
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            
            # Convert back to tokens and then to string
            input_ids = inputs["input_ids"][0]
            tokens = input_ids[answer_start:answer_end]
            answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Clear memory after processing
            MemoryOptimizer.clear_gpu_memory()
            
            return answer
            
        except Exception as e:
            print(f"Error in DistilBERT QA model: {str(e)}")
            return ""
            
    def answer_question_with_chunks(self, context_chunks, question):
        """Process multiple context chunks and return best answer"""
        best_answer = ""
        best_score = -float('inf')
        
        # Process each chunk to find the best answer
        for i, chunk in enumerate(context_chunks):
            # Clear memory before processing each chunk
            MemoryOptimizer.clear_gpu_memory()
            
            try:
                # Tokenize input for this chunk
                inputs = self.tokenizer(
                    question,
                    chunk,
                    max_length=self.max_length,
                    truncation="only_second",
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process results
                start_scores = outputs.start_logits[0]
                end_scores = outputs.end_logits[0]
                
                # Find the best answer span
                answer_start = torch.argmax(start_scores)
                answer_end = torch.argmax(end_scores) + 1
                
                # Calculate confidence score
                score = float(start_scores[answer_start] + end_scores[answer_end-1])
                
                # Convert to tokens and string
                input_ids = inputs["input_ids"][0]
                tokens = input_ids[answer_start:answer_end]
                answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Only consider non-empty answers with a minimum length
                if answer and len(answer.strip().split()) >= 3:
                    if score > best_score:
                        best_score = score
                        best_answer = answer
                
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
                
        # Clear memory after processing all chunks
        MemoryOptimizer.clear_gpu_memory()
        
        return best_answer

class QAEngine:
    def __init__(self):
        # Replace MiniLM with DistilBERT
        self.qa_model = DistilBERTQAModel()
        self.context_chunks = []
        self.chunk_embeddings = []
        # Maximum context size for DistilBERT
        self.max_model_tokens = 384
        
    def prepare_context(self, text):
        """Split text into semantic chunks and compute embeddings"""
        self.context_chunks = chunk_text(text)
        # Process embeddings in batches to save memory
        self.chunk_embeddings = []
        batch_size = 8  # Process 8 chunks at a time
        
        for i in range(0, len(self.context_chunks), batch_size):
            batch = self.context_chunks[i:i+batch_size]
            batch_embeddings = [get_embedding(chunk) for chunk in batch]
            self.chunk_embeddings.extend(batch_embeddings)
            # Clear memory after each batch
            MemoryOptimizer.clear_gpu_memory()
    
    def get_top_context(self, question, top_k=5):
        """Retrieve the most relevant context chunks for the question"""
        if not self.context_chunks:
            return ["No context available. Please upload a document first."]
        
        q_embed = get_embedding(question)
        
        # Calculate similarities in batches to save memory
        scored = []
        batch_size = 20  # Process 20 similarities at a time
        
        for i in range(0, len(self.context_chunks), batch_size):
            batch_chunks = self.context_chunks[i:i+batch_size]
            batch_embeddings = self.chunk_embeddings[i:i+batch_size]
            
            batch_scores = [(chunk, cosine_similarity(q_embed, emb)) 
                           for chunk, emb in zip(batch_chunks, batch_embeddings)]
            scored.extend(batch_scores)
            
            # Clear memory after each batch
            MemoryOptimizer.clear_gpu_memory()
            
        # Sort by similarity score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k chunks with highest similarity
        top_chunks = [chunk for chunk, _ in scored[:top_k]]
        
        # Add one more distant chunk for broader context if available
        if len(scored) > top_k + 5:
            # Add a chunk from further down the list for additional context
            top_chunks.append(scored[top_k + 5][0])
            
        return top_chunks
    
    def get_truncated_context(self, question, context_chunks):
        """Smartly truncate context to fit within token limits"""
        # For DistilBERT, we'll process chunks separately rather than concatenating
        # This function now just joins the chunks for compatibility with old code
        return " ".join(context_chunks)
    
    def clean_and_format_text(self, text):
        """Clean up text by fixing common formatting issues"""
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix incomplete sentences at beginning
        if text and not text[0].isupper() and len(text) > 1:
            # If starts with lowercase, likely incomplete sentence
            first_capital = re.search(r'[A-Z]', text)
            if first_capital:
                start_idx = first_capital.start()
                # If the capital letter isn't too far in, trim the beginning
                if start_idx < 30:
                    text = text[start_idx:]
        
        # Ensure the text ends with proper punctuation
        if text and text[-1] not in '.!?':
            # Look for the last sentence end
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > len(text) * 0.7:  # If reasonably close to the end
                text = text[:last_period+1]
            else:
                text += "."
                
        return text.strip()
    
    def enhance_answer(self, raw_answer, question, context):
        """Create a comprehensive, coherent answer from the raw model output"""
        # Clean up the raw answer first
        raw_answer = self.clean_and_format_text(raw_answer)
        
        # If answer is empty or too short, generate a new one from context
        if not raw_answer or len(raw_answer.strip().split()) < 5:
            # Split context into clean sentences
            sentences = []
            for sent in re.split(r'(?<=[.!?])\s+', context):
                if sent.strip():
                    sentences.append(sent.strip())
            
            # Extract keywords from question
            question_words = question.lower().split()
            question_keywords = [w for w in question_words if len(w) > 3 
                                and w not in ('what', 'when', 'where', 'which', 'how', 'does', 'from', 'about')]
            
            # Find relevant sentences
            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance_score = sum(1 for keyword in question_keywords if keyword in sentence_lower)
                if relevance_score > 0:
                    relevant_sentences.append((sentence, relevance_score))
            
            # Sort by relevance score
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:4]]  # Take top 4 sentences
            
            if top_sentences:
                # Create introduction based on question type
                intro = self.get_question_based_intro(question)
                
                # Join sentences into coherent paragraph
                answer_body = " ".join(top_sentences)
                
                # Ensure we have complete sentences
                answer_body = self.clean_and_format_text(answer_body)
                
                enhanced_answer = intro + answer_body
                return enhanced_answer
            else:
                return "I couldn't find specific information addressing this question in the document."
        
        # If we have a decent raw answer, enhance it further
        intro = self.get_question_based_intro(question)
        
        # Check if raw answer already has a good introduction
        first_sentence = raw_answer.split('.')[0] + '.'
        if len(first_sentence.split()) > 7:
            # Answer likely already has a good introduction
            return self.clean_and_format_text(raw_answer)
        else:
            # Add our custom introduction
            return intro + raw_answer
    
    def get_question_based_intro(self, question):
        """Generate appropriate introduction based on question type"""
        question_lower = question.lower()
        
        if "what" in question_lower and ("main" in question_lower or "key" in question_lower):
            if "point" in question_lower or "finding" in question_lower or "conclusion" in question_lower:
                return "The main point from this paper is that "
            elif "contribution" in question_lower:
                return "The key contribution of this paper is "
            elif "purpose" in question_lower:
                return "The main purpose of this paper is to "
            else:
                return "The main focus of this paper is on "
        elif "how does" in question_lower or "how do" in question_lower:
            return "According to the document, "
        elif "why" in question_lower:
            return "The document explains that "
        elif "when" in question_lower:
            return "Based on the timeline presented in the document, "
        else:
            return "Based on the document, "
            
    def answer_question(self, question):
        try:
            # Clear memory at the start
            MemoryOptimizer.clear_gpu_memory()
            
            # Get relevant context chunks
            context_chunks = self.get_top_context(question)
            
            if not context_chunks:
                return "I couldn't find relevant information to answer this question in the document."
            
            # Process with new approach: send chunks directly to model
            raw_answer = self.qa_model.answer_question_with_chunks(context_chunks, question)
            
            # Get concatenated context for enhancement function
            full_context = " ".join(context_chunks)
            
            # Enhance and format the answer
            enhanced_answer = self.enhance_answer(raw_answer, question, full_context)
            
            # Final quality check
            if len(enhanced_answer.split()) < 8:
                return "I couldn't extract a clear answer from the document. The paper may not directly address this specific question."
                
            # Clear memory before returning
            MemoryOptimizer.clear_gpu_memory()
            
            return enhanced_answer
            
        except Exception as e:
            # Provide a graceful error message
            return (f"I apologize, but I encountered an issue while analyzing the document to answer your question. "
                   f"This could be due to the complexity of the content or how the document was processed. "
                   f"Please try a more specific question or check if the document was uploaded correctly. "
                   f"Technical details: {str(e)}")