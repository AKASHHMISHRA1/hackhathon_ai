from agents.base_agent import BaseAgent
from typing import Dict, Tuple, Optional, List
import json
import os
from datetime import datetime

class CacheAgent(BaseAgent):
    """
    Cache agent that stores and retrieves similar questions and answers.
    Uses semantic similarity to find cached responses for similar questions.
    """
    
    def __init__(self, name: str = "Cache", model: str = "gpt-4o-mini", cache_file: str = "cache.json"):
        super().__init__(name, model)
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.similarity_threshold = 0.85  # Minimum similarity to use cached answer
    
    def _load_cache(self) -> List[Dict]:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def find_similar_question(self, question: str) -> Tuple[Optional[Dict], int, int]:
        """
        Find similar question in cache using semantic similarity.
        
        Returns:
            (cached_entry or None, input_tokens, output_tokens)
        """
        if not self.cache:
            return None, 0, 0
        
        # Use LLM to find most similar question
        cache_questions = [entry['question'] for entry in self.cache]
        cache_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(cache_questions)])
        
        system_prompt = """You are a cache retrieval agent. Your role is to determine if a new question 
is semantically similar to any cached questions. Return the index number (1-based) of the most similar 
question, or 0 if no question is similar enough (similarity < 85%).

Consider questions similar if they:
- Ask about the same financial metric or topic
- Request the same type of analysis
- Have the same intent despite different wording

Return ONLY a number: the index (1-N) if similar, or 0 if not similar enough."""

        user_prompt = f"""New Question: {question}

Cached Questions:
{cache_list}

Is the new question similar to any cached question? Return the index number or 0."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        try:
            index = int(response["content"].strip())
            if 1 <= index <= len(self.cache):
                cached_entry = self.cache[index - 1]
                # Verify similarity score
                similarity_score = self._calculate_similarity_score(question, cached_entry['question'])
                if similarity_score >= self.similarity_threshold:
                    return cached_entry, input_tokens, output_tokens
        except (ValueError, IndexError):
            pass
        
        return None, input_tokens, output_tokens
    
    def _calculate_similarity_score(self, question1: str, question2: str) -> float:
        """
        Calculate similarity score between two questions using LLM.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        system_prompt = """You are a similarity scorer. Rate how similar two questions are on a scale 
of 0.0 to 1.0, where 1.0 means identical intent and 0.0 means completely different.

Return ONLY a decimal number between 0.0 and 1.0."""

        user_prompt = f"""Question 1: {question1}

Question 2: {question2}

How similar are these questions? Return a number between 0.0 and 1.0."""

        response, _, _, _ = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        try:
            score = float(response["content"].strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except ValueError:
            return 0.0
    
    def store_answer(self, question: str, answer: str, metadata: Dict = None):
        """
        Store a question-answer pair in cache.
        
        Args:
            question: The question asked
            answer: The answer provided
            metadata: Additional metadata (sources, cost, etc.)
        """
        entry = {
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        self.cache.append(entry)
        self._save_cache()
    
    def update_access_count(self, question: str):
        """Update access count for a cached question."""
        for entry in self.cache:
            if entry['question'] == question:
                entry['access_count'] = entry.get('access_count', 0) + 1
                self._save_cache()
                break
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        return {
            "total_entries": len(self.cache),
            "total_accesses": sum(entry.get('access_count', 0) for entry in self.cache),
            "most_accessed": max(self.cache, key=lambda x: x.get('access_count', 0))['question'] if self.cache else None
        }
