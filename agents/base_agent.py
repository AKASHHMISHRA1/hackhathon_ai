import os
import tiktoken
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    def __init__(self, name: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.model = model
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        # Model pricing (per 1M tokens)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00}
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        prices = self.pricing.get(self.model, self.pricing["gpt-4o-mini"])
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost

    def call_llm(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> tuple[Dict[str, Any], int, int, float]:
        """
        Make API call to OpenAI.
        
        Returns:
            (response_dict, input_tokens, output_tokens, cost)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }, input_tokens, output_tokens, cost
            
        except Exception as e:
            raise Exception(f"OpenAI API error in {self.name}: {str(e)}")
