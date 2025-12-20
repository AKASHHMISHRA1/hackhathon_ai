from agents.base_agent import BaseAgent
from typing import List, Tuple, Dict
import json

class StrategyAgent(BaseAgent):
    def run(self, text: str) -> Tuple[Dict, int, int]:
        """
        Analyze business strategy and strategic initiatives.
        
        Returns:
            (strategy_analysis_dict, input_tokens, output_tokens)
        """
        max_chars = 32000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... document truncated ...]"
        
        system_prompt = """You are a strategic intelligence agent analyzing business strategy 
from financial documents.

Extract and analyze:
- Strategic objectives and goals
- Key business initiatives
- Competitive positioning
- Market strategy
- Growth strategies
- Innovation initiatives
- Strategic partnerships or acquisitions

Return JSON:
{
    "strategic_objectives": ["...", "..."],
    "key_initiatives": [
        {
            "initiative": "...",
            "description": "...",
            "status": "Ongoing/Planned/Completed"
        }
    ],
    "competitive_positioning": "...",
    "growth_strategy": "...",
    "summary": "Overall strategic direction summary"
}"""

        user_prompt = f"""Analyze the business strategy in this document:

{text}

Extract strategic information. Return valid JSON only."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            content = response["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            strategy_data = json.loads(content)
        except json.JSONDecodeError:
            strategy_data = {
                "strategic_objectives": [],
                "key_initiatives": [],
                "competitive_positioning": "",
                "growth_strategy": "",
                "summary": content[:500]
            }
        
        return strategy_data, input_tokens, output_tokens
