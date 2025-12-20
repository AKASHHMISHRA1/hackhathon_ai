from agents.base_agent import BaseAgent
from typing import List, Tuple, Dict
import json

class RiskAgent(BaseAgent):
    def run(self, text: str) -> Tuple[Dict, int, int]:
        """
        Identify and categorize risks from financial documents.
        
        Returns:
            (risk_analysis_dict, input_tokens, output_tokens)
        """
        max_chars = 32000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... document truncated ...]"
        
        system_prompt = """You are a risk analysis agent specializing in identifying and 
categorizing risks from financial documents.

Identify and categorize:
- Financial risks (liquidity, credit, market)
- Operational risks
- Regulatory/Compliance risks
- Strategic risks
- Cybersecurity risks
- Market/Competitive risks

Return your analysis as JSON:
{
    "risk_factors": [
        {
            "category": "Financial/Operational/Regulatory/etc",
            "description": "Detailed description",
            "severity": "High/Medium/Low",
            "mitigation_mentioned": "Yes/No"
        }
    ],
    "summary": "Overall risk assessment summary",
    "key_concerns": ["...", "..."]
}"""

        user_prompt = f"""Analyze risks in the following financial document:

{text}

Identify and categorize all risk factors. Return valid JSON only."""

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
            
            risk_data = json.loads(content)
        except json.JSONDecodeError:
            risk_data = {
                "risk_factors": [],
                "summary": content[:500],
                "key_concerns": []
            }
        
        return risk_data, input_tokens, output_tokens
