from agents.base_agent import BaseAgent
from typing import Dict, Tuple
import json

class SynthesisAgent(BaseAgent):
    def run(self, question: str, data: Dict) -> Tuple[str, int, int]:
        """
        Synthesize information from multiple agents to answer the question.
        
        Returns:
            (answer, input_tokens, output_tokens)
        """
        # Format the data from different agents
        data_summary = f"""
Financial Analysis:
{json.dumps(data.get('financials', {}), indent=2)}

Risk Analysis:
{json.dumps(data.get('risks', {}), indent=2)}

Strategy Analysis:
{json.dumps(data.get('strategy', {}), indent=2)}

Timeline Analysis:
{json.dumps(data.get('timeline', {}), indent=2)}
"""
        
        system_prompt = """You are a synthesis agent that combines insights from multiple 
specialized agents to provide clear, concise answers to financial questions.

Your role:
- Synthesize information from financial, risk, strategy, and timeline analyses
- Provide a clear, direct answer to the question
- Include specific data points and metrics when relevant
- Keep the answer concise but comprehensive
- Do NOT use markdown formatting - return plain text only"""

        user_prompt = f"""Question: {question}

Available Analysis Data:
{data_summary}

Provide a clear, direct answer to the question. Include specific numbers and metrics when relevant. 
Return only the answer text, no markdown formatting."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=2000
        )
        
        answer = response["content"].strip()
        # Remove any markdown formatting if present
        answer = answer.replace("###", "").replace("##", "").replace("#", "")
        answer = answer.replace("**", "").replace("*", "")
        
        return answer, input_tokens, output_tokens
