from agents.base_agent import BaseAgent
from typing import List, Tuple, Dict
import json

class FactCheckerAgent(BaseAgent):
    def run(self, facts: List[str], documents: Dict[str, str] = None) -> Tuple[Dict, int]:
        """
        Validate facts against documents and check for consistency.
        
        Returns:
            (validation_dict, tokens_used)
        """
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        
        doc_context = ""
        if documents:
            doc_previews = []
            for name, text in list(documents.items())[:3]:  # Limit to 3 docs
                preview = text[:1000].replace('\n', ' ')
                doc_previews.append(f"{name}: {preview}...")
            doc_context = "\n\nDocument Context:\n" + "\n".join(doc_previews)
        
        system_prompt = """You are a fact-checker agent that validates claims and checks 
for consistency across financial documents.

For each fact, determine:
- Is it supported by the documents?
- Is it consistent across documents?
- Are there any contradictions?
- What is the confidence level?

Return JSON:
{
    "validated_facts": [
        {
            "fact": "...",
            "status": "Supported/Contradicted/Unclear",
            "confidence": "High/Medium/Low",
            "evidence": "...",
            "contradictions": ["..."] if any
        }
    ],
    "consistency_score": "High/Medium/Low",
    "summary": "Overall validation summary"
}"""

        user_prompt = f"""Validate these facts{doc_context}:

Facts to validate:
{facts_text}

Check each fact for support and consistency. Return valid JSON only."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        try:
            content = response["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            validation_data = json.loads(content)
        except json.JSONDecodeError:
            validation_data = {
                "validated_facts": [],
                "consistency_score": "Unknown",
                "summary": content[:500]
            }
        
        total_tokens = input_tokens + output_tokens
        return validation_data, total_tokens
