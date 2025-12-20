from agents.base_agent import BaseAgent
from typing import Dict, Tuple, List
import json

class TimelineAgent(BaseAgent):
    def run(self, text: str) -> Tuple[Dict, int, int]:
        """
        Extract temporal information and create timeline of events.
        
        Returns:
            (timeline_dict, input_tokens, output_tokens)
        """
        max_chars = 32000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... document truncated ...]"
        
        system_prompt = """You are a timeline agent that extracts temporal information and 
creates chronological narratives from financial documents.

Extract:
- Important dates and events
- Financial reporting periods
- Key milestones
- Regulatory filings dates
- Significant business events
- Historical context

Return JSON:
{
    "timeline_events": [
        {
            "date": "YYYY-MM-DD or period description",
            "event": "Description of event",
            "category": "Financial/Regulatory/Business/etc",
            "significance": "High/Medium/Low"
        }
    ],
    "reporting_periods": ["...", "..."],
    "chronological_narrative": "Narrative summary of key events in chronological order"
}"""

        user_prompt = f"""Extract temporal information from this document:

{text}

Create a timeline of events. Return valid JSON only."""

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
            
            timeline_data = json.loads(content)
        except json.JSONDecodeError:
            timeline_data = {
                "timeline_events": [],
                "reporting_periods": [],
                "chronological_narrative": content[:500]
            }
        
        return timeline_data, input_tokens, output_tokens
