from agents.base_agent import BaseAgent
from typing import Dict, Tuple, List
import json
from utils.text_chunks import TextChunker

class FinancialAgent(BaseAgent):
    def __init__(self, name: str = "Financial", model: str = "gpt-4o-mini"):
        super().__init__(name, model)
        self.chunker = TextChunker(chunk_size=4000, chunk_overlap=200)
    
    def run(self, text: str) -> Tuple[Dict, int, int]:
        """
        Extract financial metrics and analyze financial statements.
        Handles large documents by chunking.
        
        Returns:
            (financial_data_dict, input_tokens, output_tokens)
        """
        # Check if text needs chunking
        tokens = len(self.encoder.encode(text))
        total_input_tokens = 0
        total_output_tokens = 0
        all_results = []
        
        if tokens > 30000:
            # Process in chunks
            chunks = self.chunker.chunk_document_for_agent(text, agent_type="financial")
            
            for chunk in chunks:
                result, input_toks, output_toks, cost = self._process_chunk(chunk)
                all_results.append(result)
                total_input_tokens += input_toks
                total_output_tokens += output_toks
            
            # Combine results
            combined_result = self._combine_results(all_results)
        else:
            # Process entire document
            combined_result, input_toks, output_toks, cost = self._process_chunk(text)
            total_input_tokens = input_toks
            total_output_tokens = output_toks
        
        return combined_result, total_input_tokens, total_output_tokens
    
    def _process_chunk(self, text: str):
        """Process a single chunk of text."""
        max_chars = 32000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... document truncated ...]"
        
        system_prompt = """You are a financial analysis agent specializing in extracting and 
analyzing financial metrics from SEC filings (10-K, 10-Q reports).

Extract key financial information including:
- Revenue/Net Sales
- Net Income
- Earnings Per Share (EPS)
- Total Assets
- Total Liabilities
- Shareholders' Equity
- Key financial ratios mentioned
- Financial trends and year-over-year changes

Return your analysis as a JSON object with the following structure:
{
    "metrics": {
        "revenue": "...",
        "net_income": "...",
        "eps": "...",
        "assets": "...",
        "liabilities": "...",
        "equity": "..."
    },
    "trends": {
        "revenue_trend": "increasing/decreasing/stable",
        "profitability_trend": "...",
        "key_changes": ["...", "..."]
    },
    "ratios": {
        "debt_to_equity": "...",
        "current_ratio": "...",
        "other_ratios": ["..."]
    },
    "summary": "Brief summary of financial position"
}"""

        user_prompt = f"""Analyze the following financial document excerpt:

{text}

Extract and analyze the financial metrics. Return valid JSON only."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1500
        )
        
        # Parse JSON response
        try:
            content = response["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            financial_data = json.loads(content)
        except json.JSONDecodeError:
            financial_data = {
                "metrics": {},
                "trends": {},
                "ratios": {},
                "summary": content[:500]
            }
        
        return financial_data, input_tokens, output_tokens, cost
    
    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple chunks."""
        combined = {
            "metrics": {},
            "trends": {},
            "ratios": {},
            "summary": []
        }
        
        for result in results:
            # Merge metrics
            if "metrics" in result:
                combined["metrics"].update(result["metrics"])
            
            # Merge trends
            if "trends" in result:
                combined["trends"].update(result["trends"])
            
            # Merge ratios
            if "ratios" in result:
                combined["ratios"].update(result["ratios"])
            
            # Collect summaries
            if "summary" in result:
                if isinstance(result["summary"], str):
                    combined["summary"].append(result["summary"])
        
        # Combine summaries
        combined["summary"] = " ".join(combined["summary"]) if combined["summary"] else "Financial analysis completed."
        
        return combined
