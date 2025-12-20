from agents.base_agent import BaseAgent
from typing import List, Tuple, Dict
import json

class DocumentTriageAgent(BaseAgent):
    def run(self, question: str, documents: Dict[str, str] = None, vector_store=None) -> Tuple[List[str], List[str], int, int]:
        """
        Analyze question to determine:
        1. Which documents are relevant
        2. Which agents should be called
        
        Returns:
            (relevant_document_names, agent_names_to_call, input_tokens, output_tokens)
        """
        # Use vector store if available, otherwise use documents dict
        if vector_store:
            # Get relevant documents from vector store
            triage_chunks = vector_store.search(question, k=10)
            relevant_docs = list(set([
                chunk["metadata"].get("document_name")
                for chunk in triage_chunks
                if chunk["metadata"].get("document_name")
            ]))
            
            # If no documents found, get all available
            if not relevant_docs:
                stats = vector_store.get_stats()
                relevant_docs = stats.get("documents", [])[:3]
        else:
            # Fallback: use document summaries
            doc_summaries = []
            for name, text in (documents or {}).items():
                preview = text[:500].replace('\n', ' ') if text else ""
                doc_summaries.append(f"- {name}: {preview}...")
            
            doc_list = "\n".join(doc_summaries) if doc_summaries else "No documents available"
            relevant_docs = list(documents.keys())[:3] if documents else []
        
        # Determine which agents are needed based on the question
        system_prompt = """You are a document triage and routing agent for financial intelligence analysis.
Your role is to:
1. Analyze the user's question to determine which specialized agents are needed
2. Determine which documents are relevant

Available agents:
- financial: For questions about revenue, income, earnings, assets, liabilities, financial metrics, EPS, balance sheet, income statement
- risk: For questions about risks, risk factors, uncertainties, challenges, threats, vulnerabilities
- strategy: For questions about business strategy, strategic plans, objectives, competitive positioning, growth strategy
- timeline: For questions about dates, timelines, periods, quarters, years, chronological events, historical data
- fact_checker: For questions requiring fact validation, consistency checks, or verification of claims

Analyze the question and return a JSON response with:
1. "relevant_documents": List of document filenames (or empty if using vector store)
2. "agents_needed": List of agent names that should be called (e.g., ["financial"], ["financial", "risk"], etc.)
   - Only include agents that are actually relevant to the question
   - Be selective - don't include agents unless they're clearly needed

Example responses:
- Question about revenue → {"agents_needed": ["financial"]}
- Question about risks → {"agents_needed": ["risk"]}
- Question about strategy → {"agents_needed": ["strategy"]}
- Question about Q1 2023 revenue → {"agents_needed": ["financial", "timeline"]}
- Question about financial risks → {"agents_needed": ["financial", "risk"]}
- Question asking to verify/validate facts → {"agents_needed": ["fact_checker"]}"""

        user_prompt = f"""Question: {question}

Analyze this question and determine:
1. Which agents are needed to answer it
2. Which documents are relevant (if documents dict provided)

Return ONLY valid JSON in this format:
{{
    "agents_needed": ["agent1", "agent2"],
    "relevant_documents": ["doc1.pdf", "doc2.pdf"]
}}

Be selective - only include agents that are clearly relevant to the question."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=300
        )
        
        # Parse JSON response
        try:
            content = response["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            triage_result = json.loads(content)
            agents_needed = triage_result.get("agents_needed", [])
            
            # Validate agent names
            valid_agents = ["financial", "risk", "strategy", "timeline", "fact_checker"]
            agents_needed = [a for a in agents_needed if a in valid_agents]
            
            # If no agents determined, default to financial (most common)
            if not agents_needed:
                agents_needed = ["financial"]
            
            # Use documents from response if provided, otherwise use vector store results
            if vector_store:
                # relevant_docs already determined from vector store
                pass
            else:
                doc_names_from_response = triage_result.get("relevant_documents", [])
                if doc_names_from_response:
                    # Filter to only documents that exist
                    relevant_docs = [d for d in doc_names_from_response if d in (documents or {}).keys()]
                # If still empty, use fallback
                if not relevant_docs and documents:
                    relevant_docs = list(documents.keys())[:3]
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: determine agents based on keywords
            question_lower = question.lower()
            agents_needed = []
            
            if any(word in question_lower for word in ["revenue", "income", "earnings", "profit", "asset", "liability", "financial", "eps", "balance sheet"]):
                agents_needed.append("financial")
            if any(word in question_lower for word in ["risk", "uncertainty", "challenge", "threat", "vulnerability"]):
                agents_needed.append("risk")
            if any(word in question_lower for word in ["strategy", "strategic", "plan", "objective", "competitive", "growth"]):
                agents_needed.append("strategy")
            if any(word in question_lower for word in ["date", "time", "period", "quarter", "year", "when", "timeline", "chronological"]):
                agents_needed.append("timeline")
            if any(word in question_lower for word in ["verify", "validate", "check", "fact", "consistent", "contradict"]):
                agents_needed.append("fact_checker")
            
            # Default to financial if nothing matched
            if not agents_needed:
                agents_needed = ["financial"]
        
        return relevant_docs, agents_needed, input_tokens, output_tokens
