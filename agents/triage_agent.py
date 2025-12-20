from agents.base_agent import BaseAgent
from typing import List, Tuple, Dict

class DocumentTriageAgent(BaseAgent):
    def run(self, question: str, documents: Dict[str, str]) -> Tuple[List[str], int, int]:
        """
        Analyze question and documents to determine which documents are relevant.
        
        Returns:
            (relevant_document_names, input_tokens, output_tokens)
        """
        # Create summary of available documents
        doc_summaries = []
        for name, text in documents.items():
            # Get first 500 chars as summary
            preview = text[:500].replace('\n', ' ')
            doc_summaries.append(f"- {name}: {preview}...")
        
        doc_list = "\n".join(doc_summaries)
        
        system_prompt = """You are a document triage agent for financial intelligence analysis.
Your role is to analyze user questions and determine which financial documents (10-K, 10-Q filings) 
are most relevant to answer the question.

Analyze the question and available documents, then return ONLY a comma-separated list of 
document filenames that are relevant to answering the question. Be precise and only include 
documents that are actually relevant."""

        user_prompt = f"""Question: {question}

Available Documents:
{doc_list}

Which documents are relevant? Return only a comma-separated list of filenames."""

        response, input_tokens, output_tokens, cost = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=200
        )
        
        # Parse response to get document names
        relevant_docs = []
        content = response["content"].strip()
        
        # Extract document names from response
        for doc_name in documents.keys():
            if doc_name in content or any(part in doc_name for part in content.split(',')):
                relevant_docs.append(doc_name)
        
        # If no matches, return all (fallback)
        if not relevant_docs:
            relevant_docs = list(documents.keys())[:3]  # Limit to 3
        
        return relevant_docs, input_tokens, output_tokens
