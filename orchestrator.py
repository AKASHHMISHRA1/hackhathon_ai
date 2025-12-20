from typing import Dict, Optional
import time
from agents import (
    triage_agent, financial_agent, risk_agent,
    strategy_agent, timeline_agent, fact_checker_agent, synthesis_agent, cache_agent
)
from utils.cost_tracker import CostTracker
from utils.text_chunks import TextChunker
from utils.vector_store import VectorStore

class Orchestrator:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.cost = CostTracker()
        self.cache = cache_agent.CacheAgent("Cache")
        self.vector_store = vector_store
        self.chunker = TextChunker(
            chunk_size=4000,  # characters
            chunk_overlap=200,
            use_recursive=True
        )
        
    def _chunk_document(self, text: str, agent_type: str = None) -> str:
        """
        Chunk document if it's too large, otherwise return as-is.
        For very large documents, returns first few chunks combined.
        
        Args:
            text: Document text
            agent_type: Type of agent processing the document
            
        Returns:
            Text ready for agent processing (may be chunked)
        """
        # Estimate tokens
        tokens = len(self.chunker.encoding.encode(text))
        
        # If document is small enough, return as-is
        if tokens <= 30000:  # ~30k tokens is safe for most models
            return text
        
        # Document is too large, chunk it
        chunks = self.chunker.chunk_document_for_agent(
            text, 
            max_chunks=10,  # Limit to first 10 chunks for processing
            agent_type=agent_type
        )
        
        # Combine chunks with separator
        return "\n\n---\n\n".join(chunks)
    
    def _format_sources(self, doc_names: list) -> list:
        """
        Format source document names.
        Since we don't have page numbers, we'll format as filename.
        If page numbers become available, they can be added.
        """
        return [f"{doc_name}" for doc_name in doc_names]
    
    def _retrieve_relevant_chunks(
        self,
        query: str,
        agent_type: str,
        document_names: Optional[list] = None,
        k: int = 5
    ) -> str:
        """
        Retrieve relevant document chunks using RAG.
        
        Args:
            query: Search query
            agent_type: Type of agent (financial, risk, strategy, timeline, fact_checker)
            document_names: Optional list of document names to filter by
            k: Number of chunks to retrieve
            
        Returns:
            Combined text from relevant chunks
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please provide vector_store to Orchestrator.")
        
        # Search for relevant chunks
        chunks = self.vector_store.search_by_agent_type(
            query=query,
            agent_type=agent_type,
            k=k,
            document_names=document_names
        )
        
        if not chunks:
            return ""
        
        # Combine chunks with metadata
        combined_text = []
        for chunk in chunks:
            source = chunk["metadata"].get("document_name", "unknown")
            chunk_idx = chunk["metadata"].get("chunk_index", 0)
            combined_text.append(
                f"[Source: {source}, Chunk {chunk_idx}]\n{chunk['text']}"
            )
        
        return "\n\n---\n\n".join(combined_text)
        
    def run(self, question: str, documents: Optional[Dict[str, str]] = None):
        """
        Main orchestration flow:
        1. Cache Check - check if similar question exists in cache
        2. Triage - determine relevant documents AND which agents to call
        3. Specialized agents analyze relevant chunks (only if needed)
        4. Synthesis - combine insights to answer question
        5. Fact Checker - verify the facts
        6. Cache - store the answer for future use
        
        Returns:
            Dict with format:
            {
                "answer": str,
                "sources": list[str],
                "cost": {
                    "total_usd": float,
                    "input_tokens": int,
                    "output_tokens": int,
                    "agents_used": list[dict]
                },
                "latency_seconds": float
            }
        """
        start_time = time.time()
        
        # Step 0: Check cache for similar questions
        cached_entry, cache_input, cache_output = self.cache.find_similar_question(question)
        cache_cost = self.cache.calculate_cost(cache_input, cache_output)
        self.cost.add("CacheAgent", cache_input, cache_output, cache_cost)
        
        if cached_entry:
            # Update access count
            self.cache.update_access_count(cached_entry['question'])
            latency = time.time() - start_time
            cost_report = self.cost.report()
            return {
                "answer": cached_entry['answer'],
                "sources": self._format_sources(cached_entry['metadata'].get('sources', [])),
                "cost": cost_report,
                "latency_seconds": round(latency, 2)
            }
        
        # Step 1: Triage - Determine relevant documents AND which agents to call
        triage = triage_agent.DocumentTriageAgent("Triage")
        relevant_docs, agents_needed, triage_input, triage_output = triage.run(
            question=question,
            documents=documents,
            vector_store=self.vector_store
        )
        triage_cost = triage.calculate_cost(triage_input, triage_output)
        self.cost.add("TriageAgent", triage_input, triage_output, triage_cost)
        
        if not relevant_docs:
            latency = time.time() - start_time
            cost_report = self.cost.report()
            return {
                "answer": "No relevant documents found for this question.",
                "sources": [],
                "cost": cost_report,
                "latency_seconds": round(latency, 2)
            }
        
        # Step 2: Call only the agents determined by triage
        analysis_data = {}
        
        # Financial Agent - only if needed
        if "financial" in agents_needed:
            financial = financial_agent.FinancialAgent("Financial")
            fin_text = self._retrieve_relevant_chunks(
                query=question,
                agent_type="financial",
                document_names=relevant_docs,
                k=5
            )
            if fin_text:
                fin_data, fin_input, fin_output = financial.run(fin_text)
                fin_cost = financial.calculate_cost(fin_input, fin_output)
                self.cost.add("FinancialAgent", fin_input, fin_output, fin_cost)
                analysis_data['financials'] = fin_data
            else:
                analysis_data['financials'] = {}
        
        # Risk Agent - only if needed
        if "risk" in agents_needed:
            risk = risk_agent.RiskAgent("Risk")
            risk_text = self._retrieve_relevant_chunks(
                query=question,
                agent_type="risk",
                document_names=relevant_docs,
                k=5
            )
            if risk_text:
                risks, risk_input, risk_output = risk.run(risk_text)
                risk_cost = risk.calculate_cost(risk_input, risk_output)
                self.cost.add("RiskAgent", risk_input, risk_output, risk_cost)
                analysis_data['risks'] = risks
            else:
                analysis_data['risks'] = {}
        
        # Strategy Agent - only if needed
        if "strategy" in agents_needed:
            strategy = strategy_agent.StrategyAgent("Strategy")
            strategy_text = self._retrieve_relevant_chunks(
                query=question,
                agent_type="strategy",
                document_names=relevant_docs,
                k=5
            )
            if strategy_text:
                strategy_data, strat_input, strat_output = strategy.run(strategy_text)
                strat_cost = strategy.calculate_cost(strat_input, strat_output)
                self.cost.add("StrategyAgent", strat_input, strat_output, strat_cost)
                analysis_data['strategy'] = strategy_data
            else:
                analysis_data['strategy'] = {}
        
        # Timeline Agent - only if needed
        if "timeline" in agents_needed:
            timeline = timeline_agent.TimelineAgent("Timeline")
            timeline_text = self._retrieve_relevant_chunks(
                query=question,
                agent_type="timeline",
                document_names=relevant_docs,
                k=5
            )
            if timeline_text:
                timeline_data, time_input, time_output = timeline.run(timeline_text)
                time_cost = timeline.calculate_cost(time_input, time_output)
                self.cost.add("TimelineAgent", time_input, time_output, time_cost)
                analysis_data['timeline'] = timeline_data
            else:
                analysis_data['timeline'] = {}
        
        # Fact Checker Agent - only if needed
        if "fact_checker" in agents_needed:
            fact_checker = fact_checker_agent.FactCheckerAgent("FactChecker")
            # For fact checker, we need to extract facts from the question or previous analysis
            # For now, we'll use the question itself and relevant chunks
            fact_checker_text = self._retrieve_relevant_chunks(
                query=question,
                agent_type="financial",  # Use financial as default for fact checking
                document_names=relevant_docs,
                k=5
            )
            if fact_checker_text:
                # Extract potential facts from the question
                facts_to_check = [question]  # Can be enhanced to extract specific facts
                validation_data, fact_checker_tokens = fact_checker.run(
                    facts=facts_to_check,
                    documents={doc: fact_checker_text for doc in relevant_docs} if relevant_docs else None
                )
                # Estimate input/output tokens (fact_checker returns total tokens)
                fact_checker_input = fact_checker_tokens // 2  # Rough estimate
                fact_checker_output = fact_checker_tokens // 2
                fact_checker_cost = fact_checker.calculate_cost(fact_checker_input, fact_checker_output)
                self.cost.add("FactCheckerAgent", fact_checker_input, fact_checker_output, fact_checker_cost)
                analysis_data['fact_checker'] = validation_data
            else:
                analysis_data['fact_checker'] = {}
        
        # Step 3: Synthesis
        synthesis = synthesis_agent.SynthesisAgent("Synthesis")
        answer, synth_input, synth_output = synthesis.run(question, analysis_data)
        synth_cost = synthesis.calculate_cost(synth_input, synth_output)
        self.cost.add("SynthesisAgent", synth_input, synth_output, synth_cost)
        
        latency = time.time() - start_time
        cost_report = self.cost.report()
        
        result = {
            "answer": answer,
            "sources": self._format_sources(relevant_docs),
            "cost": cost_report,
            "latency_seconds": round(latency, 2)
        }
        
        # Step 4: Store in cache for future use
        self.cache.store_answer(
            question=question,
            answer=answer,
            metadata={
                "sources": relevant_docs,
                "agents_used": agents_needed,
                "analysis": analysis_data,
                "cost": cost_report
            }
        )
        
        return result
