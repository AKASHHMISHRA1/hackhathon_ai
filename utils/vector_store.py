"""
Vector Store Utility for RAG (Retrieval Augmented Generation)

This module provides:
- ChromaDB vector database integration
- Document embedding and storage
- Semantic search for relevant document chunks
- Metadata management for document tracking
"""

import os
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from utils.text_chunks import TextChunker


class VectorStore:
    """
    Vector store manager using ChromaDB and OpenAI embeddings.
    Handles document storage, embedding, and retrieval for RAG.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "financial_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize OpenAI embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"  # Cost-effective embedding model
        )
        
        # Initialize text chunker
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_recursive=True
        )
        
        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        self._initialized = False
    
    def add_documents(
        self,
        documents: Dict[str, str],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add documents to the vector store.
        Documents are chunked and embedded before storage.
        
        Args:
            documents: Dictionary mapping document names to text content
            metadata: Optional metadata to attach to all documents
            
        Returns:
            Number of chunks added
        """
        all_chunks = []
        all_metadatas = []
        
        for doc_name, text in documents.items():
            # Chunk the document
            chunks = self.chunker.chunk_text(
                text,
                metadata={"source": doc_name, "document_name": doc_name}
            )
            
            # Create LangChain Documents with metadata
            for i, chunk_data in enumerate(chunks):
                chunk_metadata = {
                    "source": doc_name,
                    "document_name": doc_name,
                    "chunk_index": chunk_data["chunk_index"],
                    "chunk_id": f"{doc_name}_{i}",
                    **(metadata or {}),
                    **(chunk_data.get("metadata", {}))
                }
                
                doc = Document(
                    page_content=chunk_data["text"],
                    metadata=chunk_metadata
                )
                all_chunks.append(doc)
                all_metadatas.append(chunk_metadata)
        
        # Add to vector store
        if all_chunks:
            self.vectorstore.add_documents(all_chunks)
            self.vectorstore.persist()
        
        self._initialized = True
        return len(all_chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant document chunks using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"source": "doc.pdf"})
            
        Returns:
            List of dictionaries with:
            - text: Chunk text
            - metadata: Chunk metadata
            - score: Similarity score
        """
        if not self._initialized:
            return []
        
        # Perform similarity search
        if filter_metadata:
            # Use where clause for filtering
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return formatted_results
    
    def search_by_agent_type(
        self,
        query: str,
        agent_type: str,
        k: int = 5,
        document_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks optimized for specific agent types.
        Can filter by document names if provided.
        
        Args:
            query: Search query
            agent_type: Type of agent (financial, risk, strategy, timeline)
            k: Number of results to return
            document_names: Optional list of document names to filter by
            
        Returns:
            List of relevant chunks
        """
        # Build query with agent-specific context
        agent_context = {
            "financial": "financial metrics, revenue, income, assets, liabilities, earnings",
            "risk": "risks, risk factors, uncertainties, challenges, threats",
            "strategy": "strategy, business strategy, plans, objectives, goals",
            "timeline": "dates, timeline, periods, quarters, years, chronological"
        }
        
        enhanced_query = query
        if agent_type in agent_context:
            enhanced_query = f"{query} {agent_context[agent_type]}"
        
        # Filter by document names if provided
        filter_metadata = None
        if document_names:
            # ChromaDB uses $in for list filtering
            filter_metadata = {"document_name": {"$in": document_names}}
        
        return self.search(enhanced_query, k=k, filter_metadata=filter_metadata)
    
    def get_document_chunks(
        self,
        document_name: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_name: Name of the document
            limit: Optional limit on number of chunks
            
        Returns:
            List of chunks for the document
        """
        # Search with document filter
        results = self.search(
            query="",  # Empty query to get all chunks
            k=limit or 1000,
            filter_metadata={"document_name": document_name}
        )
        
        # Sort by chunk_index if available
        results.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        
        return results
    
    def clear_collection(self):
        """Clear all documents from the vector store."""
        # Delete the collection and recreate
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=self.collection_name)
        except:
            pass
        
        # Reinitialize
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self._initialized = False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            return {"total_chunks": 0, "documents": []}
        
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get unique documents
            results = collection.get(limit=10000)
            documents = set()
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata and "document_name" in metadata:
                        documents.add(metadata["document_name"])
            
            return {
                "total_chunks": count,
                "documents": list(documents),
                "num_documents": len(documents)
            }
        except Exception as e:
            return {"total_chunks": 0, "documents": [], "error": str(e)}

