"""
Text Chunking Utility using LangChain - Splits large documents into manageable chunks.

This module provides:
- Intelligent text chunking for large documents using LangChain
- Overlap management for context preservation
- Token-aware chunking
- Section-aware chunking for SEC filings
"""

from typing import List, Dict, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
import tiktoken


class TextChunker:
    """
    Utility class for chunking large text documents using LangChain.
    
    Handles intelligent splitting while preserving context and
    managing token limits for LLM processing.
    """
    
    def __init__(
        self, 
        chunk_size: int = 4000, 
        chunk_overlap: int = 200,
        model: str = "gpt-4o-mini",
        use_recursive: bool = True
    ):
        """
        Initialize text chunker with LangChain.
        
        Args:
            chunk_size: Target chunk size in characters (for RecursiveCharacterTextSplitter)
                       or tokens (for TokenTextSplitter)
            chunk_overlap: Overlap between chunks
            model: Model name for token counting
            use_recursive: Use RecursiveCharacterTextSplitter (True) or TokenTextSplitter (False)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize LangChain text splitter
        if use_recursive:
            # RecursiveCharacterTextSplitter is better for preserving context
            # It tries to split on: \n\n, \n, " ", then character
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,  # Count by characters
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            # TokenTextSplitter for token-aware splitting
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name="cl100k_base"  # GPT-4 tokenizer
            )
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Split text into chunks using LangChain.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Create LangChain Document
        doc = Document(page_content=text, metadata=metadata or {})
        
        # Split using LangChain
        chunks = self.splitter.split_documents([doc])
        
        # Convert to our format
        result = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            tokens = len(self.encoding.encode(chunk_text))
            
            result.append({
                "chunk_index": i,
                "text": chunk_text,
                "tokens": tokens,
                "characters": len(chunk_text),
                "metadata": chunk.metadata
            })
        
        return result
    
    def chunk_by_sections(self, text: str, section_markers: Optional[List[str]] = None) -> List[Dict[str, any]]:
        """
        Chunk text by sections (e.g., Item 1, Item 2, etc. in SEC filings).
        Uses LangChain for final chunking of each section.
        
        Args:
            text: Text to chunk
            section_markers: List of regex patterns for section markers
            
        Returns:
            List of chunk dictionaries
        """
        import re
        
        if section_markers is None:
            section_markers = [
                r'Item\s+\d+[A-Z]?:',
                r'PART\s+[IVX]+',
                r'Section\s+\d+',
                r'Chapter\s+\d+',
                r'ITEM\s+\d+[A-Z]?:'
            ]
        
        # Find all section boundaries
        sections = []
        pattern = '|'.join(f'({marker})' for marker in section_markers)
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        
        if not matches:
            # No sections found, use regular chunking
            return self.chunk_text(text)
        
        # Extract sections
        for i, match in enumerate(matches):
            if i < len(matches) - 1:
                # Section from current match to next match
                section_text = text[match.start():matches[i+1].start()]
            else:
                # Last section
                section_text = text[match.start():]
            
            sections.append({
                "section_header": match.group(0),
                "text": section_text,
                "start": match.start()
            })
        
        # Chunk each section if needed
        final_chunks = []
        for section in sections:
            section_text = section["text"]
            section_tokens = len(self.encoding.encode(section_text))
            
            # If section is too large, chunk it
            if section_tokens > self.chunk_size * 2:  # Allow 2x chunk size for sections
                sub_chunks = self.chunk_text(
                    section_text,
                    metadata={"section_header": section["section_header"]}
                )
                for sub_chunk in sub_chunks:
                    sub_chunk["section_header"] = section["section_header"]
                    final_chunks.append(sub_chunk)
            else:
                # Section fits in one chunk
                final_chunks.append({
                    "chunk_index": len(final_chunks),
                    "text": section_text,
                    "tokens": section_tokens,
                    "characters": len(section_text),
                    "section_header": section["section_header"],
                    "metadata": {"section_header": section["section_header"]}
                })
        
        return final_chunks
    
    def chunk_document_for_agent(
        self, 
        text: str, 
        max_chunks: Optional[int] = None,
        agent_type: Optional[str] = None
    ) -> List[str]:
        """
        Chunk document specifically for agent processing.
        Returns list of chunk texts ready for agent consumption.
        
        Args:
            text: Text to chunk
            max_chunks: Maximum number of chunks to return (None for all)
            agent_type: Type of agent (can customize chunking strategy)
            
        Returns:
            List of chunk text strings
        """
        # For financial documents, prefer section-based chunking
        if agent_type in ["financial", "risk", "strategy"]:
            chunks = self.chunk_by_sections(text)
        else:
            chunks = self.chunk_text(text)
        
        # Limit number of chunks if specified
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        # Return just the text content
        return [chunk["text"] for chunk in chunks]
    
    def get_chunk_summary(self, chunks: List[Dict]) -> Dict:
        """
        Get summary statistics about chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Summary dictionary
        """
        total_tokens = sum(chunk.get("tokens", 0) for chunk in chunks)
        total_chars = sum(chunk.get("characters", 0) for chunk in chunks)
        
        return {
            "num_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
            "avg_chars_per_chunk": total_chars / len(chunks) if chunks else 0
        }

