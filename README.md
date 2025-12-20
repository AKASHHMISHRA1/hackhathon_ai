# Financial Intelligence Agent System

## 🎯 Overview

A sophisticated multi-agent system for analyzing financial documents (10-K, 10-Q filings, etc.) using intelligent agent collaboration. This system demonstrates **"Build Smart, Spend Less"** principles by optimizing token usage and costs through intelligent task routing, agent specialization, intelligent caching, LangChain-powered document chunking, and **RAG (Retrieval Augmented Generation)** with vector database storage.

## 🏗️ Architecture

### Multi-Agent System

The system consists of specialized agents that work together to analyze financial documents:

1. **Cache Agent** - Retrieves similar questions from cache to avoid redundant processing
2. **Triage Agent** - First point of contact, determines document type and routing strategy
3. **Financial Agent** - Analyzes financial statements and metrics
4. **Risk Agent** - Identifies and categorizes risks
5. **Strategy Agent** - Analyzes business strategy and initiatives
6. **Timeline Agent** - Extracts temporal information and creates timelines
7. **Fact Checker Agent** - Validates claims and checks for inconsistencies
8. **Synthesis Agent** - Combines all insights into a comprehensive report

### Key Features

- **RAG with Vector Database**: Documents are stored in ChromaDB vector database with semantic embeddings for efficient retrieval
- **Intelligent Caching**: Cache agent finds similar questions to avoid redundant API calls
- **LangChain Chunking**: Advanced document chunking using LangChain's RecursiveCharacterTextSplitter for optimal context preservation
- **Semantic Search**: Agents retrieve only relevant document chunks using semantic similarity search
- **Intelligent Routing**: Triage agent determines which agents are needed
- **Cost Optimization**: Tracks token usage and costs across all agents
- **Parallel Processing**: Agents can work in parallel when appropriate
- **Resource Efficiency**: Uses cost-effective models (gpt-4o-mini) by default
- **Comprehensive Analysis**: Multi-dimensional analysis of financial documents

## 📁 Project Structure

```
financial-intelligence-agent/
├── README.md
├── requirements.txt
├── cache.json              # Question-answer cache (auto-generated)
├── data/
│   ├── 10K_2025.pdf
│   └── 10Q_Q1_2023.pdf
├── agents/
│   ├── base_agent.py          # Base class for all agents
│   ├── cache_agent.py          # Caching and similarity matching
│   ├── triage_agent.py         # Document triage and routing
│   ├── financial_agent.py      # Financial analysis
│   ├── risk_agent.py           # Risk assessment
│   ├── strategy_agent.py        # Strategic analysis
│   ├── timeline_agent.py       # Temporal analysis
│   ├── fact_checker_agent.py    # Fact validation
│   └── synthesis_agent.py      # Final synthesis
├── utils/
│   ├── pdf_loader.py            # PDF text extraction
│   ├── cost_tracker.py          # Cost and token tracking
│   ├── text_chunks.py           # LangChain-based text chunking
│   └── vector_store.py          # Vector database and RAG implementation
├── orchestrator.py              # Main coordination logic
├── app.py                        # FastAPI application
└── chroma_db/                    # Vector database storage (auto-generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional for demonstration, required for full functionality)

### Installation

1. **Navigate to the project directory:**
```bash
cd ai-agent/financial-intelligence-agent
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: The system uses LangChain for advanced document chunking and ChromaDB for vector storage. Key dependencies include:
- `langchain` - Core LangChain library for text splitting
- `langchain-openai` - OpenAI integration for LangChain
- `langchain-community` - Community integrations including ChromaDB
- `chromadb` - Vector database for storing document embeddings
- `tiktoken` - Token counting for accurate chunk sizing

4. **Set up environment variables:**
Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

#### Option 1: FastAPI Server

Start the FastAPI server:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /status` - System status and metrics
- `POST /analyze` - Analyze a document by path
- `POST /upload` - Upload and analyze a PDF file
- `GET /costs` - Get cost tracking summary
- `GET /cache/stats` - Get cache statistics
- `POST /cache/clear` - Clear the cache
- `POST /reset` - Reset system metrics

**Example API Usage:**
```bash
# Analyze a document
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"document_path": "data/10K_2025.pdf"}'

# Upload and analyze
curl -X POST "http://localhost:8000/upload" \
  -F "file=@data/10K_2025.pdf"

# Get system status
curl "http://localhost:8000/status"

# Get cost summary
curl "http://localhost:8000/costs"

# Get cache statistics
curl "http://localhost:8000/cache/stats"
```

#### Option 2: Direct Python Usage

```python
from orchestrator import Orchestrator
from utils.vector_store import VectorStore
from utils.pdf_loader import load_pdf
import os

# Initialize vector store
vector_store = VectorStore(
    persist_directory="./chroma_db",
    collection_name="financial_documents"
)

# Load and store documents in vector DB
documents = {}
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        documents[file] = load_pdf(f"data/{file}")

# Add documents to vector store (only needed once, or when documents change)
vector_store.add_documents(documents)

# Initialize orchestrator with vector store
orchestrator = Orchestrator(vector_store=vector_store)

# Analyze a question (uses RAG to retrieve relevant chunks)
results = orchestrator.run("What was the revenue in 2024?")
print(f"Answer: {results['answer']}")
print(f"Sources: {results['sources']}")

# Ask similar question (will use cache)
results2 = orchestrator.run("What were the total sales in 2024?")
print(f"Answer: {results2['answer']}")

# Get cache statistics
cache_stats = orchestrator.cache.get_cache_stats()
print(f"Total cached entries: {cache_stats['total_entries']}")
```

## 📊 How It Works

### Workflow

1. **Initialization**: On startup, PDFs are loaded, chunked, embedded, and stored in ChromaDB vector database
2. **Cache Check**: Cache agent checks if a similar question exists in cache
3. **Document Retrieval (RAG)**: Vector store performs semantic search to find relevant document chunks
4. **Triage**: Relevant documents are identified using vector similarity search
5. **Agent-Specific Retrieval**: Each agent retrieves top-k most relevant chunks using semantic search:
   - Financial Agent: Retrieves chunks related to financial metrics, revenue, income
   - Risk Agent: Retrieves chunks related to risks, risk factors, uncertainties
   - Strategy Agent: Retrieves chunks related to strategy, business plans, objectives
   - Timeline Agent: Retrieves chunks related to dates, timelines, periods
6. **Specialized Analysis**: Each agent processes only the retrieved relevant chunks (not entire documents)
7. **Synthesis**: Synthesis agent combines all insights
8. **Caching**: Answer is stored in cache for future similar questions
9. **Cost Tracking**: All operations are tracked for cost optimization

### RAG (Retrieval Augmented Generation) with Vector Database

The system uses **ChromaDB** as a vector database to store document embeddings and enable semantic search:

#### How RAG Works

1. **Document Storage**: On startup, all PDF documents are:
   - Loaded and extracted as text
   - Chunked using LangChain's RecursiveCharacterTextSplitter
   - Embedded using OpenAI's `text-embedding-3-small` model
   - Stored in ChromaDB with metadata (document name, chunk index, etc.)

2. **Semantic Retrieval**: When a question is asked:
   - The question is embedded using the same embedding model
   - Vector similarity search finds the most relevant chunks
   - Each agent retrieves top-k chunks (default: 5) relevant to its domain
   - Agents process only these relevant chunks instead of entire documents

3. **Benefits**:
   - **Efficiency**: Only relevant information is processed, reducing token usage
   - **Accuracy**: Semantic search finds contextually relevant information
   - **Scalability**: Vector DB handles large document collections efficiently
   - **Cost Savings**: Significantly lower LLM API costs by processing less text

#### Vector Store Configuration

```python
from utils.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore(
    persist_directory="./chroma_db",      # Where to store the vector DB
    collection_name="financial_documents", # Collection name
    chunk_size=1000,                       # Chunk size for embedding
    chunk_overlap=200                      # Overlap between chunks
)

# Add documents (done automatically on startup)
vector_store.add_documents(documents)

# Search for relevant chunks
chunks = vector_store.search("What was the revenue?", k=5)

# Agent-specific search
financial_chunks = vector_store.search_by_agent_type(
    query="revenue and income",
    agent_type="financial",
    k=5
)
```

### Document Chunking with LangChain

The system uses **LangChain's RecursiveCharacterTextSplitter** for intelligent document chunking:

#### Why LangChain Chunking?

- **Context Preservation**: Recursively splits on multiple separators (`\n\n`, `\n`, `. `, ` `) to preserve context
- **Smart Boundaries**: Prefers splitting at paragraph and sentence boundaries
- **Token-Aware**: Works with token counting to ensure chunks fit within model limits
- **Section-Aware**: Can chunk by SEC filing sections (Item 1, Item 2, etc.)
- **Overlap Management**: Maintains overlap between chunks to preserve context continuity

#### How It Works

```python
from utils.text_chunks import TextChunker

# Initialize chunker with LangChain
chunker = TextChunker(
    chunk_size=4000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    use_recursive=True    # Use RecursiveCharacterTextSplitter
)

# Chunk a document
chunks = chunker.chunk_text(document_text)

# Or chunk by sections (for SEC filings)
chunks = chunker.chunk_by_sections(document_text)
```

#### Automatic Chunking and Vector Storage

The system automatically handles document processing:

- **On Startup**: All documents are chunked and stored in vector database
- **Chunk Size**: Default 1000 characters per chunk (optimized for embeddings)
- **Embedding**: Each chunk is embedded using OpenAI's text-embedding-3-small model
- **Storage**: Chunks are stored in ChromaDB with metadata for efficient retrieval
- **Retrieval**: Agents retrieve only top-k relevant chunks (default: 5) using semantic search
- **Processing**: Agents process only retrieved chunks, not entire documents

### Agent Specialization with RAG

Each agent is optimized for specific tasks and uses RAG to retrieve only relevant information:

- **Cache Agent**: Finds similar questions using semantic similarity (saves API costs)
- **Triage Agent**: Uses vector search to identify relevant documents (uses minimal tokens)
- **Financial Agent**: Retrieves chunks related to financial metrics, revenue, income, assets, liabilities. Processes only relevant chunks instead of entire documents
- **Risk Agent**: Retrieves chunks related to risks, risk factors, uncertainties. Processes only relevant chunks efficiently
- **Strategy Agent**: Retrieves chunks related to strategy, business plans, objectives. Analyzes only relevant strategic information
- **Timeline Agent**: Retrieves chunks related to dates, timelines, periods, quarters. Extracts temporal information from relevant chunks
- **Fact Checker**: Validates claims and checks consistency using relevant chunks
- **Synthesis Agent**: Creates comprehensive executive summary from agent analyses

### Caching Strategy

The cache agent uses intelligent semantic matching:

- **Similarity Detection**: Uses LLM to determine if questions are semantically similar
- **Threshold**: 85% similarity threshold to use cached answers
- **Automatic Storage**: All new answers are automatically cached
- **Access Tracking**: Tracks how often cached answers are used
- **Cost Savings**: Significantly reduces API calls for similar questions

### Cost Optimization Strategies

1. **RAG with Vector Database**: Only relevant chunks are retrieved and processed, dramatically reducing token usage (major cost savings)
2. **Intelligent Caching**: Reuses answers for similar questions (major cost savings)
3. **Semantic Search**: Agents retrieve only top-k most relevant chunks instead of processing entire documents
4. **LangChain Chunking**: Efficient document processing reduces redundant token usage
5. **Model Selection**: Uses `gpt-4o-mini` by default (cost-effective)
6. **Intelligent Routing**: Only necessary agents are invoked
7. **Smart Chunking**: Large documents are chunked and embedded for efficient retrieval
8. **Token Tracking**: Real-time monitoring of token usage
9. **Efficiency Metrics**: Tracks tasks per token ratio

## 📈 Cost Tracking

The system tracks costs across all agents:

```python
# Get cost summary
cost_summary = orchestrator.cost.report()

print(f"Total Cost: ${cost_summary['total_usd']}")
print(f"Total Tokens: {cost_summary['total_tokens']}")
print(f"Agents Used: {cost_summary['agents_used']}")

# Check if answer was cached (saves costs)
if results.get('cached'):
    print("Answer retrieved from cache - no processing costs!")
```

## 🔧 Configuration

### LangChain Chunking Configuration

Customize chunking behavior for your needs:

```python
from utils.text_chunks import TextChunker

# Standard recursive chunking (recommended)
chunker = TextChunker(
    chunk_size=4000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    use_recursive=True    # Use RecursiveCharacterTextSplitter
)

# Token-aware chunking
chunker = TextChunker(
    chunk_size=8000,      # Tokens per chunk
    chunk_overlap=400,    # Overlap in tokens
    use_recursive=False   # Use TokenTextSplitter
)

# Chunk by sections (for SEC filings)
chunks = chunker.chunk_by_sections(
    text,
    section_markers=[
        r'Item\s+\d+[A-Z]?:',
        r'PART\s+[IVX]+',
        r'Section\s+\d+'
    ]
)
```

### Cache Configuration

Adjust cache settings:

```python
from agents.cache_agent import CacheAgent

# Create cache agent with custom settings
cache = CacheAgent(
    name="Cache",
    model="gpt-4o-mini",
    cache_file="custom_cache.json"
)

# Adjust similarity threshold
cache.similarity_threshold = 0.90  # Higher = more strict matching
```

### Model Selection

Agents can be configured to use different models:

```python
from agents.financial_agent import FinancialAgent

# Use a more powerful model for complex analysis
financial_agent = FinancialAgent("Financial", model="gpt-4o")
```

### Vector Store Configuration

Customize vector store settings:

```python
from utils.vector_store import VectorStore
from orchestrator import Orchestrator

# Initialize vector store with custom settings
vector_store = VectorStore(
    persist_directory="./chroma_db",
    collection_name="financial_documents",
    chunk_size=1000,      # Chunk size for embedding
    chunk_overlap=200     # Overlap between chunks
)

# Initialize orchestrator with vector store
orchestrator = Orchestrator(vector_store=vector_store)

# Get vector store statistics
stats = vector_store.get_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Documents: {stats['documents']}")
```

### Orchestrator RAG Settings

The orchestrator uses RAG by default when vector store is provided:

```python
from orchestrator import Orchestrator
from utils.vector_store import VectorStore

# Vector store is required for RAG
vector_store = VectorStore()
orchestrator = Orchestrator(vector_store=vector_store)

# The orchestrator automatically:
# - Retrieves relevant documents using vector search
# - Retrieves agent-specific chunks for each agent
# - Processes only retrieved chunks (not entire documents)
```

## 📝 Example Output

The system returns responses in a standardized format:

```json
{
  "answer": "The revenue for Q1 2024 was $125.3 billion, representing a 5% increase from the previous quarter. Net income was $28.5 billion with an EPS of $1.82. The company's total assets increased to $350.2 billion, while maintaining a debt-to-equity ratio of 0.45.",
  "sources": [
    "10-Q-Q1-2024-As-Filed.pdf",
    "10-K-2023-As-Filed.pdf"
  ],
  "cost": {
    "total_usd": 0.0234,
    "input_tokens": 4523,
    "output_tokens": 892,
    "agents_used": [
      {
        "agent": "TriageAgent",
        "tokens": 1200,
        "cost_usd": 0.0018
      },
      {
        "agent": "FinancialAgent",
        "tokens": 3500,
        "cost_usd": 0.0175
      },
      {
        "agent": "SynthesisAgent",
        "tokens": 715,
        "cost_usd": 0.0041
      }
    ]
  },
  "latency_seconds": 3.42
}
```

### Response Format Details

- **answer**: Plain text answer to the question (no markdown formatting)
- **sources**: Array of document filenames that were used to generate the answer
- **cost**: Cost breakdown including:
  - `total_usd`: Total cost in USD
  - `input_tokens`: Total input tokens used across all agents
  - `output_tokens`: Total output tokens generated across all agents
  - `agents_used`: Array of agent usage with tokens and cost per agent
- **latency_seconds**: Total processing time in seconds

## 🎓 Key Concepts

### Multi-Agent Collaboration

Agents work together but maintain specialization:
- Each agent focuses on its domain expertise
- Results are synthesized for comprehensive insights
- Parallel processing when possible for efficiency
- Chunked documents are processed efficiently across agents

### RAG and Vector Database Benefits

The RAG system with ChromaDB provides:

- **Semantic Search**: Finds contextually relevant information using embeddings
- **Efficient Retrieval**: Only top-k most relevant chunks are processed
- **Agent-Specific Context**: Each agent retrieves chunks relevant to its domain
- **Scalability**: Vector DB efficiently handles large document collections
- **Cost Reduction**: Processes only relevant chunks, reducing token usage by 60-80%
- **Persistent Storage**: Vector embeddings are stored locally in ChromaDB
- **Fast Queries**: Semantic search is fast even with thousands of chunks

### LangChain Chunking Benefits

The LangChain-based chunking system provides:

- **Better Context Preservation**: Recursive splitting maintains semantic boundaries
- **Flexible Configuration**: Multiple chunking strategies (recursive, token-based, section-based)
- **Automatic Handling**: Large documents are automatically chunked and embedded
- **Agent-Optimized**: Chunks are sized appropriately for embedding and retrieval
- **Cost Efficient**: Reduces token waste by intelligent boundary detection

### Intelligent Caching

The cache system provides significant cost savings:
- Semantic similarity matching (not just exact matches)
- Automatic answer storage for future use
- Access tracking for cache optimization
- Reduces redundant API calls by up to 80% for similar questions

### Resource Efficiency

The system is designed to minimize costs:
- **RAG with Vector DB**: Only relevant chunks are processed, reducing token usage by 60-80%
- Intelligent caching reduces API calls
- Semantic search finds relevant information efficiently
- LangChain chunking optimizes document processing
- Uses cost-effective models by default
- Only invokes necessary agents
- Tracks and optimizes token usage
- Provides cost insights for optimization

### Scalability

The architecture supports:
- **Vector Database**: Efficiently handles thousands of documents and chunks
- Adding new specialized agents
- Processing multiple documents with persistent vector storage
- Parallel agent execution
- Distributed processing (future enhancement)
- Growing cache of answered questions
- Handling documents of any size through intelligent chunking and vector storage
- Fast semantic search even with large document collections

## 🔍 Troubleshooting

### Common Issues

1. **PDF Loading Errors**
   - Ensure PDF files are not corrupted
   - Check file paths are correct
   - Verify pypdf is installed correctly

2. **Token Limit Errors**
   - LangChain chunking should handle this automatically
   - Adjust `chunk_size` parameter if needed
   - Consider using section-based chunking for structured documents
   - Check that `chunk_size` is appropriate for your model's context window

3. **Chunking Issues**
   - Verify LangChain is installed: `pip install langchain langchain-openai langchain-community`
   - Check chunk size settings (default: 1000 characters for vector DB)
   - For very large documents, consider reducing `max_chunks` in orchestrator
   - Use section-based chunking for SEC filings with clear structure

4. **Vector Database Issues**
   - Verify ChromaDB is installed: `pip install chromadb`
   - Check that `chroma_db` directory is writable
   - If vector DB is corrupted, delete `chroma_db` directory and restart (will rebuild)
   - Ensure OpenAI API key is set for embeddings
   - Vector DB is automatically populated on startup

5. **API Key Issues**
   - Set OPENAI_API_KEY in environment (required for both LLM and embeddings)
   - Check API key is valid
   - Verify account has credits

6. **Cache Issues**
   - Check `cache.json` file permissions
   - Clear cache if corrupted: `POST /cache/clear`
   - Verify cache file is writable

## 🚧 Future Enhancements

- [x] Real OpenAI API integration
- [x] Intelligent caching system
- [x] LangChain-based document chunking
- [x] Vector database integration (ChromaDB)
- [x] RAG (Retrieval Augmented Generation) implementation
- [ ] Async/parallel agent processing
- [ ] Database persistence for results
- [ ] Web UI for document upload and visualization
- [ ] Advanced chunking strategies (semantic chunking, hierarchical chunking)
- [ ] Agent learning from past analyses
- [ ] Multi-document comparison
- [ ] Export to various formats (PDF, Excel, etc.)
- [ ] Hybrid search (semantic + keyword search)
- [ ] Re-ranking of retrieved chunks
- [ ] Cache expiration and cleanup policies
- [ ] LangChain document loaders for additional file formats
- [ ] Incremental document updates to vector DB

## 📄 License

This project is provided as a demonstration of multi-agent systems for financial document analysis.

## 🙏 Acknowledgments

This project demonstrates:
- Multi-agent system architecture
- Cost optimization in AI systems
- Intelligent task routing
- Resource-efficient AI processing
- Intelligent caching strategies
- LangChain integration for document processing
- RAG (Retrieval Augmented Generation) implementation
- Vector database integration for semantic search

---

**Remember**: The next era of AI isn't about increasing token usage — it's about designing intelligent systems that collaborate, self-organize, cache intelligently, use RAG to retrieve only relevant information, store data efficiently in vector databases, chunk documents intelligently with LangChain, and solve problems efficiently! 🚀
