import os
from fastapi import FastAPI
from orchestrator import Orchestrator
from utils.pdf_loader import load_pdf
from utils.vector_store import VectorStore

app = FastAPI()

# Initialize vector store
vector_store = VectorStore(
    persist_directory="./chroma_db",
    collection_name="financial_documents",
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize orchestrator with vector store
orchestrator = Orchestrator(vector_store=vector_store)

# Load PDF files from data folder and populate vector DB
print("Loading PDF files and populating vector database...")
documents = {}
for file in os.listdir("/Users/akash.mishra/MyProject/ai-agent/financial-intelligence-agent/data"):
    if file.endswith(".pdf"):
        print(f"Loading {file}...")
        documents[file] = load_pdf(f"/Users/akash.mishra/MyProject/ai-agent/financial-intelligence-agent/data/{file}")

# Add documents to vector store
if documents:
    num_chunks = vector_store.add_documents(documents)
    print(f"Vector database initialized with {num_chunks} chunks from {len(documents)} documents")
    stats = vector_store.get_stats()
    print(f"Vector DB stats: {stats}")
else:
    print("Warning: No PDF files found in data folder")


# orchestrator.run(question = "What was the revenue in Q1 2023?")

@app.get("/ask")
def ask(q: str):
    return orchestrator.run(q)


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)  # pyright: ignore[reportUndefinedVariable]
