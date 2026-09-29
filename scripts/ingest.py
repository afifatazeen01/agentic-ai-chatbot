import os
import sys

# Ensure app is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.ingestion.indexer import index_chunks

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Ebook-Agentic-AI.pdf")

def main():
    print(f"Loading PDF from {PDF_PATH}...")
    try:
        documents = load_pdf(PDF_PATH)
        print(f"Loaded {len(documents)} pages.")
        
        print("Chunking documents...")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        
        print("Indexing chunks to Pinecone...")
        index_chunks(chunks)
        
        print("Ingestion complete.")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()
