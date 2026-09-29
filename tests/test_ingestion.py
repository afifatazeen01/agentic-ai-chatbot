import os
import pytest
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_documents
from app.ingestion.indexer import index_chunks
from app.retrieval.retriever import retrieve_documents
from langchain_core.documents import Document

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Ebook-Agentic-AI.pdf")

def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent.pdf")

def test_chunk_documents():
    docs = [
        Document(
            page_content="This is a test document. " * 50,
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "page" in chunk.metadata

# FAISS testing without requiring Ollama
def test_faiss_indexing_and_retrieval(tmp_path, monkeypatch):
    # Mock settings to ensure FAISS is used
    monkeypatch.setattr("app.config.settings.vector_store", "faiss")
    monkeypatch.setattr("app.config.settings.embedding_provider", "local")
    
    # Override save directory for test isolation
    test_faiss_dir = tmp_path / "storage" / "faiss"
    
    def mock_save(self, save_dir):
        os.makedirs(test_faiss_dir, exist_ok=True)
        # We don't actually need to save to disk for this simple test to verify the function executes
        pass

    import langchain_community.vectorstores.faiss as faiss_module
    monkeypatch.setattr(faiss_module.FAISS, "save_local", mock_save)
    
    # Create tiny chunks
    docs = [
        Document(page_content="Agentic AI is the future.", metadata={"page": 1, "source": "test.pdf", "chunk_id": "1"}),
        Document(page_content="The sky is blue.", metadata={"page": 2, "source": "test.pdf", "chunk_id": "2"})
    ]
    
    # Index
    # Note: SentenceTransformers will be downloaded to cache if not present. It runs locally.
    index_chunks(docs)
    
    # We mock retrieval to avoid needing to load from disk in test
    def mock_get_retriever():
        embeddings = __import__("app.ingestion.indexer").ingestion.indexer.get_embeddings()
        return faiss_module.FAISS.from_documents(docs, embeddings)
        
    monkeypatch.setattr("app.retrieval.retriever.get_retriever", mock_get_retriever)
    
    results = retrieve_documents("What is Agentic AI?", top_k=1)
    
    assert len(results) == 1
    assert "future" in results[0]["text"]
    assert results[0]["page"] == 1
    assert "score" in results[0]
