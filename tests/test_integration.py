import os
import pytest
from app.graph.workflow import rag_app
from app.config import settings
import requests

def is_ollama_running():
    try:
        response = requests.get(settings.ollama_base_url)
        return response.status_code == 200
    except Exception:
        return False

def has_faiss_index():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "storage", "faiss")
    return os.path.exists(save_dir)

@pytest.mark.skipif(not is_ollama_running() or not has_faiss_index(), reason="Requires Ollama running and FAISS index")
def test_out_of_scope_behavior():
    query = "What is the capital of France?"
    result = rag_app.invoke({"query": query})
    
    # Should refuse
    assert result["should_refuse"] is True or result["grounded"] is False
    assert "I couldn't find information about this" in result["final_answer"]
    assert result["confidence_score"] <= 0.5
