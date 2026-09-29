import pytest
from app.graph.state import RAGState
from app.graph.nodes import relevance_check_node, confidence_node

def test_relevance_check_refusal():
    # Below threshold (0.65)
    state: RAGState = {
        "retrieval_scores": [0.5, 0.4, 0.6]
    }
    result = relevance_check_node(state)
    assert result["should_refuse"] is True
    assert result["relevance_score"] == 0.5

def test_relevance_check_pass():
    # Above threshold (0.65)
    state: RAGState = {
        "retrieval_scores": [0.8, 0.9, 0.75]
    }
    result = relevance_check_node(state)
    assert result["should_refuse"] is False
    
def test_confidence_score_range():
    # Ideal scenario
    state: RAGState = {
        "relevance_score": 0.9,
        "grounded": True,
        "should_refuse": False
    }
    result = confidence_node(state)
    score = result["confidence_score"]
    assert 0 <= score <= 1
    assert score > 0.8
    
    # Un-grounded scenario
    state2: RAGState = {
        "relevance_score": 0.9,
        "grounded": False,
        "should_refuse": False
    }
    result2 = confidence_node(state2)
    assert result2["confidence_score"] < 0.5
