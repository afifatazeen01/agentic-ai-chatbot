from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict):
    query: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    relevance_score: float
    final_answer: str
    confidence_score: float
    grounded: bool
    should_refuse: bool
    reasoning_summary: str
