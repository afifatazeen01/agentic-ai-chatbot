from app.api.routes import ChatResponse, RetrievedChunk

def test_chat_response_schema():
    # Ensure Pydantic validates the expected response format
    chunk = RetrievedChunk(
        text="Test chunk",
        page=1,
        source="test.pdf",
        score=0.9
    )
    
    response = ChatResponse(
        query="What is AI?",
        final_answer="AI is artificial intelligence.",
        retrieved_context_chunks=[chunk],
        confidence_score=0.95
    )
    
    assert response.query == "What is AI?"
    assert response.confidence_score == 0.95
    assert len(response.retrieved_context_chunks) == 1
