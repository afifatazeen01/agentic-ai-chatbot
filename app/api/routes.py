from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from app.graph.workflow import rag_app

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class RetrievedChunk(BaseModel):
    text: str
    page: int
    source: str
    score: float

class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[RetrievedChunk]
    confidence_score: float

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        # Run LangGraph workflow
        initial_state = {"query": request.query}
        final_state = rag_app.invoke(initial_state)
        
        # Format chunks
        retrieved_chunks = []
        if not final_state.get("should_refuse"):
            for chunk in final_state.get("retrieved_documents", []):
                retrieved_chunks.append(RetrievedChunk(
                    text=chunk["text"],
                    page=chunk["page"],
                    source=chunk["source"],
                    score=chunk["score"]
                ))
                
        return ChatResponse(
            query=final_state["query"],
            final_answer=final_state.get("final_answer", ""),
            retrieved_context_chunks=retrieved_chunks,
            confidence_score=final_state.get("confidence_score", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy"}
