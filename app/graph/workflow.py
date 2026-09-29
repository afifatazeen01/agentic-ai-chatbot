from langgraph.graph import StateGraph, START, END
from app.graph.state import RAGState
from app.graph.nodes import (
    retrieve_node,
    relevance_check_node,
    generate_answer_node,
    grounding_check_node,
    refusal_node,
    confidence_node
)

def create_workflow() -> StateGraph:
    """
    Creates and compiles the RAG LangGraph workflow.
    """
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("relevance_check", relevance_check_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("grounding_check", grounding_check_node)
    workflow.add_node("refusal", refusal_node)
    workflow.add_node("confidence", confidence_node)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "relevance_check")

    # Conditional routing based on relevance
    def route_relevance(state: RAGState):
        if state.get("should_refuse"):
            return "refusal"
        return "generate"

    workflow.add_conditional_edges(
        "relevance_check",
        route_relevance,
        {
            "generate": "generate",
            "refusal": "refusal"
        }
    )

    workflow.add_edge("generate", "grounding_check")
    
    # Conditional routing based on grounding check
    def route_grounding(state: RAGState):
        if not state.get("grounded"):
            # Prefer safer grounded refusal if verification fails
            return "refusal"
        return "confidence"
        
    workflow.add_conditional_edges(
        "grounding_check",
        route_grounding,
        {
            "confidence": "confidence",
            "refusal": "refusal"
        }
    )

    workflow.add_edge("refusal", END)
    workflow.add_edge("confidence", END)

    return workflow.compile()

rag_app = create_workflow()
