import requests
from app.graph.state import RAGState
from app.retrieval.retriever import retrieve_documents
from app.llm.prompts import generation_prompt_template
from app.evaluation.grounding import GroundingVerification, GROUNDING_VERIFICATION_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings

def get_llm():
    if settings.llm_provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.0
        )
    else:
        # Default to local Ollama
        from langchain_ollama import ChatOllama
        
        # Check if Ollama is available
        try:
            response = requests.get(settings.ollama_base_url)
            if response.status_code != 200:
                raise ConnectionError("Ollama returned non-200 status.")
        except Exception:
            raise RuntimeError("Local LLM is not available. Install/start Ollama and download the configured model.")
            
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0
        )

def retrieve_node(state: RAGState) -> dict:
    query = state["query"]
    docs = retrieve_documents(query)
    scores = [d["score"] for d in docs]
    return {"retrieved_documents": docs, "retrieval_scores": scores}

def relevance_check_node(state: RAGState) -> dict:
    scores = state.get("retrieval_scores", [])
    if scores:
        avg_score = sum(scores) / len(scores)
    else:
        avg_score = 0.0
    should_refuse = avg_score < settings.relevance_threshold
    return {"relevance_score": avg_score, "should_refuse": should_refuse}

def generate_answer_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state["retrieved_documents"]
    context = "\n\n".join([f"Page {d['page']}: {d['text']}" for d in docs])
    
    llm = get_llm()
    chain = generation_prompt_template | llm
    
    response = chain.invoke({"query": query, "context": context})
    return {"final_answer": response.content}

def grounding_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state["retrieved_documents"]
    answer = state["final_answer"]
    
    if "not available in the provided eBook" in answer.lower() or "not available" in answer.lower():
        return {"grounded": True, "reasoning_summary": "Answer is a valid refusal."}
        
    context = "\n\n".join([f"Page {d['page']}: {d['text']}" for d in docs])
    
    llm = get_llm()
    
    try:
        # Many small Ollama models might struggle with strict structured output.
        # LangChain attempts to coerce it. We use the official with_structured_output.
        structured_llm = llm.with_structured_output(GroundingVerification)
        prompt = ChatPromptTemplate.from_template(GROUNDING_VERIFICATION_PROMPT)
        chain = prompt | structured_llm
        
        verification = chain.invoke({
            "query": query,
            "context": context,
            "answer": answer
        })
        
        pages = list(set([str(d['page']) for d in docs]))
        reasoning = f"Answer supported by pages {', '.join(pages)}." if verification.grounded else f"Unsupported claims: {', '.join(verification.unsupported_claims)}"
        
        return {"grounded": verification.grounded, "reasoning_summary": reasoning}
    except Exception as e:
        # Fallback if the local model fails to generate valid JSON structured output
        print(f"Warning: Grounding check structured output failed ({str(e)}). Defaulting to safe refusal.")
        return {"grounded": False, "reasoning_summary": f"Structured evaluation failed: {str(e)}"}

def refusal_node(state: RAGState) -> dict:
    return {
        "final_answer": "I couldn't find information about this in the provided Agentic AI eBook.",
        "grounded": True,
        "reasoning_summary": "Query out of scope; rejected due to low relevance threshold or grounding failure.",
        "confidence_score": 0.1
    }

def confidence_node(state: RAGState) -> dict:
    if state.get("should_refuse"):
        return {"confidence_score": 0.1}
        
    rel_score = state.get("relevance_score", 0.0)
    grounded = state.get("grounded", False)
    
    capped_rel = min(rel_score, 1.0)
    confidence = (capped_rel * 0.6) + (0.4 if grounded else 0.0)
    
    if not grounded:
        confidence *= 0.5
        
    return {"confidence_score": round(confidence, 2)}
