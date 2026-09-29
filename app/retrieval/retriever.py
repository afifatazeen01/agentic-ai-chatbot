import os
from typing import List, Dict, Any
from app.config import settings
from app.ingestion.indexer import get_embeddings

def get_retriever():
    embeddings = get_embeddings()
    
    if settings.vector_store.lower() == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
            namespace=settings.pinecone_namespace
        )
    else:
        from langchain_community.vectorstores import FAISS
        save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "faiss")
        
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Local FAISS index not found at {save_dir}. Run ingestion first.")
            
        return FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)

def retrieve_documents(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents based on semantic similarity.
    """
    if top_k is None:
        top_k = settings.top_k
        
    vectorstore = get_retriever()
    
    # similarity_search_with_relevance_scores normalizes FAISS L2 distance and Pinecone Cosine
    # to a 0.0 - 1.0 scale (higher is better)
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page", 0),
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score": float(score)
        })
        
    return formatted_results
