import os
from typing import List
from langchain_core.documents import Document
from app.config import settings

def get_embeddings():
    if settings.embedding_provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.openai_api_key)
    else:
        # Default to local sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"Loading local embedding model: {settings.embedding_model}...")
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

def index_chunks(chunks: List[Document]):
    """Embed and store chunks in the configured vector store."""
    if not chunks:
        print("No chunks to index.")
        return
        
    embeddings = get_embeddings()
    
    if settings.vector_store.lower() == "pinecone":
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore
        
        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if settings.pinecone_index_name not in existing_indexes:
            print(f"Creating Pinecone index: {settings.pinecone_index_name}...")
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=1536, # Assuming openai embeddings if pinecone is used
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
        print(f"Indexing {len(chunks)} chunks into Pinecone...")
        PineconeVectorStore.from_documents(
            chunks, 
            embeddings, 
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace
        )
        print("Pinecone indexing complete.")
        
    else:
        # Default to local FAISS
        from langchain_community.vectorstores import FAISS
        
        print(f"Indexing {len(chunks)} chunks into local FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save locally
        save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "faiss")
        os.makedirs(save_dir, exist_ok=True)
        
        vectorstore.save_local(save_dir)
        print(f"FAISS index saved locally at {save_dir}")
