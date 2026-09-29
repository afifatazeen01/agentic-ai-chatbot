from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Core RAG Settings
    top_k: int = Field(default=5)
    relevance_threshold: float = Field(default=0.65)
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=100)
    
    # Providers
    vector_store: str = Field(default="faiss")  # faiss or pinecone
    embedding_provider: str = Field(default="local")  # local or openai
    llm_provider: str = Field(default="ollama")  # ollama or openai
    
    # Local Models
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    ollama_model: str = Field(default="llama3.1")
    ollama_base_url: str = Field(default="http://localhost:11434")
    
    # Cloud Models (Optional)
    openai_api_key: str = Field(default="")
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="agentic-ai-index")
    pinecone_namespace: str = Field(default="default")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
