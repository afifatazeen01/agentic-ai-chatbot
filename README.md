# Agentic AI RAG Chatbot

## Problem Statement
Developing a production-ready Retrieval-Augmented Generation (RAG) system requires more than just connecting an LLM to a vector database. Challenges like hallucinations, ungrounded responses, and out-of-scope inquiries need to be addressed robustly.

## Objective
To build an enterprise-grade RAG chatbot for the *Agentic AI eBook*. The system explicitly focuses on grounded answer generation, robust hallucination verification, explicit confidence scoring, and strict out-of-scope question refusal.

## Zero-Cost Local Architecture (Default)
This assignment was originally designed for Pinecone and OpenAI. However, to support **100% free, local, zero-cost execution**, the architecture has been implemented with robust abstractions.

By default, the application runs using:
- **Vector Store**: [FAISS](https://faiss.ai/) (Local, CPU-based)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) via HuggingFace
- **LLM**: [Ollama](https://ollama.com/) (Local, `llama3.1` recommended)

### Why FAISS and Sentence Transformers?
The assignment permits alternative vector databases. FAISS provides excellent local performance without any cloud dependencies or API keys. Combined with `sentence-transformers`, we can ingest and retrieve documents entirely offline at zero cost.

### Why Ollama?
Ollama provides a lightweight runner for local LLMs. LangGraph's logic (including Pydantic structured output for grounding checks) works locally using Ollama models like `llama3.1`.

### Cloud Configuration (Optional)
The original Pinecone and OpenAI integrations are strictly preserved and configurable via `.env`.

---

## Core Pipeline Architecture
The system uses LangGraph to orchestrate the RAG flow through a StateGraph, allowing explicit conditional routing and verification. 

```text
             USER QUERY
                  |
                  v
          Pinecone / FAISS Retrieval
                  |
                  v
         Retrieved Chunks
                  |
                  v
        Relevance Evaluation  (Threshold: 0.65)
                  |
         +--------+--------+
         |                 |
    Relevant          Not Relevant (Out-of-Scope)
         |                 |
         v                 v
  Answer Generation     Refusal
         |
         v
  Grounding Check (LLM as a Judge)
         |
   +-----+------+
   |            |
Grounded      Not Grounded
   |            |
   v            v
Final Answer   Refusal
   |
   v
Confidence Score Calculation
   |
   v
Structured JSON Response via API / Streamlit UI
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`. 
By default, `.env` is configured for local FAISS and Ollama. No API keys are required!

### 3. Install Ollama (Required for Local LLM)
If you are using the zero-cost local path, you must install Ollama:
1. Download from [ollama.com](https://ollama.com)
2. Start Ollama
3. Download the model:
   ```bash
   ollama run llama3.1
   ```

### 4. PDF Ingestion
Place `Ebook-Agentic-AI.pdf` into the `data/` directory. Then run the ingestion script to chunk, embed (using `sentence-transformers`), and build the FAISS index locally:
```bash
python scripts/ingest.py
```
*Note: The first time you run this, it will download the embedding model from HuggingFace.*

---

## Running the Application

### 1. Run the FastAPI Backend
```bash
python run.py --api
```
*API available at http://127.0.0.1:8000/docs*

### 2. Run the Streamlit UI
```bash
python run.py --ui
```
*UI available at http://localhost:8501*

### 3. Run Evaluation Script
Test the 5 core questions and the mandatory out-of-scope question:
```bash
python scripts/evaluate.py
```

### 4. Run Pytest
Run the test suite (includes FAISS specific tests and Ollama integration tests):
```bash
pytest tests/
```

---

## Limitations & Future Improvements
1. **Local Model Grounding Checks**: Smaller local models sometimes struggle with strict JSON generation for the grounding check. LangChain's structured output attempts to handle this, but if it fails, the system safely falls back to a refusal.
2. **Deduplication**: Currently, multiple overlapping chunks might be retrieved. Future versions could implement Maximal Marginal Relevance (MMR) for diversity.
