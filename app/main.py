from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Agentic AI RAG API",
    description="API for querying the Agentic AI eBook using RAG",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
