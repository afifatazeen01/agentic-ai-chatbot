import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF and extracts text page-by-page.
    Preserves page numbers and source in metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The PDF file at {file_path} was not found.")
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    cleaned_documents = []
    for doc in documents:
        # Clean unnecessary whitespace
        cleaned_text = " ".join(doc.page_content.split())
        
        # Ensure metadata is robust
        metadata = {
            "source": os.path.basename(file_path),
            "page": doc.metadata.get("page", 0) + 1  # 1-indexed for user readability
        }
        
        cleaned_documents.append(Document(page_content=cleaned_text, metadata=metadata))
        
    return cleaned_documents
