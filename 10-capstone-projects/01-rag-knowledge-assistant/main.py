"""
Capstone 01: FastAPI Grounded RAG Assistant with Source Citations.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from rag_service import KnowledgeRAGService

app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
    description="Grounded knowledge retrieval service with citation tracking."
)

rag_service = KnowledgeRAGService()

class QueryRequest(BaseModel):
    question: str = Field(..., description="User prompt or question")
    session_id: str = Field("default_session", description="Unique session identifier for memory tracking")

class SourceCitation(BaseModel):
    source: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]
    session_id: str

@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "indexed_documents": rag_service.get_doc_count()}

@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    result = rag_service.answer_query(request.question, request.session_id)
    return result

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Knowledge Assistant on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
