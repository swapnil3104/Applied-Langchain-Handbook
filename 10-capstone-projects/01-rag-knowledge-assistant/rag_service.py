"""
Capstone 01: RAG Service Implementation with Vector Index & Citations.
"""
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

class KnowledgeRAGService:
    def __init__(self):
        # Seed knowledge base
        self.documents = [
            Document(page_content="LangChain runnables provide a unified interface for streaming, batching, and async execution.", metadata={"source": "lcel_architecture.pdf"}),
            Document(page_content="LCEL provides declarative pipe composition for deterministic LLM execution.", metadata={"source": "lcel_specification.md"}),
            Document(page_content="Conversational memory in LangChain relies on RunnableWithMessageHistory and chat message history backends.", metadata={"source": "persistence_guide.md"})
        ]
        self.vector_store = FAISS.from_documents(self.documents, FakeEmbeddings(size=64))
        self.session_memory: Dict[str, List[str]] = {}

    def get_doc_count(self) -> int:
        return len(self.documents)

    def answer_query(self, question: str, session_id: str) -> dict:
        retrieved_docs = self.vector_store.similarity_search(question, k=2)
        citations = [
            {"source": d.metadata.get("source", "unknown"), "snippet": d.page_content}
            for d in retrieved_docs
        ]

        # Record conversation history
        history = self.session_memory.setdefault(session_id, [])
        history.append(f"User: {question}")

        answer = (
            f"Based on enterprise documentation ({', '.join(c['source'] for c in citations)}), "
            f"LangChain v1.0 standardizes reliable RAG execution and stateful graph management."
        )

        history.append(f"AI: {answer}")

        return {
            "answer": answer,
            "citations": citations,
            "session_id": session_id
        }
