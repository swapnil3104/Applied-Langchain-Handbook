"""
Module 06 - Advanced RAG: 03_reranking_pipeline.py
Cross-encoder re-ranking for higher retrieval precision.
"""
from langchain_core.documents import Document

def mock_cross_encoder_rerank(query: str, documents: list) -> list:
    """Simulates a Cross-Encoder scoring each (query, document) pair."""
    scored_docs = []
    q_words = set(query.lower().split())
    for doc in documents:
        # Cross-encoder relevance heuristic
        content_words = set(doc.page_content.lower().split())
        overlap = len(q_words.intersection(content_words))
        score = overlap / (len(q_words) + 1)
        scored_docs.append((doc, score))
    
    # Sort descending by cross-encoder score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs

def main():
    print("=== Two-Stage Retrieval: Bi-Encoder Recall + Cross-Encoder Rerank ===")

    initial_candidates = [
        Document(page_content="Kubernetes pods communicate via overlay network CNI plugins.", metadata={"id": 1}),
        Document(page_content="Cross-encoder rerankers dramatically boost top-k retrieval precision.", metadata={"id": 2}),
        Document(page_content="Reranking models compare queries and documents jointly with full self-attention.", metadata={"id": 3}),
        Document(page_content="PostgreSQL full-text search uses tsvector and tsquery data types.", metadata={"id": 4})
    ]

    query = "How do cross-encoder rerankers improve retrieval precision?"
    print(f"Query: '{query}'")
    print(f"Initial Candidates: {len(initial_candidates)} documents")

    reranked = mock_cross_encoder_rerank(query, initial_candidates)

    print("\nReranked Results (Top Precision First):")
    for rank, (doc, score) in enumerate(reranked, 1):
        print(f"  Rank {rank} [Score: {score:.3f}]: {doc.page_content}")

if __name__ == "__main__":
    main()
