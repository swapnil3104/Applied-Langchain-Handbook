"""
Module 06 - Advanced RAG: 01_hybrid_search.py
Combining BM25 keyword search with dense vector retrieval via EnsembleRetriever.
"""
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.retrievers import EnsembleRetriever

def main():
    print("=== Hybrid Search: BM25 (Keyword) + FAISS (Dense Vector) ===")

    corpus = [
        Document(page_content="Error code ERR-8092 indicates connection pool starvation.", metadata={"type": "error_log"}),
        Document(page_content="To tune connection pools, increase max_connections in postgresql.conf.", metadata={"type": "guide"}),
        Document(page_content="Database performance drops under heavy concurrent load.", metadata={"type": "troubleshooting"}),
        Document(page_content="Vector database latency is affected by quantization settings.", metadata={"type": "vector_guide"})
    ]

    # 1. Sparse BM25 Keyword Retriever
    bm25_retriever = BM25Retriever.from_documents(corpus)
    bm25_retriever.k = 2

    # 2. Dense Vector Retriever
    embeddings = FakeEmbeddings(size=64)
    faiss_store = FAISS.from_documents(corpus, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 2})

    # 3. Hybrid Ensemble (Reciprocal Rank Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    query = "What does error ERR-8092 mean?"
    results = ensemble_retriever.invoke(query)

    print(f"Query: '{query}'")
    print(f"Retrieved {len(results)} merged documents:")
    for idx, doc in enumerate(results, 1):
        print(f"  [{idx}]: {doc.page_content} (Type: {doc.metadata['type']})")

if __name__ == "__main__":
    main()
