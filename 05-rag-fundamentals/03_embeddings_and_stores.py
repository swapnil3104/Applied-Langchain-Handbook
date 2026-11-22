"""
Module 05 - RAG Fundamentals: 03_embeddings_and_stores.py
Generating embeddings, indexing in FAISS / Chroma, and configuring advanced
retrieval algorithms including Maximal Marginal Relevance (MMR) and score thresholds.
"""
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    print("=== Indexing Documents in FAISS Vector Store ===")

    docs = [
        Document(
            page_content="LangChain standardizes LCEL pipelines, unified ChatModels, and structured output parsing.",
            metadata={"topic": "core", "source": "overview.md"}
        ),
        Document(
            page_content="LangChain LCEL syntax uses the pipe operator (|) to compose prompts, models, and parsers.",
            metadata={"topic": "core", "source": "lcel_syntax.md"}  # Intentionally similar to test MMR deduplication
        ),
        Document(
            page_content="Vector stores index embeddings for nearest-neighbor semantic search (HNSW, Cosine, L2).",
            metadata={"topic": "retrieval", "source": "vectors.md"}
        ),
        Document(
            page_content="FastAPI allows high-throughput asynchronous serving of LangChain RAG pipelines.",
            metadata={"topic": "deployment", "source": "serving.md"}
        ),
        Document(
            page_content="RunnableWithMessageHistory manages conversational state across multi-tenant sessions.",
            metadata={"topic": "memory", "source": "memory.md"}
        )
    ]

    # Deterministic 64-dimensional mock embeddings for offline reliability
    embeddings = FakeEmbeddings(size=64)

    # 1. Ingest documents into FAISS Vector Store
    vector_store = FAISS.from_documents(docs, embeddings)
    print(f"Successfully indexed {len(docs)} documents into FAISS vector store.\n")

    # 2. Basic Similarity Search with Distance Scores
    query = "How do we write LCEL pipelines in LangChain?"
    print(f"--- 1. Raw Similarity Search with Distance Scores for Query: '{query}' ---")
    raw_results = vector_store.similarity_search_with_score(query, k=2)
    for doc, score in raw_results:
        print(f"  [Distance Score: {score:.4f}] {doc.page_content[:75]}... (Source: {doc.metadata['source']})")

    # 3. Converting to standard Retriever: vector_store.as_retriever()
    print("\n--- 2. Converting Vector Store to Retriever Interface ---")
    standard_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = standard_retriever.invoke(query)
    print(f"Standard Retriever fetched {len(retrieved_docs)} documents.")

    # 4. Maximal Marginal Relevance (MMR) Retrieval
    # MMR balances relevance with diversity to avoid returning near-duplicate documents.
    # lambda_mult: 1.0 = pure relevance, 0.0 = maximum diversity. Default 0.5 is ideal for RAG.
    print("\n--- 3. Maximal Marginal Relevance (MMR) Search ---")
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,          # Number of diverse documents to return
            "fetch_k": 4,    # Initial pool of relevant candidates to consider
            "lambda_mult": 0.5  # Diversity factor (0 = max diversity, 1 = max similarity)
        }
    )
    mmr_docs = mmr_retriever.invoke(query)
    print(f"MMR Retriever selected {len(mmr_docs)} diverse documents:")
    for idx, d in enumerate(mmr_docs, 1):
        print(f"  Doc {idx}: {d.page_content[:70]}... (Topic: {d.metadata['topic']})")

    # 5. Score-Thresholded Retrieval
    print("\n--- 4. Similarity Score Threshold Filter ---")
    threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2}
    )
    threshold_docs = threshold_retriever.invoke(query)
    print(f"Filtered {len(threshold_docs)} documents meeting distance threshold.")

    # 6. Index Persistence (Save to & Load from Disk)
    print("\n--- 5. Vector Store Local Persistence ---")
    # In FAISS: vector_store.save_local("local_index") & FAISS.load_local("local_index", embeddings)
    # In Chroma: Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("FAISS persistence methods:")
    print("  Save: vector_store.save_local('faiss_index')")
    print("  Load: FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)")


if __name__ == "__main__":
    main()
