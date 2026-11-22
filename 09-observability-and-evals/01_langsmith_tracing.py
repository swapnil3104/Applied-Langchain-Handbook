"""
Module 10 - Observability and Evals: 01_langsmith_tracing.py
Tracing runs, latency, and token usage with custom metadata.
"""
import os
import time
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.runnables import RunnableConfig

load_dotenv()

@traceable(name="vector_similarity_search", run_type="retriever")
def search_vector_db(query: str, top_k: int = 3):
    """Simulated retriever step instrumented for LangSmith."""
    time.sleep(0.05) # simulate latency
    return [
        {"id": 1, "text": f"Grounded document matching '{query}'", "score": 0.92},
        {"id": 2, "text": "LangSmith records full execution call-trees automatically.", "score": 0.88}
    ]

@traceable(name="grounded_synthesis", run_type="llm")
def synthesize_response(query: str, docs: list):
    """Simulated LLM synthesis step instrumented with custom metadata."""
    time.sleep(0.08)
    return f"Synthesized answer for '{query}' based on {len(docs)} documents."

@traceable(name="enterprise_rag_pipeline", run_type="chain")
def run_rag_pipeline(user_query: str, session_id: str):
    print(f"Tracing pipeline execution for session: {session_id}")
    docs = search_vector_db(user_query)
    answer = synthesize_response(user_query, docs)
    return {"query": user_query, "answer": answer, "retrieved_count": len(docs)}

def main():
    print("=== LangSmith Tracing & Observability Setup ===")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    project_name = os.getenv("LANGSMITH_PROJECT", "default")
    
    print(f"LangSmith Tracing Active: {tracing_enabled}")
    print(f"LangSmith Project Target: {project_name}")

    # Execute instrumented pipeline
    result = run_rag_pipeline(
        user_query="What are the best practices for prompt engineering?",
        session_id="user_session_49102"
    )

    print("\nExecution Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if not tracing_enabled or not os.getenv("LANGSMITH_API_KEY"):
        print("\n[Tip] Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to visualize real-time spans in the LangSmith dashboard.")

if __name__ == "__main__":
    main()
