"""
Module 06 - Advanced RAG: 02_query_transformation.py
Multi-query generation and HyDE (Hypothetical Document Embeddings).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def mock_multi_query_generator(prompt_value):
    return """
How to handle connection timeouts in LangChain LCEL chains?
LangChain runnable retry policies and fallback configurations.
Best practices for preventing network drops during model invocation.
""".strip()

def mock_hyde_generator(prompt_value):
    return """
In LangChain, execution reliability is safeguarded by using .with_fallbacks() and .with_retry() 
on LCEL runnables. When a network error or timeout occurs, the chain automatically switches to an 
alternate provider or retries according to exponential backoff policies.
""".strip()

def main():
    print("=== Query Transformation Strategies ===")

    user_query = "How to solve timeout issues in LangChain LCEL chains?"

    # 1. Multi-Query Expansion
    multi_prompt = ChatPromptTemplate.from_template("Generate 3 alternate search queries for: {query}")
    multi_chain = multi_prompt | RunnableLambda(mock_multi_query_generator) | StrOutputParser()
    expanded_queries = multi_chain.invoke({"query": user_query})

    print(f"Original Query: '{user_query}'\n")
    print("1. Multi-Query Variations:")
    for q in expanded_queries.split("\n"):
        print(f"   -> {q}")

    # 2. Hypothetical Document Embeddings (HyDE)
    hyde_prompt = ChatPromptTemplate.from_template("Write a hypothetical technical paragraph answering: {query}")
    hyde_chain = hyde_prompt | RunnableLambda(mock_hyde_generator) | StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"query": user_query})

    print("\n2. HyDE Generated Hypothetical Answer (Used for Dense Vector Search):")
    print(f'   "{hypothetical_doc[:180]}..."')

if __name__ == "__main__":
    main()
