"""
Module 05 - RAG Fundamentals: 04_basic_rag_chain.py
End-to-end retrieval and grounded generation pipeline using LCEL.
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

def format_docs(docs):
    return "\n\n".join(f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs))

def mock_generator(prompt_value):
    p_str = prompt_value.to_string()
    return f"Grounded Answer based on retrieved context:\nAccording to provided enterprise guidelines, LangChain v1.0 establishes deterministic agent graphs and robust persistence."

def main():
    print("=== End-to-End Grounded RAG Chain ===")

    knowledge_base = [
        Document(page_content="LangChain v0.3 standardizes LCEL pipelines and conversational memory persistence.", metadata={"source": "whitepaper.pdf"}),
        Document(page_content="Deterministic guardrails must wrap LLM tool executions in production.", metadata={"source": "security_policy.md"}),
        Document(page_content="Vector retrieval uses approximate nearest neighbor algorithms like HNSW.", metadata={"source": "ir_primer.md"})
    ]

    vector_store = FAISS.from_documents(knowledge_base, FakeEmbeddings(size=64))
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    rag_prompt = ChatPromptTemplate.from_template("""You are an enterprise AI assistant.
Answer the question strictly based only on the following retrieved context:

{context}

Question: {question}
Answer:""")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | RunnableLambda(mock_generator)
        | StrOutputParser()
    )

    query = "What are the production requirements for LangChain v1.0?"
    response = rag_chain.invoke(query)
    print(f"User Query: {query}\n")
    print(response)

if __name__ == "__main__":
    main()
