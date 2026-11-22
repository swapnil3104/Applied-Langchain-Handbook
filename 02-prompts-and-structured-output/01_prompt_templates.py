"""
Module 02 - Prompts & Structured Output: 01_prompt_templates.py
Demonstrates ChatPromptTemplate, MessagesPlaceholder, and dynamic variables.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def main():
    print("=== Composing ChatPromptTemplate with MessagesPlaceholder ===")
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant specialized in {domain}. Reply concisely."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

    history = [
        HumanMessage(content="What is a vector index?"),
        AIMessage(content="A data structure optimized for fast approximate nearest neighbor (ANN) search over embeddings.")
    ]

    prompt_value = template.format_messages(
        domain="Information Retrieval",
        chat_history=history,
        user_input="How does HNSW compare to Flat indexing?"
    )

    print("Formatted message list:")
    for msg in prompt_value:
        print(f"[{msg.type.upper()}]: {msg.content}")

if __name__ == "__main__":
    main()
