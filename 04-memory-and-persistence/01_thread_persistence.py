"""
Module 04 - Memory and Persistence: 01_thread_persistence.py
Session-based memory management in pure LangChain using `RunnableWithMessageHistory`
and `InMemoryChatMessageHistory`.
"""
import os
from typing import Dict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Global in-memory multi-tenant session storage
# In production, replace with RedisChatMessageHistory or PostgresChatMessageHistory
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Factory function returning the isolated message history for a given session."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


def main():
    print("=== Pure LangChain Session Memory with RunnableWithMessageHistory ===")

    # 1. Prompt template incorporating conversation history via MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent enterprise systems engineer. Answer concisely based on conversation context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[Mock Mode] Walking through RunnableWithMessageHistory lifecycle:")
        
        # Simulating Thread 1: User Alice
        print("\n--- Thread 1: Session 'user_alice_42' ---")
        alice_hist = get_session_history("user_alice_42")
        alice_hist.add_user_message("Hello, my primary database cluster is located in us-east-1.")
        alice_hist.add_ai_message("Acknowledged. I've noted that your primary DB is in us-east-1.")
        
        # Next turn for Alice
        alice_hist.add_user_message("What region is my database in?")
        # Simulated model response using history
        alice_response = "Your primary database cluster is located in us-east-1."
        alice_hist.add_ai_message(alice_response)
        
        print(f"Alice asked: 'What region is my database in?'")
        print(f"Model replied: '{alice_response}'")
        print(f"Alice Total Message Turns: {len(alice_hist.messages)}")

        # Simulating Thread 2: User Bob (Strict multi-tenant isolation)
        print("\n--- Thread 2: Session 'user_bob_99' (Tenant Isolation) ---")
        bob_hist = get_session_history("user_bob_99")
        bob_hist.add_user_message("What region is my database in?")
        bob_response = "I don't have any record of your database region yet. Could you please specify it?"
        bob_hist.add_ai_message(bob_response)
        
        print(f"Bob asked: 'What region is my database in?'")
        print(f"Model replied: '{bob_response}'")
        print(f"Bob Total Message Turns: {len(bob_hist.messages)}")
        print("\n[Verification] Alice and Bob states are completely isolated across session IDs.")
        return

    # Real LLM Execution when API key is provided
    from langchain.chat_models import init_chat_model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    raw_chain = prompt | model | StrOutputParser()

    conversational_chain = RunnableWithMessageHistory(
        raw_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config_alice = {"configurable": {"session_id": "alice_prod_session"}}
    res1 = conversational_chain.invoke(
        {"input": "Hi! Remember that our target deployment VPC is vpc-0a8912c."},
        config=config_alice
    )
    print(f"Turn 1 Response: {res1}")

    res2 = conversational_chain.invoke(
        {"input": "Which VPC are we deploying to?"},
        config=config_alice
    )
    print(f"Turn 2 Response: {res2}")


if __name__ == "__main__":
    main()
