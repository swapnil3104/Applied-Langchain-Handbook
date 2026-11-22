"""
Module 01 - Foundations: 01_init_chat_model.py
Demonstrates provider-agnostic initialization with `init_chat_model()`.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import FakeListChatModel

load_dotenv()

def get_chat_model(model: str = "gpt-4o-mini", provider: str = "openai"):
    """Dynamically returns the requested ChatModel or falls back to FakeListChatModel if no API key is provided."""
    key_name = f"{provider.upper()}_API_KEY"
    if not os.getenv(key_name):
        print(f"[Fallback] {key_name} not found. Using simulated FakeListChatModel.")
        return FakeListChatModel(responses=[
            "LangChain v1.0 provider-agnostic init_chat_model() decouples business logic from specific LLM vendors."
        ])
    return init_chat_model(model, model_provider=provider, temperature=0.3)

def main():
    print("=== Initializing ChatModel via init_chat_model() ===")
    model = get_chat_model()
    
    messages = [HumanMessage(content="Explain provider-agnostic initialization in LangChain.")]
    response = model.invoke(messages)
    
    print(f"\nResponse Type: {type(response).__name__}")
    print(f"Response Content:\n{response.content}\n")
    if hasattr(response, "response_metadata") and response.response_metadata:
        print(f"Metadata: {response.response_metadata}")

if __name__ == "__main__":
    main()
