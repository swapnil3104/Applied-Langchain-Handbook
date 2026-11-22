"""
Module 10 - Observability and Evals: 02_unit_testing_llms.py
Deterministic testing using `GenericFakeChatModel`.
"""
from langchain_community.chat_models.generic import GenericFakeChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_summarizer_pipeline(chat_model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the text in 5 words or less."),
        ("human", "{text}")
    ])
    return prompt | chat_model | StrOutputParser()

def test_summarizer_deterministic_behavior():
    print("=== Running Deterministic LLM Unit Test ===")
    
    # 1. Setup deterministic mock model with pre-programmed responses
    fake_model = GenericFakeChatModel(
        messages=iter([
            AIMessage(content="Deterministic fast summary output.")
        ])
    )

    # 2. Build pipeline with mock model
    pipeline = create_summarizer_pipeline(fake_model)

    # 3. Assert exact pipeline output
    input_text = "LangChain provides comprehensive abstractions for deterministic agent evaluation."
    result = pipeline.invoke({"text": input_text})

    print(f"Input: '{input_text}'")
    print(f"Pipeline Result: '{result}'")

    assert result == "Deterministic fast summary output.", f"Unexpected result: {result}"
    print("\n[SUCCESS] Unit Test Passed: Output matches deterministic specification perfectly!")

if __name__ == "__main__":
    test_summarizer_deterministic_behavior()
