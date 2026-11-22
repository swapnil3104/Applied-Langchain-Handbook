"""
Module 01 - Foundations: 03_streaming_and_batching.py
Demonstrates .stream(), .astream(), and .batch() execution patterns.
"""
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import FakeListChatModel

load_dotenv()

def get_model():
    if not os.getenv("OPENAI_API_KEY"):
        return FakeListChatModel(responses=[
            "Token 1 | Token 2 | Token 3 | Token 4 | Token 5",
            "Batch Response A: High throughput batch execution.",
            "Batch Response B: Concurrent processing across requests."
        ])
    return init_chat_model("gpt-4o-mini", model_provider="openai")

async def run_async_stream(model):
    print("\n=== 3. Async Streaming (.astream()) ===")
    prompt = [HumanMessage(content="Write a 1-sentence description of async programming.")]
    async for chunk in model.astream(prompt):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
    print()

def main():
    model = get_model()

    print("=== 1. Synchronous Streaming (.stream()) ===")
    prompt = [HumanMessage(content="Count from 1 to 5 with short commentary.")]
    for chunk in model.stream(prompt):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
    print()

    print("\n=== 2. Batch Execution (.batch()) ===")
    batch_prompts = [
        [HumanMessage(content="Explain latency in 5 words.")],
        [HumanMessage(content="Explain throughput in 5 words.")]
    ]
    results = model.batch(batch_prompts)
    for idx, res in enumerate(results, 1):
        print(f"Batch {idx} Output: {res.content.strip()}")

    asyncio.run(run_async_stream(model))

if __name__ == "__main__":
    main()
