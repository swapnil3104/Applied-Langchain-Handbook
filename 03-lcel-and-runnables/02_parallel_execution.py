"""
Module 03 - LCEL and Runnables: 02_parallel_execution.py
Concurrent execution using `RunnableParallel` and `RunnablePassthrough`.
"""
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

def analyze_sentiment(data: dict) -> str:
    text = data["content"].lower()
    return "Positive" if "excellent" in text or "fast" in text else "Neutral"

def extract_topics(data: dict) -> list:
    tokens = [w.strip(".,") for w in data["content"].split() if len(w) > 4]
    return list(set(tokens))[:4]

def calculate_stats(data: dict) -> dict:
    text = data["content"]
    return {"word_count": len(text.split()), "char_count": len(text)}

def main():
    print("=== Parallel Execution with RunnableParallel ===")

    pipeline = RunnableParallel({
        "input_passthrough": RunnablePassthrough(),
        "sentiment": RunnableLambda(analyze_sentiment),
        "topics": RunnableLambda(extract_topics),
        "statistics": RunnableLambda(calculate_stats)
    })

    sample_doc = {"content": "LangChain v1.0 delivers excellent performance and fast developer iteration."}
    output = pipeline.invoke(sample_doc)

    print("Parallel Pipeline Output:")
    for k, v in output.items():
        print(f"  [{k}]: {v}")

if __name__ == "__main__":
    main()
