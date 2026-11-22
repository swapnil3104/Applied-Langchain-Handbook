"""
Module 03 - LCEL and Runnables: 03_routing_and_fallbacks.py
Conditional branching and `.with_fallbacks()` for reliability.
"""
from langchain_core.runnables import RunnableLambda

def math_branch(query: str) -> str:
    return f"[Math Engine] Calculating mathematical solution for: {query}"

def code_branch(query: str) -> str:
    return f"[Code Engine] Generating software patch for: {query}"

def general_branch(query: str) -> str:
    return f"[General Engine] Processing conversational query: {query}"

def route_query(input_dict: dict):
    q = input_dict["query"].lower()
    if any(k in q for k in ["calculate", "math", "sum", "integral"]):
        return RunnableLambda(lambda x: math_branch(x["query"]))
    elif any(k in q for k in ["code", "function", "bug", "python"]):
        return RunnableLambda(lambda x: code_branch(x["query"]))
    return RunnableLambda(lambda x: general_branch(x["query"]))

def unstable_api(x):
    raise TimeoutError("504 Gateway Timeout: Primary API unreachable")

def fallback_api(x):
    return f"[Fallback Success] Secondary model handled request: '{x['query']}'"

def main():
    print("=== 1. Dynamic Routing Engine ===")
    router = RunnableLambda(route_query)

    print(router.invoke({"query": "Write a python function to parse JSON."}))
    print(router.invoke({"query": "Calculate the sum of primes below 50."}))
    print(router.invoke({"query": "Tell me a fun fact about Mars."}))

    print("\n=== 2. Resilient Execution with .with_fallbacks() ===")
    primary = RunnableLambda(unstable_api)
    fallback = RunnableLambda(fallback_api)

    resilient_chain = primary.with_fallbacks([fallback])
    result = resilient_chain.invoke({"query": "Process financial ledger update"})
    print(f"Resilient Result: {result}")

if __name__ == "__main__":
    main()
