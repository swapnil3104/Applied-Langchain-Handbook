"""
Module 07 - Tools and Agents: 02_prebuilt_tools.py
Integrating simulated web search, Python REPL, and SQL database tools with safety controls.
"""
from langchain_core.tools import tool

@tool
def simulated_web_search(query: str) -> str:
    """Simulates web search over internet sources for real-time information."""
    print(f"  [Tool Action] Searching web for: '{query}'")
    return f"Search result: LangChain v0.3 released with unified ChatModel interfaces and stable LCEL runnables."

@tool
def sandboxed_python_calculator(expression: str) -> str:
    """Evaluates safe arithmetic expressions using Python syntax."""
    print(f"  [Tool Action] Evaluating math expression: '{expression}'")
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Expression contains forbidden characters for security safety."
    try:
        # Safe math evaluation
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

def main():
    print("=== Prebuilt Tool Sandbox Demonstration ===")
    tools = [simulated_web_search, sandboxed_python_calculator]

    for t in tools:
        print(f"Registered Tool: {t.name} -> {t.description}")

    print("\n1. Executing Safe Math Tool:")
    res_math = sandboxed_python_calculator.invoke({"expression": "(125 * 4) + 750 / 5"})
    print(f"Math Result: {res_math}")

    print("\n2. Executing Web Search Tool:")
    res_search = simulated_web_search.invoke({"query": "LangChain v1.0 release notes"})
    print(f"Search Result: {res_search}")

if __name__ == "__main__":
    main()
