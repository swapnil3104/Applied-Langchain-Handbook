"""
Module 03 - LCEL and Runnables: 01_sequential_chains.py
Composing prompts, models, and parsers using the pipe operator (|).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def simulate_llm(prompt_value) -> str:
    """Simulated LLM runnable step for consistent execution."""
    text = prompt_value.to_string()
    return f"[LLM Response] Transformed text: {text.strip()}"

def uppercase_parser(text: str) -> str:
    return text.upper()

def main():
    print("=== Sequential Chain Composition with Pipe (|) ===")
    prompt = ChatPromptTemplate.from_template("Summarize the significance of: {topic}")
    
    # Composing: Prompt -> Model Runnable -> StrOutputParser -> Custom Formatter
    chain = prompt | RunnableLambda(simulate_llm) | StrOutputParser() | RunnableLambda(uppercase_parser)

    result = chain.invoke({"topic": "LangChain Expression Language"})
    print(f"Final Pipeline Result:\n{result}")

if __name__ == "__main__":
    main()
