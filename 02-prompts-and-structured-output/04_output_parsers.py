"""
Module 02 - Prompts and Structured Output: 04_output_parsers.py
Comprehensive demonstration of LangChain Output Parsers:
- StrOutputParser: Extracts string content from AIMessage
- JsonOutputParser: Parses raw JSON strings into Python dictionaries
- PydanticOutputParser: Validates LLM output against Pydantic models with format instructions
- CommaSeparatedListOutputParser: Parses comma-delimited strings into Python lists
"""
import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    PydanticOutputParser,
)
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model

load_dotenv()


# 1. Pydantic Model for Schema-Enforced Parsing
class ActionItem(BaseModel):
    title: str = Field(description="Brief title of the action item")
    priority: str = Field(description="Priority level: HIGH, MEDIUM, or LOW")
    assignee: str = Field(description="Name or team responsible")
    estimated_hours: int = Field(description="Estimated hours to complete")


class ProjectBreakdown(BaseModel):
    project_name: str = Field(description="Name of the project")
    items: List[ActionItem] = Field(description="List of action items")


def demo_str_output_parser():
    print("\n--- 1. StrOutputParser ---")
    parser = StrOutputParser()
    
    # Simulating an AIMessage returned by a ChatModel
    sample_message = AIMessage(content="LangChain v0.3 standardizes LCEL pipelines across all providers.")
    result = parser.invoke(sample_message)
    print(f"Parsed type: {type(result).__name__}")
    print(f"Content: {result}")


def demo_comma_separated_list_parser():
    print("\n--- 2. CommaSeparatedListOutputParser ---")
    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()
    print(f"Format Instructions provided to LLM:\n  '{format_instructions}'")

    raw_llm_response = AIMessage(content="redis, postgresql, faiss, chromadb, qdrant")
    parsed_list = parser.invoke(raw_llm_response)
    print(f"Parsed List ({len(parsed_list)} items): {parsed_list}")
    print(f"Type check: {isinstance(parsed_list, list)} (First item: {parsed_list[0]})")


def demo_json_output_parser():
    print("\n--- 3. JsonOutputParser ---")
    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()
    print(f"Format Instructions:\n  {format_instructions[:90]}...")

    raw_json_message = AIMessage(
        content='```json\n{"service": "auth-api", "status": "healthy", "latency_ms": 24.5, "uptime": 99.98}\n```'
    )
    parsed_dict = parser.invoke(raw_json_message)
    print(f"Parsed type: {type(parsed_dict).__name__}")
    print(f"Parsed payload: {parsed_dict}")
    print(f"Accessing key 'latency_ms': {parsed_dict.get('latency_ms')} ms")


def demo_pydantic_output_parser():
    print("\n--- 4. PydanticOutputParser with Schema Enforcement ---")
    parser = PydanticOutputParser(pydantic_object=ProjectBreakdown)
    format_instructions = parser.get_format_instructions()
    print("Injected Format Instructions (snippet):")
    print("\n".join(f"  {line}" for line in format_instructions.splitlines()[:6]))

    prompt = PromptTemplate(
        template="Generate a structured implementation breakdown for the following initiative.\n{format_instructions}\nInitiative: {initiative}\n",
        input_variables=["initiative"],
        partial_variables={"format_instructions": format_instructions},
    )

    sample_initiative = "Deploy Distributed RAG Index with Hybrid Search"

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[Mock Execution] Simulating LLM generation matching Pydantic parser instructions:")
        mock_raw_output = AIMessage(
            content='''{
  "project_name": "Deploy Distributed RAG Index with Hybrid Search",
  "items": [
    {
      "title": "Benchmark BM25 vs Dense Index",
      "priority": "HIGH",
      "assignee": "Data Platform Team",
      "estimated_hours": 16
    },
    {
      "title": "Configure Cross-Encoder Reranker Endpoint",
      "priority": "MEDIUM",
      "assignee": "ML Engineering",
      "estimated_hours": 24
    }
  ]
}'''
        )
        parsed_project: ProjectBreakdown = parser.invoke(mock_raw_output)
        print(f"Successfully validated against Pydantic model: {type(parsed_project).__name__}")
        print(f"Project: {parsed_project.project_name}")
        for item in parsed_project.items:
            print(f"  * [{item.priority}] {item.title} ({item.assignee}, ~{item.estimated_hours}h)")
        return

    # Real LLM LCEL Chain execution
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
    chain = prompt | model | parser
    result = chain.invoke({"initiative": sample_initiative})
    print(f"Live Chain Output: {result}")


def main():
    print("=== LangChain Output Parsers Deep Dive ===")
    demo_str_output_parser()
    demo_comma_separated_list_parser()
    demo_json_output_parser()
    demo_pydantic_output_parser()


if __name__ == "__main__":
    main()
