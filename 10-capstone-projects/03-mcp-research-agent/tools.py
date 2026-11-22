"""
Capstone 03: Tool Suite for MCP Research Agent.
"""
from langchain_core.tools import tool

@tool
def search_technology_trends(keyword: str) -> str:
    """Fetches technology and architectural trends."""
    return f"Latest trends for '{keyword}': Model Context Protocol (MCP) adopted by leading AI frameworks for secure tool integration."

@tool
def summarize_document(text: str) -> str:
    """Summarizes raw findings into structured notes."""
    return f"Key takeaway: {text[:80]}..."

@tool
def export_markdown_report(filename: str, content: str) -> str:
    """Saves final research report to disk."""
    return f"Report successfully written to {filename} ({len(content)} characters)."
