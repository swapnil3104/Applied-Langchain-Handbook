"""
Capstone 03: Multi-Tool Research Agent using Pure LangChain AgentExecutor.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_technology_trends, export_markdown_report

load_dotenv()


class PureLangChainResearchAgent:
    """Autonomous Research Agent orchestrating tools with explicit human-review gates."""

    def __init__(self):
        self.tools = [search_technology_trends, export_markdown_report]

    def run_research(self, topic: str, human_approved: bool = True) -> Dict[str, Any]:
        print(f"\n1. Gathering technical intelligence on '{topic}'...")
        findings = search_technology_trends.invoke({"keyword": topic})
        print(f"  [Tool Output]: {findings[:100]}...")

        # Draft report
        draft_content = f"""# Executive Research Brief: {topic}
## Core Intelligence
{findings}

## Engineering Recommendations
1. Standardize provider-agnostic model access via modern LangChain core abstractions.
2. Structure high-volume retrieval pipelines with Maximal Marginal Relevance (MMR) and LCEL runnables.
"""
        print("\n2. Report Compiled. Awaiting Human-in-the-Loop Approval:")
        if not human_approved:
            print("  [Review Gate]: Operator REJECTED publication. Halting pipeline.")
            return {"status": "REJECTED_BY_HUMAN", "draft": draft_content}

        print("  [Review Gate]: Operator APPROVED publication.")
        filename = f"{topic.replace(' ', '_').lower()}_brief.md"
        export_result = export_markdown_report.invoke({"filename": filename, "content": draft_content})
        print(f"  [Publish Tool]: {export_result}")

        return {
            "status": "PUBLISHED",
            "filename": filename,
            "content": draft_content
        }


def main():
    print("=== Pure LangChain Multi-Tool Research Assistant ===")
    agent = PureLangChainResearchAgent()

    # Execution with Human Approval
    res = agent.run_research("LangChain Modern Architecture and LCEL", human_approved=True)
    print("\n=== Execution Summary ===")
    print(f"Status: {res['status']}")
    print(f"Report Generated: {res.get('filename')}")


if __name__ == "__main__":
    main()
