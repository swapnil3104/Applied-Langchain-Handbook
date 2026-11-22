"""
Module 04 - Memory and Persistence: 03_summarization_memory.py
Condensing long conversations with rolling summarization logic.
"""
from typing import List, Dict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

class SummarizationManager:
    """Maintains an executive summary of older messages while retaining recent turns."""

    def __init__(self, recent_turn_limit: int = 2):
        self.recent_turn_limit = recent_turn_limit
        self.summary: str = "No prior history."
        self.recent_messages: List[BaseMessage] = []

    def condense(self, new_messages: List[BaseMessage]):
        for msg in new_messages:
            self.recent_messages.append(msg)

        if len(self.recent_messages) > self.recent_turn_limit * 2:
            # Extract messages to summarize
            to_summarize = self.recent_messages[:-self.recent_turn_limit * 2]
            self.recent_messages = self.recent_messages[-self.recent_turn_limit * 2:]
            
            # Update summary
            extracted_points = [f"{m.type}: {m.content}" for m in to_summarize]
            self.summary = f"Summary of past context: {'; '.join(extracted_points)}"

    def get_context_for_prompt(self) -> List[BaseMessage]:
        return [
            SystemMessage(content=f"Context Summary: {self.summary}"),
            *self.recent_messages
        ]

def main():
    print("=== Rolling Conversation Summarization ===")
    manager = SummarizationManager(recent_turn_limit=2)

    turns = [
        HumanMessage(content="We are migrating our PostgreSQL cluster to Aurora."),
        AIMessage(content="Ensure you check compatibility flags and connection poolers."),
        HumanMessage(content="We use PgBouncer in transaction pooling mode."),
        AIMessage(content="Aurora handles transaction pooling cleanly with PgBouncer."),
        HumanMessage(content="What cutover window do you recommend?"),
        AIMessage(content="A 30-minute scheduled window during low-traffic hours.")
    ]

    manager.condense(turns)
    final_context = manager.get_context_for_prompt()

    print(f"Executive Summary:\n  {manager.summary}\n")
    print("Messages passed to next model turn:")
    for m in final_context:
        print(f"  [{m.type.upper()}]: {m.content}")

if __name__ == "__main__":
    main()
