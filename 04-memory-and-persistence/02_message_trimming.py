"""
Module 04 - Memory and Persistence: 02_message_trimming.py
Managing token limits using `trim_messages`.
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages

def mock_token_counter(messages: list) -> int:
    """Calculates token estimation based on character length."""
    total_words = sum(len(msg.content.split()) for msg in messages)
    return total_words

def main():
    print("=== Token Limit Management with trim_messages ===")

    messages = [
        SystemMessage(content="You are a secure database auditor."),
        HumanMessage(content="User 1 executed SELECT * FROM users at 10:00 AM."),
        AIMessage(content="Logged event for user 1."),
        HumanMessage(content="User 2 executed UPDATE balances SET amount = 500 at 10:05 AM."),
        AIMessage(content="Logged event for user 2."),
        HumanMessage(content="User 3 attempted DROP TABLE transactions at 10:10 AM."),
        AIMessage(content="ALERT: Prevented drop table attempt for user 3."),
        HumanMessage(content="What was the latest critical security event?")
    ]

    print(f"Original message count: {len(messages)}")

    # Trim to last 25 tokens while preserving the SystemMessage
    trimmed = trim_messages(
        messages,
        max_tokens=25,
        strategy="last",
        token_counter=mock_token_counter,
        include_system=True,
        start_on="human"
    )

    print(f"Trimmed message count: {len(trimmed)}")
    print("\nRetained Messages in Context Window:")
    for msg in trimmed:
        print(f"[{msg.type.upper()}]: {msg.content}")

if __name__ == "__main__":
    main()
