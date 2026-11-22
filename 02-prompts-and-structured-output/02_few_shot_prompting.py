"""
Module 02 - Prompts & Structured Output: 02_few_shot_prompting.py
Demonstrates guiding model formatting with dynamic few-shot examples.
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

def main():
    print("=== Dynamic Few-Shot Prompt Formatting ===")

    examples = [
        {"input": "The database connection timed out after 30 seconds.", "sentiment": "Negative", "category": "Infrastructure"},
        {"input": "Query latency dropped by 50% following index rebuilding.", "sentiment": "Positive", "category": "Performance"},
        {"input": "Scheduled maintenance will occur on Saturday at midnight.", "sentiment": "Neutral", "category": "Maintenance"}
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "Sentiment: {sentiment} | Category: {category}")
    ])

    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an operations log classifier. Follow the example format precisely."),
        few_shot_template,
        ("human", "{input}")
    ])

    output = final_prompt.format_messages(input="Deploy failed because container memory limit was exceeded.")
    for msg in output:
        print(f"[{msg.type.upper()}]: {msg.content}")

if __name__ == "__main__":
    main()
