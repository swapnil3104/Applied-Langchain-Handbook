"""
Module 01 - Foundations: 02_messages_and_blocks.py
Demonstrates System, Human, and AI messages alongside multimodal `content_blocks`.
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def main():
    print("=== 1. Structured Conversational Sequence ===")
    conversation = [
        SystemMessage(content="You are an expert AI software architect."),
        HumanMessage(content="What is the difference between an LLM and a ChatModel?"),
        AIMessage(content="An LLM accepts and returns raw strings; a ChatModel accepts and returns structured, typed messages."),
        HumanMessage(content="Why does that matter for tool calling?")
    ]
    
    for msg in conversation:
        print(f"[{msg.type.upper()}]: {msg.content}")

    print("\n=== 2. Multimodal Content Blocks ===")
    # Modern ChatModels accept a list of dictionary blocks for text, images, and audio
    multimodal_msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze the following UI layout mockup and suggest improvements:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png",
                    "detail": "high"
                }
            }
        ]
    )

    print("Multimodal message constructed:")
    for idx, block in enumerate(multimodal_msg.content, 1):
        print(f"  Block {idx} - Type: {block.get('type')}")

if __name__ == "__main__":
    main()
