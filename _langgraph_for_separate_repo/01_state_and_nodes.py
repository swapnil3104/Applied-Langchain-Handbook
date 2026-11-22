"""
Module 08 - LangGraph Workflows: 01_state_and_nodes.py
Defining `TypedDict` state, reducers, and deterministic graph nodes.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    validation_status: str
    execution_step: int

def ingest_node(state: WorkflowState):
    print("  -> Executing Ingest Node")
    return {
        "execution_step": state.get("execution_step", 0) + 1,
        "validation_status": "PENDING"
    }

def validate_node(state: WorkflowState):
    print("  -> Executing Validate Node")
    # Verify input text
    last_msg = state["messages"][-1].content
    status = "VALID" if len(last_msg) > 5 else "INVALID"
    return {
        "execution_step": state["execution_step"] + 1,
        "validation_status": status
    }

def finalize_node(state: WorkflowState):
    print("  -> Executing Finalize Node")
    reply = f"Pipeline completed. Validation: {state['validation_status']}, Steps taken: {state['execution_step']}."
    return {
        "messages": [AIMessage(content=reply)]
    }

def main():
    print("=== LangGraph State, Reducers, and Node Pipeline ===")

    graph = StateGraph(WorkflowState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("finalize", END)

    app = graph.compile()

    initial_input = {
        "messages": [HumanMessage(content="Process order #98214")],
        "execution_step": 0,
        "validation_status": "NONE"
    }

    result = app.invoke(initial_input)
    print(f"\nFinal Graph State:")
    print(f"  Validation Status: {result['validation_status']}")
    print(f"  Execution Steps: {result['execution_step']}")
    print(f"  Final Message: {result['messages'][-1].content}")

if __name__ == "__main__":
    main()
