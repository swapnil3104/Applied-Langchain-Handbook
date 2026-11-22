"""
Module 08 - LangGraph Workflows: 02_conditional_edges.py
Building cyclic loops and dynamic routing decisions.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class CodeReviewState(TypedDict):
    code: str
    iteration: int
    quality_score: int
    feedback: str

def generate_code_node(state: CodeReviewState):
    iteration = state["iteration"] + 1
    # Simulate improving code across iterations
    score = min(100, 50 + (iteration * 20))
    print(f"  [Coder] Iteration {iteration}: Generated code snippet (Quality Score: {score}/100)")
    return {
        "iteration": iteration,
        "quality_score": score,
        "feedback": "Needs optimization" if score < 90 else "Ready for production"
    }

def reviewer_node(state: CodeReviewState):
    print(f"  [Reviewer] Inspecting code quality: {state['quality_score']}/100 - Feedback: {state['feedback']}")
    return state

def should_continue(state: CodeReviewState) -> str:
    """Conditional router function: routes back to coder if quality is insufficient."""
    if state["quality_score"] >= 90:
        print("  -> Quality threshold reached! Exiting loop.")
        return "approved"
    elif state["iteration"] >= 4:
        print("  -> Maximum iterations reached. Escalating.")
        return "escalated"
    print("  -> Quality below threshold. Cycling back to coder.")
    return "rework"

def main():
    print("=== Cyclic LangGraph Workflow with Conditional Routing ===")

    builder = StateGraph(CodeReviewState)
    builder.add_node("coder", generate_code_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "coder")
    builder.add_edge("coder", "reviewer")

    # Add dynamic conditional routing edge
    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "approved": END,
            "escalated": END,
            "rework": "coder"
        }
    )

    app = builder.compile()

    final_state = app.invoke({
        "code": "def process(): pass",
        "iteration": 0,
        "quality_score": 0,
        "feedback": "Initial draft"
    })

    print(f"\nWorkflow Complete after {final_state['iteration']} iterations. Final Score: {final_state['quality_score']}")

if __name__ == "__main__":
    main()
