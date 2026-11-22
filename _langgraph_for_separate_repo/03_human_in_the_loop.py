"""
Module 08 - LangGraph Workflows: 03_human_in_the_loop.py
Interrupting graph execution for human approval and state edits.
"""
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class DeploymentState(TypedDict):
    service: str
    target_cluster: str
    status: str
    approved: bool

def plan_node(state: DeploymentState):
    print(f"  [Plan] Prepared deployment plan for {state['service']} to {state['target_cluster']}")
    return {"status": "AWAITING_APPROVAL"}

def approval_checkpoint_node(state: DeploymentState):
    print(f"  [Gate] Checking human authorization...")
    if not state.get("approved", False):
        raise RuntimeError("Safety error: Attempted deployment without human sign-off!")
    return {"status": "APPROVED"}

def deploy_node(state: DeploymentState):
    print(f"  [Deploy] Executing deployment into {state['target_cluster']} cluster!")
    return {"status": "SUCCESS"}

def main():
    print("=== Human-in-the-Loop Approval Pattern ===")
    
    builder = StateGraph(DeploymentState)
    builder.add_node("plan", plan_node)
    builder.add_node("approval_gate", approval_checkpoint_node)
    builder.add_node("deploy", deploy_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approval_gate")
    builder.add_edge("approval_gate", "deploy")
    builder.add_edge("deploy", END)

    memory = InMemorySaver()
    # Interrupt execution right before approval_gate node runs
    app = builder.compile(checkpointer=memory, interrupt_before=["approval_gate"])

    config = {"configurable": {"thread_id": "deploy_session_001"}}

    print("\n1. Initial Trigger by Automated System:")
    app.invoke({"service": "payment-api", "target_cluster": "production-us-east-1", "status": "PENDING", "approved": False}, config=config)

    # Check paused state
    current_snapshot = app.get_state(config)
    print(f"\nExecution Interrupted! Next node scheduled to run: {current_snapshot.next}")
    print(f"Current State Values: {current_snapshot.values}")

    print("\n2. Simulating Human Reviewer Action: Approving deployment...")
    # Human operator edits state in memory
    app.update_state(config, {"approved": True})

    print("\n3. Resuming Workflow Execution:")
    final_output = app.invoke(None, config=config)
    print(f"Final State: {final_output['status']}")

if __name__ == "__main__":
    main()
