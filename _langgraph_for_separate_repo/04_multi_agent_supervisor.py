"""
Module 08 - LangGraph Workflows: 04_multi_agent_supervisor.py
Coordinating specialized worker agents via a supervisor graph.
"""
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

class TeamState(TypedDict):
    task: str
    research_notes: str
    code_artifact: str
    next_worker: str

def supervisor_node(state: TeamState):
    print("  [Supervisor] Evaluating progress and dispatching work...")
    if not state.get("research_notes"):
        print("    -> Dispatching to Researcher")
        return {"next_worker": "researcher"}
    elif not state.get("code_artifact"):
        print("    -> Dispatching to Software Engineer")
        return {"next_worker": "coder"}
    print("    -> All deliverables satisfied. Finishing task.")
    return {"next_worker": "FINISH"}

def researcher_worker(state: TeamState):
    print("  [Researcher Worker] Synthesizing domain literature and architectural patterns...")
    return {
        "research_notes": "Identified high-performance indexing algorithm: HNSW cosine graph."
    }

def coder_worker(state: TeamState):
    print("  [Coder Worker] Implementing production Python code based on research notes...")
    return {
        "code_artifact": "def build_hnsw_index(embeddings): return FAISS.IndexHNSWFlat(embeddings)"
    }

def route_worker(state: TeamState) -> Literal["researcher", "coder", "__end__"]:
    target = state.get("next_worker")
    if target == "FINISH":
        return END
    return target

def main():
    print("=== Multi-Agent Supervisor Orchestration ===")

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_worker)
    builder.add_node("coder", coder_worker)

    builder.add_edge(START, "supervisor")
    
    # Dynamic routing edge from supervisor to workers
    builder.add_conditional_edges("supervisor", route_worker, {
        "researcher": "researcher",
        "coder": "coder",
        END: END
    })

    # Workers report back to supervisor upon completion
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")

    app = builder.compile()

    initial_job = {
        "task": "Build vector indexing module",
        "research_notes": "",
        "code_artifact": "",
        "next_worker": ""
    }

    final_result = app.invoke(initial_job)
    print(f"\nWorkflow Complete Deliverables:")
    print(f"  Research Notes: {final_result['research_notes']}")
    print(f"  Code Artifact:  {final_result['code_artifact']}")

if __name__ == "__main__":
    main()
