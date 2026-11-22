"""
Module 10 - Observability and Evals: 03_agent_trajectory_evals.py
Evaluating tool-call sequences and final answers.
"""
from typing import List, Dict

class TrajectoryEvaluator:
    """Evaluates whether an agent followed an expected sequence of tool calls."""

    @staticmethod
    def evaluate(trajectory: List[Dict], expected_tools: List[str]) -> Dict:
        actual_tools = [step.get("tool") for step in trajectory if "tool" in step]
        
        # Check order and completeness
        all_expected_called = all(tool in actual_tools for tool in expected_tools)
        exact_order_match = actual_tools == expected_tools
        extra_tools = [t for t in actual_tools if t not in expected_tools]

        score = 1.0 if exact_order_match else (0.5 if all_expected_called else 0.0)

        return {
            "score": score,
            "actual_tool_sequence": actual_tools,
            "expected_tool_sequence": expected_tools,
            "all_expected_called": all_expected_called,
            "extra_tools_invoked": extra_tools
        }

def main():
    print("=== Agent Trajectory & Intermediate Step Evaluation ===")

    # Simulated agent trajectory during a customer refund request
    agent_recorded_trajectory = [
        {"step": 1, "tool": "lookup_user_account", "args": {"user_id": "usr_992"}},
        {"step": 2, "tool": "check_purchase_eligibility", "args": {"order_id": "ord_108"}},
        {"step": 3, "tool": "issue_stripe_refund", "args": {"amount": 49.99}}
    ]

    expected_policy = [
        "lookup_user_account",
        "check_purchase_eligibility",
        "issue_stripe_refund"
    ]

    print("Evaluating Agent Execution Trajectory against Compliance Policy:")
    report = TrajectoryEvaluator.evaluate(agent_recorded_trajectory, expected_policy)

    for k, v in report.items():
        print(f"  {k}: {v}")

    if report["score"] == 1.0:
        print("\n[PASSED] Agent trajectory perfectly compliant with enterprise refund policy.")
    else:
        print("\n[FAILED] Agent deviated from required tool execution path.")

if __name__ == "__main__":
    main()
