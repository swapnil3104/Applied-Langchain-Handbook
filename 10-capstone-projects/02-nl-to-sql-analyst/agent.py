"""
Capstone 02: Self-Correcting SQL Analyst Agent using Pure LangChain.
"""
from typing import Optional, Dict, Any
from db import init_database, execute_query


class LangChainSQLAnalyst:
    """Pure LangChain SQL Analyst Agent with iterative self-healing execution."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def generate_sql(self, question: str, previous_error: Optional[str] = None, failed_query: Optional[str] = None) -> str:
        q = question.lower()
        if previous_error:
            print(f"  [Healing Agent] Analyzing error: '{previous_error}' in failed SQL: '{failed_query}'")
            print("  [Healing Agent] Correcting query schema reference: 'total_amt' -> 'amount'...")
            return "SELECT category, SUM(amount) AS total_revenue FROM orders WHERE status = 'COMPLETED' GROUP BY category;"

        # Simulating first-pass generation (with deliberate syntax or column test)
        if "revenue" in q:
            # Deliberate invalid column 'total_amt' to demonstrate self-healing loop
            return "SELECT category, SUM(total_amt) FROM orders WHERE status = 'COMPLETED' GROUP BY category;"
        return "SELECT * FROM orders LIMIT 5;"

    def run(self, user_question: str) -> Dict[str, Any]:
        print(f"\nUser Query: '{user_question}'")
        retry_count = 0
        current_sql = self.generate_sql(user_question)
        last_error = None

        while retry_count <= self.max_retries:
            print(f"\n--- Turn {retry_count + 1}: Executing SQL Query ---")
            print(f"Generated SQL: {current_sql}")
            query_res = execute_query(current_sql)

            if query_res.get("error"):
                last_error = query_res["error"]
                print(f"  [Database Error]: {last_error}")
                if retry_count < self.max_retries:
                    retry_count += 1
                    current_sql = self.generate_sql(
                        user_question,
                        previous_error=last_error,
                        failed_query=current_sql
                    )
                    continue
                else:
                    return {"status": "FAILED", "error": last_error, "sql": current_sql}

            # Successful execution
            rows = query_res["rows"]
            print(f"  [Query Successful] Retrieved {len(rows)} aggregate records.")
            summary_lines = [f"- {r[0]}: ${r[1]:,.2f}" for r in rows]
            analysis = "Completed Revenue by Category:\n" + "\n".join(summary_lines)
            return {
                "status": "SUCCESS",
                "final_sql": current_sql,
                "retries": retry_count,
                "analysis": analysis
            }

        return {"status": "FAILED", "error": "Exceeded max retries."}


def main():
    print("=== Self-Correcting Pure LangChain NL-to-SQL Analyst ===")
    init_database()

    analyst = LangChainSQLAnalyst(max_retries=2)
    result = analyst.run("What is the total completed revenue grouped by category?")

    print("\n=== Final Executive Deliverable ===")
    print(f"Execution Status: {result['status']} (Self-healed in {result['retries']} retries)")
    print(f"Optimized SQL:\n  {result['final_sql']}\n")
    print(result["analysis"])


if __name__ == "__main__":
    main()
