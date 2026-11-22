"""
Module 07 - Tools and Agents: 03_create_agent_loop.py
Building autonomous agents using pure LangChain:
`create_tool_calling_agent` and `AgentExecutor` with intermediate observation steps.
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chat_models import init_chat_model

load_dotenv()


@tool
def get_server_status(server_name: str) -> str:
    """Returns the operational status, CPU, and memory metrics for a server."""
    status_db = {
        "prod-db-1": "Status: ONLINE | CPU: 42% | Memory: 18.2GB/32GB | Uptime: 99.98%",
        "auth-api-3": "Status: DEGRADED | CPU: 96% | Memory: 7.9GB/8GB | High latency alerts active."
    }
    return status_db.get(server_name, f"Server '{server_name}' not found in registry.")


@tool
def restart_server(server_name: str) -> str:
    """Restarts a degraded server and resets CPU/memory buffers."""
    return f"Server '{server_name}' successfully restarted. Status: HEALTHY | CPU: 12% | Memory: 2.1GB/8GB."


def main():
    print("=== Pure LangChain Tool-Calling Agent Loop with AgentExecutor ===")
    tools = [get_server_status, restart_server]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an autonomous Site Reliability Engineer. Use available tools to diagnose and remedy system anomalies."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[Mock Mode] Demonstrating pure LangChain Agent ReAct reasoning cycle:")
        print("1. Human Input: 'Server auth-api-3 is alerting. Diagnose it and restart it if degraded.'")
        print("2. Agent Thought: Needs to call get_server_status for 'auth-api-3'.")
        print("   Action: get_server_status(server_name='auth-api-3')")
        diag = get_server_status.invoke({"server_name": "auth-api-3"})
        print(f"   Observation: {diag}")
        print("3. Agent Thought: Server is DEGRADED with 96% CPU. Initiating restart action.")
        print("   Action: restart_server(server_name='auth-api-3')")
        remedy = restart_server.invoke({"server_name": "auth-api-3"})
        print(f"   Observation: {remedy}")
        print("4. Agent Final Response: 'auth-api-3 was found to be DEGRADED due to 96% CPU usage. I executed a graceful restart; it is now HEALTHY with 12% CPU.'")
        return

    # Real LLM Agent Execution using LangChain's AgentExecutor
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    query = "Check the health of auth-api-3. If it is degraded, restart it."
    print(f"\nSubmitting Query to AgentExecutor: '{query}'\n")
    result = agent_executor.invoke({"input": query})
    print(f"\nFinal Agent Response:\n{result['output']}")


if __name__ == "__main__":
    main()
