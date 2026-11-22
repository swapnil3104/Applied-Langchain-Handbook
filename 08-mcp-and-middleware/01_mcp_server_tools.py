"""
Module 09 - MCP and Middleware: 01_mcp_server_tools.py
Connecting external MCP servers via tool adapters.
"""
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Schema representing an external MCP tool contract
class ReadFileArgs(BaseModel):
    filepath: str = Field(description="Absolute or relative path to file on MCP host")

class ListDirectoryArgs(BaseModel):
    directory: str = Field(description="Directory path to inspect")

class MCPToolBridge:
    """Simulates client adapter bridge for an external Model Context Protocol (MCP) server."""
    
    @staticmethod
    def read_file(filepath: str) -> str:
        # Simulated secure MCP filesystem call
        return f"[MCP Filesystem Response] Contents of {filepath}: 'DATABASE_URL=postgres://localhost:5432'"

    @staticmethod
    def list_dir(directory: str) -> list:
        return ["pyproject.toml", "README.md", "src/"]

    @classmethod
    def get_langchain_tools(cls):
        """Converts MCP tool definitions to standard LangChain StructuredTools."""
        tool_read = StructuredTool.from_function(
            func=cls.read_file,
            name="mcp_fs_read_file",
            description="Reads file contents via remote MCP Filesystem server",
            args_schema=ReadFileArgs
        )
        tool_list = StructuredTool.from_function(
            func=cls.list_dir,
            name="mcp_fs_list_directory",
            description="Lists directories via remote MCP Filesystem server",
            args_schema=ListDirectoryArgs
        )
        return [tool_read, tool_list]

def main():
    print("=== Consuming External MCP Server Tools via LangChain Adapters ===")
    
    mcp_tools = MCPToolBridge.get_langchain_tools()
    for t in mcp_tools:
        print(f"Adapted MCP Tool: '{t.name}' | Args: {list(t.args.keys())}")

    print("\nExecuting adapted MCP tool:")
    sample_read = mcp_tools[0].invoke({"filepath": "/etc/secrets/config.env"})
    print(sample_read)

if __name__ == "__main__":
    main()
