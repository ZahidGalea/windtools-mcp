import logging
import os
import sys

import mcp.types as types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for tests
server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "windtools_mcp"],
    env={
        "DATA_ROOT": os.path.join(os.getcwd(), "windtools_data"),
        "CHROMA_DB_FOLDER_NAME": "chromadb_test",
        "SENTENCE_TRANSFORMER_PATH": "jinaai/jina-embeddings-v2-base-code"
    }
)


@pytest.mark.asyncio
async def test_tools_available():
    """Test that we can connect to the server and list available tools"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools: types.ListToolsResult = await session.list_tools()
            logging.info("Tools available: %s", tools)

            assert tools
            
            # Verify key tools are available
            tool_names = [tool.name for tool in tools.tools]
            assert "list_dir" in tool_names, "list_dir tool should be available"
            assert "get_initialization_status" in tool_names, "get_initialization_status tool should be available"
            assert "codebase_search" in tool_names, "codebase_search tool should be available"
            assert "index_repository" in tool_names, "index_repository tool should be available"