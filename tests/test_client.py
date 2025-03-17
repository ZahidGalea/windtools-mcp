import logging
import os
import sys

import mcp.types as types
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "windtools_mcp"],
    env={
        "DATA_ROOT": os.path.join(os.getcwd(), "windtools_data"),
        "CHROMA_DB_FOLDER_NAME": "chromadb",
        "SENTENCE_TRANSFORMER_PATH": "jinaai/jina-embeddings-v2-base-code"
    }
)


@pytest.mark.asyncio
async def test_tools_available():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools: types.ListToolsResult = await session.list_tools()
            logging.info("Tools available: %s", tools)

            assert tools
            for tool in tools.tools:
                if tool.name == "list_dir":
                    return True
            raise Exception