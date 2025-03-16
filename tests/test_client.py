from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["../server/main.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print(prompts)

            # List available resources
            resources = await session.list_resources()
            print(resources)

            # List available tools
            tools = await session.list_tools()
            print(tools)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
