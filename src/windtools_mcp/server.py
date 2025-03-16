import logging
import os

# Add lifespan support for startup/shutdown with strong typing
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import basicConfig, INFO
from typing import Any, AsyncIterator

from mcp.server.fastmcp import Context, FastMCP

basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s",
)
# PARAMS
CHROMA_DB_FOLDER_NAME = os.environ.get("CHROMA_DB_FOLDER_NAME", "default")
SENTENCE_TRANSFORMER_PATH = os.environ.get(
    "SENTENCE_TRANSFORMER_PATH", "Linq-AI-Research/Linq-Embed-Mistral"
)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_PATH = os.path.join(DATA_FOLDER, CHROMA_DB_FOLDER_NAME)
SENTENCE_TRANSFORMER_CACHE_FOLDER = os.path.join(DATA_FOLDER, "embedding_cache")


# Server lifespan context for ChromaDB initialization
@dataclass
class ServerContext:
    chroma_client: Any
    code_collection: Any
    embedding_model: str


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Initialize and clean up resources during server lifecycle"""
    logging.info("Initializing ChromaDB and embedding model...")

    # Import ChromaDB here to allow for dependency installation
    import chromadb
    from chromadb.utils import embedding_functions

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=SENTENCE_TRANSFORMER_PATH,
        cache_folder=SENTENCE_TRANSFORMER_CACHE_FOLDER,
    )

    # Create or get the code collection
    try:
        code_collection = chroma_client.get_collection(
            name="code_collection", embedding_function=embedding_function
        )
        logging.info(
            f"Using existing code collection with {code_collection.count()} documents"
        )
    except Exception as _:
        code_collection = chroma_client.create_collection(
            name="code_collection", embedding_function=embedding_function
        )
        logging.info("Created new code collection")

    # Create the context
    ctx = ServerContext(
        chroma_client=chroma_client,
        code_collection=code_collection,
        embedding_model="Linq-AI-Research/Linq-Embed-Mistral",
    )

    try:
        yield ctx
    finally:
        logging.info("Cleaning up ChromaDB resources...")
        # ChromaDB client will be closed automatically when the process ends


mcp = FastMCP(
    "WindCodeAssistant",
    dependencies=["glob", "re", "json", "subprocess"],
    lifespan=server_lifespan,
)


@mcp.tool()
def list_dir(ctx: Context, directory_path: str) -> str:
    """
    List the contents of a directory.

    Args:
        directory_path: Path to list contents of, should be absolute path to a directory

    Returns:
        String with directory listing
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return f"Directory not found or not a directory: {directory_path}"

    results = []

    try:
        for item in sorted(os.listdir(directory_path)):
            item_path = os.path.join(directory_path, item)

            if os.path.isdir(item_path):
                # Count number of items in directory
                item_count = len(os.listdir(item_path))
                results.append(f"{item}\tdir\t{item_count} children")
            else:
                # Get file size
                size = os.path.getsize(item_path)
                results.append(f"{item}\tfile\t{size} bytes")

        if not results:
            return f"Directory {directory_path} is empty"

        return "\n".join(results)

    except Exception as e:
        return f"Error listing directory: {str(e)}"
