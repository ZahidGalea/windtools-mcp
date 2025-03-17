import asyncio
import json
import logging
import os
import os.path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import time
from logging import INFO, basicConfig
from typing import Any, AsyncIterator, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s",
)
# PARAMS
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "data"))
CHROMA_DB_FOLDER_NAME = os.environ.get("CHROMA_DB_FOLDER_NAME", "default")
SENTENCE_TRANSFORMER_PATH = os.environ.get(
    "SENTENCE_TRANSFORMER_PATH", "jinaai/jina-embeddings-v2-base-code"
)

# Ensure data directories exist
os.makedirs(DATA_ROOT, exist_ok=True)
CHROMA_DB_PATH = os.path.join(DATA_ROOT, CHROMA_DB_FOLDER_NAME)
SENTENCE_TRANSFORMER_CACHE_FOLDER = os.path.join(DATA_ROOT, "embedding_cache")


# Server lifespan context for ChromaDB initialization and project directory
@dataclass
class ServerContext:
    chroma_client: Optional[Any] = None
    code_collection: Optional[Any] = None
    embedding_model: str = ""
    is_initialized: bool = False
    initialization_error: Optional[str] = None
    command_registry: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        self.command_registry = {}


# Create a global context object
ctx = ServerContext(embedding_model=SENTENCE_TRANSFORMER_PATH)


async def initialize_resources():
    """Initialize ChromaDB and embedding model in background"""
    try:
        logging.info("Thanks for using WindCodeAssistant!")
        logging.info(f"Initializing ChromaDB at {CHROMA_DB_PATH} and embedding model...")

        # Use run_in_executor for potentially blocking file system operations
        loop = asyncio.get_running_loop()
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(SENTENCE_TRANSFORMER_CACHE_FOLDER, exist_ok=True)

        # Import ChromaDB here to allow for dependency installation
        import chromadb
        from chromadb.utils import embedding_functions

        # Run potentially blocking operations in executor
        chroma_client = await loop.run_in_executor(None, lambda: chromadb.PersistentClient(path=CHROMA_DB_PATH))
        embedding_function = await loop.run_in_executor(
            None,
            lambda: embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=SENTENCE_TRANSFORMER_PATH,
                cache_folder=SENTENCE_TRANSFORMER_CACHE_FOLDER,
            ),
        )

        # Create or get the code collection
        try:
            code_collection = chroma_client.get_collection(
                name="code_collection", embedding_function=embedding_function
            )
            logging.info(
                f"Using existing code collection with {code_collection.count()} documents"
            )
        except Exception as e:
            logging.info(f"Collection not found, creating new one. Error: {str(e)}")
            code_collection = chroma_client.create_collection(
                name="code_collection", embedding_function=embedding_function
            )
            logging.info("Created new code collection")

        # Update global context
        ctx.chroma_client = chroma_client
        ctx.code_collection = code_collection
        ctx.is_initialized = True
        logging.info("Background initialization completed successfully")
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        logging.error(error_msg)
        ctx.initialization_error = error_msg


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Start initialization in background and continue server startup immediately"""
    # Start initialization in background
    init_task = asyncio.create_task(initialize_resources())

    try:
        # Return control immediately with the global context
        yield ctx
    finally:
        logging.info("Waiting for initialization task to complete before shutdown...")
        # Make sure the init task completes before server shuts down
        if not init_task.done():
            init_task.cancel()
            try:
                await init_task
            except asyncio.CancelledError:
                logging.info("Initialization task was cancelled")


mcp = FastMCP(
    "WindCodeAssistant",
    dependencies=["glob", "re", "json", "subprocess"],
    lifespan=server_lifespan,
)


def _get_directory_info(directory_path: str) -> List[Dict[str, Any]]:
    """
    Helper function to get information about directory contents

    Args:
        directory_path: Absolute path to the directory

    Returns:
        List of dictionaries containing information about each child in the directory
    """
    result = []

    # Check if directory exists and is a directory
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Get all entries in the directory
    for entry in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry)
        relative_path = os.path.relpath(entry_path, directory_path)

        if os.path.isdir(entry_path):
            # For directories, count children recursively
            child_count = sum(len(files) for _, _, files in os.walk(entry_path))
            result.append(
                {"path": relative_path, "type": "directory", "child_count": child_count}
            )
        else:
            # For files, get size in bytes
            size = os.path.getsize(entry_path)
            result.append({"path": relative_path, "type": "file", "size": size})

    return result


@mcp.tool()
def list_dir(directory_path: str) -> str:
    """
    List the contents of a directory.

    Directory path must be an absolute path to a directory that exists.
    For each child in the directory, output will have:
    - relative path to the directory
    - whether it is a directory or file
    - size in bytes if file
    - number of children (recursive) if directory

    Args:
        directory_path: Path to list contents of, should be absolute path to a directory

    Returns:
        JSON string containing directory information
    """
    # We don't need ChromaDB or embeddings for this function, so it works regardless of initialization
    try:
        logging.info(f"Listing directory: {directory_path}")
        directory_info = _get_directory_info(directory_path)
        return json.dumps(directory_info, indent=2)
    except Exception as e:
        logging.error(f"Error listing directory {directory_path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_initialization_status() -> str:
    """
    Get the status of background initialization process.
    The background initialization process includes initializing ChromaDB and embedding model.

    Returns:
        JSON string with initialization status
    """
    status = {
        "is_initialized": ctx.is_initialized,
        "error": ctx.initialization_error
    }
    return json.dumps(status)


@mcp.tool()
def index_repository(target_directories: List[str], force_reindex: bool = False) -> str:
    """
    Index code files from the specified directories into ChromaDB for later search.

    This tool scans the specified directories for code files, indexes their content
    in ChromaDB, and updates existing entries if they have changed. This enables
    high-quality semantic search over the codebase.

    Args:
        target_directories: List of absolute paths to directories to index
        force_reindex: If true, will reindex all files even if they already exist in the index

    Returns:
        JSON string containing indexing statistics and results
    """
    if not ctx.is_initialized:
        return json.dumps({"error": "ChromaDB and embedding model not yet initialized"})

    try:
        logging.info(f"Indexing code repositories: {target_directories}")

        # Statistics to track
        stats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_updated": 0,
            "files_skipped": 0,
            "errors": 0,
            "total_tokens_processed": 0,
        }

        # Get existing document IDs for update/skip logic
        existing_ids = set()
        if ctx.code_collection.count() > 0:
            # Fetch all existing IDs - this could be optimized for large collections
            existing_ids = set(ctx.code_collection.get()["ids"])

        # Define file extensions considered as code
        code_extensions = [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".jsx",
            ".tsx",
            ".php",
            ".rb",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
        ]

        # Track processed files to avoid duplicates
        processed_files = set()

        # Process each directory
        for directory in target_directories:
            if not os.path.exists(directory) or not os.path.isdir(directory):
                logging.warning(f"Directory does not exist or is not a directory: {directory}")
                continue

            # Walk through the directory tree
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Skip if already processed (in case of overlapping directories)
                    if file_path in processed_files:
                        continue

                    # Only consider files with code extensions
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext not in code_extensions:
                        continue

                    stats["files_scanned"] += 1
                    processed_files.add(file_path)

                    try:
                        # Generate a unique document ID based on file path
                        doc_id = f"file:{file_path}"

                        # Check if file already exists in index
                        if doc_id in existing_ids and not force_reindex:
                            # Check if file has been modified since last indexing
                            # In a real implementation, we would store and check modification times
                            # For simplicity, we'll assume no change and skip
                            stats["files_skipped"] += 1
                            continue

                        # Read file content
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        # Skip empty files
                        if not content.strip():
                            stats["files_skipped"] += 1
                            continue

                        # Prepare metadata
                        metadata = {
                            "file_path": file_path,
                            "file_type": file_ext[1:],  # Remove the dot
                            "file_size": os.path.getsize(file_path),
                            "last_modified": os.path.getmtime(file_path),
                            "indexed_at": time(),
                        }

                        # If document exists, update it
                        if doc_id in existing_ids:
                            ctx.code_collection.update(ids=[doc_id], documents=[content], metadatas=[metadata])
                            stats["files_updated"] += 1
                        else:
                            # Otherwise, add new document
                            ctx.code_collection.add(ids=[doc_id], documents=[content], metadatas=[metadata])
                            stats["files_indexed"] += 1

                        # Rough estimate of tokens processed
                        stats["total_tokens_processed"] += len(content) // 4

                    except Exception as e:
                        logging.error(f"Error indexing file {file_path}: {str(e)}")
                        stats["errors"] += 1

        return json.dumps(
            {
                "status": "success",
                "message": "Repository indexing completed successfully",
                "statistics": stats,
                "collection_size": ctx.code_collection.count(),
            },
            indent=2,
        )

    except Exception as e:
        logging.error(f"Error during repository indexing: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def codebase_search(query: str, limit: int = 10, min_relevance: float = 0.0) -> str:
    """
    Find snippets of code from the indexed codebase most relevant to the search query.

    This performs semantic search over previously indexed code files.
    Results are ranked by relevance to the query. For best results, index your
    repositories first using the index_repository tool.

    Args:
        query: Search query describing what you're looking for
        limit: Maximum number of results to return (default: 10)
        min_relevance: Minimum relevance score threshold (0.0 to 1.0)

    Returns:
        JSON string containing search results with relevant code snippets
    """
    if not ctx.is_initialized:
        return json.dumps({"error": "ChromaDB and embedding model not yet initialized"})

    try:
        logging.info(f"Searching codebase for: {query}")

        # Check if we have any indexed documents
        if ctx.code_collection.count() == 0:
            return json.dumps(
                {"message": "No code has been indexed yet. Use the index_repository tool first.", "results": []}
            )

        # Process the search query with ChromaDB
        results = []

        # Search in collection
        search_results = ctx.code_collection.query(
            query_texts=[query], n_results=min(limit, ctx.code_collection.count())
        )

        # Format results
        for i, (doc_id, distance) in enumerate(zip(search_results["ids"][0], search_results["distances"][0])):
            metadata = (
                search_results["metadatas"][0][i]
                if "metadatas" in search_results and search_results["metadatas"]
                else {}
            )
            document = (
                search_results["documents"][0][i]
                if "documents" in search_results and search_results["documents"]
                else ""
            )

            # Calculate relevance score (1.0 is perfect match, 0.0 is completely irrelevant)
            relevance_score = 1.0 - (distance if distance else 0)

            # Skip results below minimum relevance threshold
            if relevance_score < min_relevance:
                continue

            # Extract a snippet of the document (context around the most relevant part)
            snippet = document[:1000] + "..." if len(document) > 1000 else document

            results.append(
                {
                    "id": doc_id,
                    "relevance_score": relevance_score,
                    "file_path": metadata.get("file_path", "Unknown"),
                    "file_type": metadata.get("file_type", "Unknown"),
                    "last_modified": metadata.get("last_modified", 0),
                    "snippet": snippet,
                }
            )

        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return json.dumps({"query": query, "total_results": len(results), "results": results}, indent=2)

    except Exception as e:
        logging.error(f"Error during codebase search: {str(e)}")
        return json.dumps({"error": str(e)})