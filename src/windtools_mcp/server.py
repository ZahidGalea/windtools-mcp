import asyncio
import glob
import json
import logging
import os
import os.path
import re
import subprocess
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


@mcp.tool()
def grep_search(search_directory: str, query: str, match_per_line: bool,
                includes: List[str], case_insensitive: bool) -> str:
    """
    Fast text-based search that finds exact pattern matches within files or directories.

    Utilizes the ripgrep command for efficient searching. Results will be formatted
    in the style of ripgrep and can be configured to include line numbers and content.
    To avoid overwhelming output, the results are capped at 50 matches. Use the
    Includes option to filter the search scope by file types or specific paths to
    narrow down the results.

    Args:
        search_directory: The directory from which to run the ripgrep command. This path must be a directory not a file.
        query: The search term or pattern to look for within files.
        match_per_line: If true, returns each line that matches the query, including line numbers and snippets.
                        If false, only returns the names of files containing the query.
        includes: The files or directories to search within. Supports file patterns (e.g., '*.txt' for all .txt files)
                 or specific paths.
        case_insensitive: If true, performs a case-insensitive search.

    Returns:
        String containing search results
    """
    try:
        logging.info(f"Grep search in {search_directory} for '{query}'")

        if not os.path.exists(search_directory) or not os.path.isdir(search_directory):
            return json.dumps({"error": f"Search directory does not exist or is not a directory: {search_directory}"})

        # Prepare command arguments
        cmd_args = ["rg", "--no-heading"]

        if case_insensitive:
            cmd_args.append("-i")

        if not match_per_line:
            cmd_args.append("-l")  # List files only
        else:
            cmd_args.append("-n")  # Show line numbers

        # Add file patterns to include
        for pattern in includes:
            cmd_args.extend(["-g", pattern])

        # Add the query and search directory
        cmd_args.extend([query, search_directory])

        # Execute the command
        process = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=30  # Timeout after 30 seconds
        )

        # Process the results
        if process.returncode not in [0, 1]:  # rg returns 1 when no matches found
            if process.stderr:
                return json.dumps({"error": process.stderr})
            return json.dumps({"error": f"Command failed with return code {process.returncode}"})

        # Limit the number of results to avoid overwhelming output
        lines = process.stdout.splitlines()
        if len(lines) > 50:
            result_str = "\n".join(lines[:50])
            result_str += f"\n... and {len(lines) - 50} more matches (output truncated)"
        else:
            result_str = process.stdout

        return result_str if result_str else "No matches found."
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Search timed out after 30 seconds"})
    except Exception as e:
        logging.error(f"Error during grep search: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def find_by_name(search_directory: str, pattern: str, includes: Optional[List[str]] = None,
                 excludes: Optional[List[str]] = None, max_depth: Optional[int] = None,
                 type_filter: Optional[str] = None) -> str:
    """
    Searches for files and directories within a specified directory, similar to the Linux `find` command.

    Supports glob patterns for searching and filtering which will all be passed in with -ipath.
    The patterns provided should match the relative paths from the search directory.
    They should use glob patterns with wildcards, for example, `**/*.py`, `**/*_test*`.

    Args:
        search_directory: The directory to search within
        pattern: Pattern to search for
        includes: Optional patterns to include
        excludes: Optional patterns to exclude
        max_depth: Maximum depth to search
        type_filter: Type filter (file or directory)

    Returns:
        JSON string containing search results
    """
    try:
        logging.info(f"Finding files in {search_directory} with pattern '{pattern}'")

        if not os.path.exists(search_directory) or not os.path.isdir(search_directory):
            return json.dumps({"error": f"Search directory does not exist or is not a directory: {search_directory}"})

        # Initialize results list
        results = []

        # Convert the pattern to a proper glob pattern if it's not already
        if "**" not in pattern and "*" not in pattern and "?" not in pattern:
            search_pattern = f"**/*{pattern}*"
        else:
            search_pattern = pattern

        # Create a full path pattern
        full_pattern = os.path.join(search_directory, search_pattern)

        # Get all matching paths
        matched_paths = glob.glob(full_pattern, recursive=True)

        # Filter by type if specified
        if type_filter:
            if type_filter == "file":
                matched_paths = [p for p in matched_paths if os.path.isfile(p)]
            elif type_filter == "directory":
                matched_paths = [p for p in matched_paths if os.path.isdir(p)]

        # Process includes patterns
        if includes:
            include_paths = set()
            for include_pattern in includes:
                full_include = os.path.join(search_directory, include_pattern)
                include_paths.update(glob.glob(full_include, recursive=True))
            matched_paths = [p for p in matched_paths if p in include_paths]

        # Process excludes patterns
        if excludes:
            exclude_paths = set()
            for exclude_pattern in excludes:
                full_exclude = os.path.join(search_directory, exclude_pattern)
                exclude_paths.update(glob.glob(full_exclude, recursive=True))
            matched_paths = [p for p in matched_paths if p not in exclude_paths]

        # Apply max_depth filter if specified
        if max_depth is not None:
            base_depth = search_directory.count(os.sep)
            matched_paths = [p for p in matched_paths if (p.count(os.sep) - base_depth) <= max_depth]

        # Process the results
        for path in matched_paths:
            rel_path = os.path.relpath(path, search_directory)

            if os.path.isdir(path):
                child_count = sum(len(files) for _, _, files in os.walk(path))
                results.append({
                    "path": rel_path,
                    "type": "directory",
                    "child_count": child_count,
                    "modified_time": os.path.getmtime(path)
                })
            else:
                results.append({
                    "path": rel_path,
                    "type": "file",
                    "size": os.path.getsize(path),
                    "modified_time": os.path.getmtime(path)
                })

        return json.dumps({"results": results}, indent=2)
    except Exception as e:
        logging.error(f"Error during find_by_name: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def view_file(absolute_path: str, start_line: int, end_line: int) -> str:
    """
    View the contents of a file with line numbers.

    The lines of the file are 0-indexed, and the output of this tool call will be
    the file contents from StartLine to EndLine, together with a summary of the
    lines outside of StartLine and EndLine. Note that this call can view at most
    200 lines at a time.

    Args:
        absolute_path: Path to file to view. Must be an absolute path.
        start_line: Startline to view (0-indexed)
        end_line: Endline to view. This cannot be more than 200 lines away from StartLine

    Returns:
        String containing the requested file contents with line summaries
    """
    try:
        logging.info(f"Viewing file: {absolute_path} from line {start_line} to {end_line}")

        if not os.path.exists(absolute_path) or not os.path.isfile(absolute_path):
            return json.dumps({"error": f"File does not exist or is not a file: {absolute_path}"})

        # Check line range
        if end_line < start_line:
            return json.dumps({"error": "End line must be greater than or equal to start line"})

        if end_line - start_line > 200:
            return json.dumps({"error": "Cannot view more than 200 lines at a time"})

        # Read the file
        with open(absolute_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Check line bounds
        total_lines = len(lines)
        if start_line >= total_lines:
            return json.dumps({"error": f"Start line {start_line} is out of bounds (file has {total_lines} lines)"})

        end_line = min(end_line, total_lines - 1)

        # Format the output
        result = []

        # Add file header
        result.append(f"File: {absolute_path}")
        result.append(f"Total lines: {total_lines}")
        result.append("")

        # Add summary of lines before start_line
        if start_line > 0:
            result.append(f"<... {start_line} lines not shown ...>")
            result.append("")

        # Add requested lines with line numbers
        for i in range(start_line, end_line + 1):
            line_content = lines[i].rstrip('\n')
            result.append(f"{i}: {line_content}")

        # Add summary of lines after end_line
        if end_line < total_lines - 1:
            result.append("")
            result.append(f"<... {total_lines - end_line - 1} more lines not shown ...>")

        return "\n".join(result)
    except Exception as e:
        logging.error(f"Error viewing file {absolute_path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def view_code_item(absolute_path: str, node_name: str) -> str:
    """
    View the content of a code item node, such as a class or a function in a file.

    You must use a fully qualified code item name. Such as those returned by the
    grep_search tool. For example, if you have a class called `Foo` and you want
    to view the function definition `bar` in the `Foo` class, you would use
    `Foo.bar` as the NodeName.

    Args:
        absolute_path: Path to the file to find the code node
        node_name: The name of the node to view

    Returns:
        String containing the code item content or an error message
    """
    try:
        logging.info(f"Viewing code item: {node_name} in file {absolute_path}")

        if not os.path.exists(absolute_path) or not os.path.isfile(absolute_path):
            return json.dumps({"error": f"File does not exist or is not a file: {absolute_path}"})

        # Read the file
        with open(absolute_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # This is a simplified approach - a real implementation would use an AST parser
        # for the specific language to locate the node more precisely

        # Determine the language based on file extension
        file_ext = os.path.splitext(absolute_path)[1].lower()

        # Extract the node content based on language
        if file_ext in ['.py']:
            # For Python files
            node_parts = node_name.split('.')

            # Look for class definition
            if len(node_parts) > 1:
                class_pattern = fr"class\s+{node_parts[0]}\s*(?:\([^)]*\))?\s*:"
                class_match = re.search(class_pattern, content)

                if class_match:
                    class_start = class_match.start()
                    # Find indentation of the class body
                    class_lines = content[class_start:].split('\n')
                    if len(class_lines) > 1:
                        class_indent = len(class_lines[1]) - len(class_lines[1].lstrip())

                        # Look for method definition
                        method_pattern = fr"^\s{{{class_indent}}}def\s+{node_parts[1]}\s*\("
                        method_match = re.search(method_pattern, content[class_start:], re.MULTILINE)

                        if method_match:
                            method_start = class_start + method_match.start()
                            method_lines = content[method_start:].split('\n')
                            method_indent = len(method_lines[0]) - len(method_lines[0].lstrip())

                            # Collect all lines in the method
                            method_content = [method_lines[0]]
                            for line in method_lines[1:]:
                                if line.strip() == "" or len(line) - len(line.lstrip()) > method_indent:
                                    method_content.append(line)
                                else:
                                    break

                            return "\n".join(method_content)

            # Look for function definition
            func_pattern = fr"def\s+{node_name}\s*\("
            func_match = re.search(func_pattern, content)

            if func_match:
                func_start = func_match.start()
                func_lines = content[func_start:].split('\n')
                func_indent = len(func_lines[0]) - len(func_lines[0].lstrip())

                # Collect all lines in the function
                func_content = [func_lines[0]]
                for line in func_lines[1:]:
                    if line.strip() == "" or len(line) - len(line.lstrip()) > func_indent:
                        func_content.append(line)
                    else:
                        break

                return "\n".join(func_content)

        elif file_ext in ['.js', '.ts']:
            # For JavaScript/TypeScript files (simplified)
            if '.' in node_name:
                parts = node_name.split('.')
                class_name = parts[0]
                method_name = parts[1]

                # Look for class and method
                class_pattern = fr"class\s+{class_name}\s*\{{"
                class_match = re.search(class_pattern, content)

                if class_match:
                    class_content = content[class_match.start():]
                    method_pattern = fr"{method_name}\s*\([^)]*\)\s*\{{"
                    method_match = re.search(method_pattern, class_content)

                    if method_match:
                        method_content = class_content[method_match.start():]
                        # Extract method content (simplified)
                        brace_count = 0
                        for i, char in enumerate(method_content):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    return method_content[:i+1]
            else:
                # Look for function
                func_pattern = fr"function\s+{node_name}\s*\("
                func_match = re.search(func_pattern, content)

                if func_match:
                    func_content = content[func_match.start():]
                    # Extract function content (simplified)
                    brace_count = 0
                    for i, char in enumerate(func_content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                return func_content[:i+1]

        # If we couldn't find the node
        return json.dumps({"error": f"Code item '{node_name}' not found in file {absolute_path}"})
    except Exception as e:
        logging.error(f"Error viewing code item {node_name} in {absolute_path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def related_files(absolute_path: str) -> str:
    """
    Finds other files that are related to or commonly used with the input file.

    Useful for retrieving adjacent files to understand context or make next edits.

    Args:
        absolute_path: Input file absolute path

    Returns:
        JSON string containing list of related files with relevance scores
    """
    try:
        logging.info(f"Finding files related to: {absolute_path}")

        if not os.path.exists(absolute_path) or not os.path.isfile(absolute_path):
            return json.dumps({"error": f"File does not exist or is not a file: {absolute_path}"})

        # Get the directory and filename
        directory = os.path.dirname(absolute_path)
        filename = os.path.basename(absolute_path)
        base_name, ext = os.path.splitext(filename)

        related = []

        # Identify related files based on various heuristics

        # 1. Files in the same directory with similar names
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isfile(entry_path) and entry != filename:
                entry_base, entry_ext = os.path.splitext(entry)

                # Calculate a simple relevance score based on name similarity
                relevance_score = 0.0

                # Same extension suggests related functionality
                if entry_ext == ext:
                    relevance_score += 0.3

                # Similar name suggests related functionality
                if entry_base.startswith(base_name) or base_name.startswith(entry_base):
                    relevance_score += 0.4
                elif entry_base in base_name or base_name in entry_base:
                    relevance_score += 0.2

                # Common naming patterns for related files
                related_patterns = {
                    "test": ["_test", "test_", "spec", "_spec"],
                    "implementation": ["impl", "implementation"],
                    "interface": ["interface", "if", "i_"],
                    "model": ["model", "schema", "entity"],
                    "controller": ["controller", "ctrl"],
                    "view": ["view", "template", "html"],
                }

                for category, patterns in related_patterns.items():
                    if any(pattern in base_name for pattern in patterns):
                        for pattern in patterns:
                            if pattern in entry_base:
                                relevance_score += 0.2
                                break

                # Check for imports (very simplified)
                if ext in ['.py', '.js', '.ts'] and relevance_score > 0:
                    try:
                        with open(absolute_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()

                        # Check if one file imports the other
                        if entry_base in content or base_name in open(entry_path, 'r', encoding='utf-8', errors='replace').read():
                            relevance_score += 0.5
                    except:
                        pass

                if relevance_score > 0:
                    related.append({
                        "path": entry_path,
                        "relevance_score": min(relevance_score, 1.0),
                        "related_type": "filename_similarity"
                    })

        # Sort by relevance score
        related.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Limit to top 10 results
        related = related[:10]

        return json.dumps({"related_files": related}, indent=2)
    except Exception as e:
        logging.error(f"Error finding related files for {absolute_path}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def run_command(command: str, cwd: str, args_list: List[str], blocking: bool,
                wait_ms_before_async: int) -> str:
    """
    PROPOSE a command to run on behalf of the user (macOS).

    Be sure to separate out the arguments into args. Passing in the full command
    with all args under "command" will not work. The user will have to approve
    the command before it is executed.

    Args:
        command: Name of the command to run
        cwd: The current working directory for the command
        args_list: The list of arguments to pass to the command
        blocking: If true, the command will block until it is entirely finished
        wait_ms_before_async: The amount of milliseconds to wait before making the command async

    Returns:
        JSON string containing command id and initial status
    """
    try:
        logging.info(f"Running command: {command} {' '.join(args_list)} in {cwd}")

        # Generate a unique ID for the command
        import uuid
        command_id = str(uuid.uuid4())

        # Format the command for display
        formatted_command = f"{command} {' '.join(args_list)}"

        # Store command details in registry
        ctx.command_registry[command_id] = {
            "command": command,
            "args": args_list,
            "cwd": cwd,
            "blocking": blocking,
            "wait_ms": wait_ms_before_async,
            "status": "pending_approval",
            "output": [],
            "error": None,
            "timestamp": asyncio.get_event_loop().time()
        }

        return json.dumps({
            "command_id": command_id,
            "command": formatted_command,
            "status": "pending_approval",
            "message": "This command requires user approval before execution."
        }, indent=2)
    except Exception as e:
        logging.error(f"Error proposing command {command}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def command_status(command_id: str, output_priority: str, output_character_count: int) -> str:
    """
    Get the status of a previously executed command by its ID.

    Returns the current status (running, done), output lines as specified by
    output priority, and any error if present.

    Args:
        command_id: ID of the command to get status for
        output_priority: Priority for displaying command output ('top', 'bottom', or 'split')
        output_character_count: Number of characters to view

    Returns:
        JSON string containing command status and output
    """
    try:
        logging.info(f"Getting status for command: {command_id}")

        if command_id not in ctx.command_registry:
            return json.dumps({"error": f"Command ID not found: {command_id}"})

        command_info = ctx.command_registry[command_id]

        # Limit output based on character count and priority
        output = "".join(command_info.get("output", []))
        if len(output) > output_character_count:
            if output_priority == "top":
                output = output[:output_character_count] + "... (output truncated)"
            elif output_priority == "bottom":
                output = "... (output truncated)" + output[-output_character_count:]
            elif output_priority == "split":
                half_count = output_character_count // 2
                output = output[:half_count] + "\n... (output truncated) ...\n" + output[-half_count:]

        return json.dumps({
            "command_id": command_id,
            "status": command_info.get("status", "unknown"),
            "output": output,
            "error": command_info.get("error", None),
            "runtime_seconds": asyncio.get_event_loop().time() - command_info.get("timestamp", 0)
        }, indent=2)
    except Exception as e:
        logging.error(f"Error getting command status for {command_id}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def write_to_file(target_file: str, code_content: str, empty_file: bool) -> str:
    """
    Create a new file with the specified content.

    The file and any parent directories will be created if they do not already exist.
    NEVER use this tool to modify or overwrite existing files.

    Args:
        target_file: The target file to create and write code to
        code_content: The code contents to write to the file
        empty_file: Set this to true to create an empty file

    Returns:
        JSON string with the result of the operation
    """
    try:
        logging.info(f"Writing to file: {target_file}")

        # Check if file already exists
        if os.path.exists(target_file):
            return json.dumps({"error": f"File already exists: {target_file}. Use edit_file to modify existing files."})

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(target_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Write the file
        with open(target_file, 'w', encoding='utf-8') as f:
            if not empty_file:
                f.write(code_content)

        file_size = os.path.getsize(target_file)

        return json.dumps({
            "success": True,
            "message": f"File created successfully: {target_file}",
            "file_path": target_file,
            "file_size": file_size,
            "is_empty": empty_file
        }, indent=2)
    except Exception as e:
        logging.error(f"Error writing to file {target_file}: {str(e)}")
        return json.dumps({"error": str(e)})


@mcp.tool()
def edit_file(target_file: str, code_edit: str, code_markdown_language: str,
              instruction: str, blocking: bool) -> str:
    """
    Edit an existing file.

    Follow these rules:
    1. Specify ONLY the precise lines of code that you wish to edit.
    2. **NEVER specify or write out unchanged code**. Instead, represent all unchanged
       code using the special placeholder: {{ ... }}.
    3. To edit multiple, non-adjacent lines of code in the same file, make a single
       call to this tool. Specify each edit in sequence with the special placeholder
       {{ ... }} to represent unchanged code in between edited lines.

    Args:
        target_file: The target file to modify
        code_edit: Specify ONLY the precise lines of code that you wish to edit
        code_markdown_language: Markdown language for the code block
        instruction: A description of the changes being made to the file
        blocking: If true, blocks until the entire file diff is generated

    Returns:
        JSON string with the result of the operation
    """
    try:
        logging.info(f"Editing file: {target_file}")

        # Check if file exists
        if not os.path.exists(target_file):
            return json.dumps({"error": f"File does not exist: {target_file}. Use write_to_file to create new files."})

        # Check if it's an .ipynb file (not allowed)
        if target_file.endswith('.ipynb'):
            return json.dumps({"error": "Editing .ipynb files is not supported"})

        # Read the original file content
        with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()

        # Process the edit instructions
        parts = code_edit.split("{{ ... }}")
        parts = [p.strip() for p in parts]

        # This is a simplified approach - a real implementation would need to handle
        # the edit more intelligently with proper diffing and patching

        # For this example, let's just log the edit instructions
        edit_details = {
            "file": target_file,
            "language": code_markdown_language,
            "instruction": instruction,
            "edit_sections": len(parts),
            "blocking": blocking
        }

        # In a real implementation, this would apply the edit to the file

        return json.dumps({
            "success": True,
            "message": f"File edit operation received: {target_file}",
            "instruction": instruction,
            "edit_details": edit_details,
            "note": "This is a simulation of the edit operation. In a real implementation, the edit would be applied to the file."
        }, indent=2)
    except Exception as e:
        logging.error(f"Error editing file {target_file}: {str(e)}")
        return json.dumps({"error": str(e)})