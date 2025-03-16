# WindTools MCP Server

MCP Server for the WindTools code assistant, providing document embedding and retrieval capabilities using ChromaDB and sentence transformers.

## Features

- **Semantic Code Search**: Uses sentence transformers for embedding code snippets and retrieval
- **Persistent Storage**: Saves code embeddings in ChromaDB for persistent retrieval
- **Directory Exploration**: Built-in tools for navigating and exploring codebases
- **Environment Configuration**: Configurable through environment variables

## Tools

1. `list_dir`
   - List the contents of a directory
   - Inputs:
     - `directory_path` (string): Path to list contents of, should be absolute path to a directory
   - Returns: String with directory listing including file types and sizes

## Technical Architecture

The WindTools MCP Server is built on these key components:

- **ChromaDB**: Vector database for storing and retrieving code embeddings
- **Sentence Transformers**: Deep learning models for creating embeddings from code
- **FastMCP**: Framework for building MCP-compliant servers
- **Async Lifespan Management**: Efficient resource initialization and cleanup

## Setup

### Environment Variables

The server can be configured with the following environment variables:

- `CHROMA_DB_FOLDER_NAME`: Name of the folder where ChromaDB stores data (default: "default")
- `SENTENCE_TRANSFORMER_PATH`: Path to the sentence transformer model (default: "Linq-AI-Research/Linq-Embed-Mistral")

### Installation

#### Using pip

```bash
pip install windtools-mcp
```

#### From source

```bash
git clone <repository-url>
cd windtools-mcp
pip install -e .
```

### Usage with Claude Desktop

Add the following to your `claude_desktop_config.json`:

#### Docker

```json
{
  "mcpServers": { 
    "windtools": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "CHROMA_DB_FOLDER_NAME",
        "-e",
        "SENTENCE_TRANSFORMER_PATH",
        "windtools-mcp"
      ],
      "env": {
        "CHROMA_DB_FOLDER_NAME": "default",
        "SENTENCE_TRANSFORMER_PATH": "Linq-AI-Research/Linq-Embed-Mistral"
      }
    }
  }
}
```

#### Direct Execution

```json
{
  "mcpServers": {
    "windtools": {
      "command": "mcp-wintools",
      "env": {
        "CHROMA_DB_FOLDER_NAME": "default",
        "SENTENCE_TRANSFORMER_PATH": "jinaai/jina-embeddings-v2-base-code"
      }
    }
  }
}
```

## Build

Docker build:

```bash
docker build -t windtools-mcp .
```

## Development

### Requirements

- Python 3.10.13 or higher
- Dependencies listed in pyproject.toml

### Development Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv sync --dev
```

### Running Tests

```bash
pytest tests/
```

## Project Structure

```
src/
  windtools_mcp/
    __init__.py
    __main__.py
    server.py
tests/
  test_client.py
.gitignore
.python-version
Dockerfile
pyproject.toml
```

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License.