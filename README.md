# WindTools MCP Server

MCP Server for the WindTools code assistant, providing document embedding and retrieval capabilities using ChromaDB and
sentence transformers.

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
    - Returns: JSON string containing directory information including file types and sizes

## Technical Architecture

The WindTools MCP Server is built on these key components:

- **ChromaDB**: Vector database for storing and retrieving code embeddings
- **Sentence Transformers**: Deep learning models for creating embeddings from code
- **FastMCP**: Framework for building MCP-compliant servers
- **Async Lifespan Management**: Efficient resource initialization and cleanup

## Setup

### Environment Variables

The server can be configured with the following environment variables:

- `DATA_ROOT`: Absolute directory where ChromaDB database and model cache will be stored (default: a 'data' directory
  inside the package)
- `CHROMA_DB_FOLDER_NAME`: Name of the folder where ChromaDB stores data (default: "default")
- `SENTENCE_TRANSFORMER_PATH`: Path to the sentence transformer model (default: "jinaai/jina-embeddings-v2-base-code")

### Installation

#### Using pip

```bash
pip install windtools-mcp
```

#### From source

```bash
git clone https://github.com/ZahidGalea/windtools-mcp
cd windtools-mcp
pip install -e .
```

### Usage with Claude Desktop

Add the following to your `claude_desktop_config.json`:

#### Direct Execution

Using Python 3.11 as ChromaDB has issues with newer Python versions.

```json
{
  "mcpServers": {
    "windtools": {
      "command": "uvx",
      "args": [
        "-p",
        "3.11",
        "windtools-mcp"
      ],
      "env": {
        "DATA_ROOT": "/Users/<user>/windtools_data",
        "CHROMA_DB_FOLDER_NAME": "chromadb",
        "SENTENCE_TRANSFORMER_PATH": "jinaai/jina-embeddings-v2-base-code"
      }
    }
  }
}
```

Data (including ChromaDB database and model cache) will be saved in the `/Users/<user>/windtools_data` directory and
persist between
container executions.

## Development

### Requirements

- Python 3.11
- Dependencies listed in pyproject.toml

### Development Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv sync --dev
```

### Inspector

```bash
npx @modelcontextprotocol/inspector uvx -p 3.11 windtools-mcp
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
pyproject.toml
README.md
```

## Release Process

La versión del proyecto se gestiona de forma centralizada en el archivo `VERSION`. El proceso de release es automático:

1. Actualizar el número de versión en el archivo `VERSION`
2. Hacer commit y push a la rama `main`
3. El flujo de trabajo de GitHub Actions automáticamente:
    - Detectará el cambio en el archivo `VERSION`
    - Creará un tag de git con el formato `v{VERSION}`
    - Generará una release en GitHub
    - Publicará el paquete en PyPI

```bash
# Ejemplo de actualización de versión
echo "0.0.13" > VERSION
git add VERSION
git commit -m "Bump version to 0.0.13"
git push origin main
```

No es necesario crear manualmente tags o publicar en PyPI, todo se gestiona automáticamente cuando se actualiza el
archivo `VERSION`.

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software,
subject to the terms and conditions of the MIT License.