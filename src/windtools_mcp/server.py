import logging
import os
import os.path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import basicConfig, INFO
from typing import Any, AsyncIterator, Optional

from mcp.server.fastmcp import Context, FastMCP

basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s",
)
# PARAMS
CHROMA_DB_FOLDER_NAME = os.environ.get("CHROMA_DB_FOLDER_NAME", "default")
SENTENCE_TRANSFORMER_PATH = os.environ.get(
    "SENTENCE_TRANSFORMER_PATH", "jinaai/jina-embeddings-v2-base-code"
)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DB_PATH = os.path.join(DATA_FOLDER, CHROMA_DB_FOLDER_NAME)
SENTENCE_TRANSFORMER_CACHE_FOLDER = os.path.join(DATA_FOLDER, "embedding_cache")


# Server lifespan context for ChromaDB initialization and project directory
@dataclass
class ServerContext:
    chroma_client: Any
    code_collection: Any
    embedding_model: str

    working_directory: Optional[str] = None


# Funciones de utilidad para validar rutas
def normalize_path(p: str) -> str:
    """Normaliza una ruta para validaciones consistentes"""
    return os.path.normpath(p)


def expand_home(filepath: str) -> str:
    """Expande el símbolo ~ para representar el directorio home"""
    if filepath.startswith("~/") or filepath == "~":
        return os.path.join(
            os.path.expanduser("~"), filepath[1:] if filepath != "~" else ""
        )
    return filepath


def validate_path(path_to_check: str, working_dir: str) -> str:
    """
    Valida que una ruta esté dentro del directorio de trabajo.

    Args:
        path_to_check: Ruta a validar
        working_dir: Directorio de trabajo base

    Returns:
        Ruta normalizada y validada

    Raises:
        ValueError: Si la ruta está fuera del directorio de trabajo o no hay directorio configurado
    """
    # Verificar si hay directorio de trabajo configurado
    if not working_dir:
        raise ValueError(
            "No hay directorio de trabajo configurado. Utiliza set_working_directory primero."
        )

    expanded_path = expand_home(path_to_check)
    absolute_path = os.path.abspath(expanded_path)
    normalized_path = normalize_path(absolute_path)
    normalized_working_dir = normalize_path(working_dir)

    # Verificar si la ruta está dentro del directorio de trabajo
    if not normalized_path.startswith(normalized_working_dir):
        raise ValueError(
            f"Acceso denegado - ruta fuera del directorio de trabajo: {absolute_path} no está en {normalized_working_dir}"
        )

    # Manejar enlaces simbólicos verificando su ruta real
    try:
        real_path = os.path.realpath(absolute_path)
        normalized_real = normalize_path(real_path)

        if not normalized_real.startswith(normalized_working_dir):
            raise ValueError(
                "Acceso denegado - destino de enlace simbólico fuera del directorio de trabajo"
            )
        return real_path
    except FileNotFoundError:
        # Para archivos nuevos que aún no existen, verificar el directorio padre
        parent_dir = os.path.dirname(absolute_path)
        try:
            real_parent = os.path.realpath(parent_dir)
            normalized_parent = normalize_path(real_parent)

            if not normalized_parent.startswith(normalized_working_dir):
                raise ValueError(
                    "Acceso denegado - directorio padre fuera del directorio de trabajo"
                )
            return absolute_path
        except FileNotFoundError:
            raise ValueError(f"El directorio padre no existe: {parent_dir}")

def get_working_directory(ctx: Context):
    if "working_directory" in ctx.request_context.lifespan_context:
        return ctx.request_context.lifespan_context["working_direcotry"]
    return "No se ha establecido un directorio de trabajo"


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

    ctx = ServerContext(
        chroma_client=chroma_client,
        code_collection=code_collection,
        embedding_model=SENTENCE_TRANSFORMER_PATH,
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
def set_working_directory(ctx: Context, directory_path: str) -> str:
    """
    Establece el directorio de trabajo para el servidor. Todas las operaciones
    de archivos estarán restringidas a este directorio y sus subdirectorios.

    Args:
        directory_path: Ruta absoluta al directorio de trabajo

    Returns:
        Mensaje de éxito o error
    """
    try:

        # Normalizar y expandir la ruta
        expanded_path = expand_home(directory_path)
        absolute_path = os.path.abspath(expanded_path)
        normalized_path = normalize_path(absolute_path)

        # Verificar que existe
        if not os.path.exists(normalized_path):
            # Si el directorio no existe, intentamos crearlo
            try:
                os.makedirs(normalized_path, exist_ok=True)
                logging.info(f"Created new working directory: {normalized_path}")
            except Exception as e:
                return f"Error al crear el directorio: {str(e)}"
        elif not os.path.isdir(normalized_path):
            return f"La ruta no es un directorio: {normalized_path}"

        # Almacenar en el contexto del servidor
        ctx.request_context.lifespan_context["working_direcotry"] = normalized_path

        # También cambiamos el directorio actual como comportamiento adicional
        os.chdir(normalized_path)

        logging.info(f"Directorio de trabajo establecido: {normalized_path}")
        return f"Directorio de trabajo establecido: {normalized_path}"
    except Exception as e:
        logging.error(f"Error al establecer directorio de trabajo: {str(e)}")
        return f"Error: {str(e)}"


@mcp.resource("config://working_directory")
def get_working_directory_resource(ctx: Context) -> str:
    """
    Obtiene el directorio de trabajo actual establecido en el contexto del servidor.

    Returns:
        Ruta al directorio de trabajo o mensaje indicando que no se ha establecido
    """
    return get_working_directory(ctx)



@mcp.resource("config://{subdirectory}/list_dir")
def list_dir(ctx: Context, subdirectory: str = "") -> str:
    """
    Lista el contenido de un subdirectorio dentro del directorio de trabajo.

    Args:
        subdirectory: Ruta relativa al subdirectorio que listar (opcional)
                     Si no se proporciona, se lista el directorio de trabajo principal.

    Returns:
        Listado del directorio como texto
    """
    try:
        working_directory = get_working_directory(ctx)

        # Construir la ruta completa
        if subdirectory:
            directory_path = os.path.join(working_directory, subdirectory)
        else:
            directory_path = working_directory

        # Validar la ruta
        valid_path = validate_path(directory_path, working_directory)

        if not os.path.exists(valid_path) or not os.path.isdir(valid_path):
            return f"Directorio no encontrado o no es un directorio: {valid_path}"

        results = []

        for item in sorted(os.listdir(valid_path)):
            item_path = os.path.join(valid_path, item)

            if os.path.isdir(item_path):
                # Contar número de elementos en el directorio
                try:
                    item_count = len(os.listdir(item_path))
                    results.append(f"{item}\tdir\t{item_count} elementos")
                except PermissionError:
                    results.append(f"{item}\tdir\t(sin acceso)")
            else:
                # Obtener el tamaño del archivo
                try:
                    size = os.path.getsize(item_path)
                    results.append(f"{item}\tfile\t{size} bytes")
                except PermissionError:
                    results.append(f"{item}\tfile\t(sin acceso)")

        if not results:
            return f"El directorio {valid_path} está vacío"

        return "\n".join(results)

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error al listar directorio: {str(e)}"