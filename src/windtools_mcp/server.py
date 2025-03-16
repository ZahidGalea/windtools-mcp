import logging
import os
# Add lifespan support for startup/shutdown with strong typing
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from logging import basicConfig, INFO
from typing import Any, AsyncIterator, List, Optional

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

# Obtener directorios permitidos de las variables de entorno
# Si no hay ninguno configurado, la lista estará vacía (ninguno permitido por defecto)
DEFAULT_ALLOWED_DIRECTORIES = [
    dir_path for dir_path in os.environ.get("ALLOWED_DIRECTORIES", "").split(",")
    if dir_path and dir_path.strip()
]

# Server lifespan context for ChromaDB initialization and project directories
@dataclass
class ServerContext:
    chroma_client: Any
    code_collection: Any
    embedding_model: str
    # Añadimos campos para el directorio de trabajo y directorios permitidos
    current_working_directory: Optional[str] = None
    allowed_directories: List[str] = field(default_factory=list)


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


def validate_path(path_to_check: str, allowed_dirs: List[str]) -> str:
    """
    Valida que una ruta esté dentro de los directorios permitidos.

    Args:
        path_to_check: Ruta a validar
        allowed_dirs: Lista de directorios permitidos

    Returns:
        Ruta normalizada y validada

    Raises:
        ValueError: Si la ruta está fuera de los directorios permitidos o no hay directorios permitidos
    """
    # Verificar si hay directorios permitidos configurados
    if not allowed_dirs:
        raise ValueError(
            "No hay directorios permitidos configurados. Utiliza add_allowed_directory primero."
        )

    expanded_path = expand_home(path_to_check)
    absolute_path = os.path.abspath(expanded_path)
    normalized_path = normalize_path(absolute_path)

    # Verificar si la ruta está dentro de los directorios permitidos
    is_allowed = any(
        normalized_path.startswith(normalize_path(d)) for d in allowed_dirs
    )
    if not is_allowed:
        allowed_dirs_str = ", ".join(allowed_dirs)
        raise ValueError(
            f"Acceso denegado - ruta fuera de directorios permitidos: {absolute_path} no está en {allowed_dirs_str}"
        )

    # Manejar enlaces simbólicos verificando su ruta real
    try:
        real_path = os.path.realpath(absolute_path)
        normalized_real = normalize_path(real_path)
        is_real_allowed = any(
            normalized_real.startswith(normalize_path(d)) for d in allowed_dirs
        )
        if not is_real_allowed:
            raise ValueError(
                "Acceso denegado - destino de enlace simbólico fuera de directorios permitidos"
            )
        return real_path
    except FileNotFoundError:
        # Para directorios nuevos que aún no existen, verificar el directorio padre
        parent_dir = os.path.dirname(absolute_path)
        try:
            real_parent = os.path.realpath(parent_dir)
            normalized_parent = normalize_path(real_parent)
            is_parent_allowed = any(
                normalized_parent.startswith(normalize_path(d)) for d in allowed_dirs
            )
            if not is_parent_allowed:
                raise ValueError(
                    "Acceso denegado - directorio padre fuera de directorios permitidos"
                )
            return absolute_path
        except FileNotFoundError:
            raise ValueError(f"El directorio padre no existe: {parent_dir}")


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

    # Normalizar los directorios permitidos desde la variable de entorno
    allowed_dirs = [
        normalize_path(expand_home(d)) for d in DEFAULT_ALLOWED_DIRECTORIES if d
    ]

    if allowed_dirs:
        logging.info(f"Directorios permitidos configurados: {allowed_dirs}")
    else:
        logging.warning("No hay directorios permitidos configurados. Las operaciones de sistema de archivos estarán restringidas hasta que se agreguen.")

    ctx = ServerContext(
        chroma_client=chroma_client,
        code_collection=code_collection,
        embedding_model=SENTENCE_TRANSFORMER_PATH,
        allowed_directories=allowed_dirs,
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
def set_project_directory(ctx: Context, directory_path: str) -> str:
    """
    Establece el directorio de proyecto para el servidor, validando que esté dentro
    de los directorios permitidos y almacenándolo en el contexto del servidor.

    Args:
        directory_path: Ruta absoluta al directorio del proyecto

    Returns:
        Mensaje de éxito o error
    """
    try:
        server_ctx = ctx.state

        # Verificar si hay directorios permitidos configurados
        if not server_ctx.allowed_directories:
            return "No hay directorios permitidos configurados. Utiliza add_allowed_directory primero."

        # Validar que la ruta esté dentro de los directorios permitidos
        valid_path = validate_path(directory_path, server_ctx.allowed_directories)

        # Verificar que es un directorio válido
        if not os.path.exists(valid_path):
            # Si el directorio no existe, intentamos crearlo
            try:
                os.makedirs(valid_path, exist_ok=True)
                logging.info(f"Created new project directory: {valid_path}")
            except Exception as e:
                return f"Error al crear el directorio: {str(e)}"
        elif not os.path.isdir(valid_path):
            return f"La ruta no es un directorio: {valid_path}"

        # Almacenar en el contexto del servidor
        server_ctx.current_working_directory = valid_path

        # También cambiamos el directorio actual como comportamiento adicional
        os.chdir(valid_path)

        return f"Directorio de proyecto establecido a: {valid_path}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logging.error(f"Error inesperado en set_project_directory: {str(e)}")
        return f"Error inesperado: {str(e)}"


@mcp.tool()
def get_project_directory(ctx: Context) -> str:
    """
    Obtiene el directorio de proyecto actual establecido en el contexto del servidor.

    Returns:
        Ruta al directorio actual o mensaje indicando que no se ha establecido
    """
    server_ctx = ctx.state
    if server_ctx.current_working_directory:
        return server_ctx.current_working_directory
    return "No se ha establecido un directorio de proyecto. Utiliza set_project_directory primero."


@mcp.tool()
def list_allowed_directories(ctx: Context) -> str:
    """
    Lista los directorios permitidos para operaciones de archivos.

    Returns:
        Lista de directorios permitidos formateada como texto
    """
    server_ctx = ctx.state
    if not server_ctx.allowed_directories:
        return "No hay directorios permitidos configurados. Utiliza add_allowed_directory para añadir directorios seguros."

    return "Directorios permitidos:\n" + "\n".join(server_ctx.allowed_directories)


@mcp.tool()
def add_allowed_directory(ctx: Context, directory_path: str) -> str:
    """
    Añade un nuevo directorio a la lista de directorios permitidos.

    Args:
        directory_path: Ruta al directorio que se desea permitir

    Returns:
        Mensaje de éxito o error
    """
    try:
        server_ctx = ctx.state

        # Normalizar y expandir la ruta
        expanded_path = expand_home(directory_path)
        absolute_path = os.path.abspath(expanded_path)
        normalized_path = normalize_path(absolute_path)

        # Verificar que existe
        if not os.path.exists(normalized_path):
            return f"El directorio no existe: {normalized_path}"

        if not os.path.isdir(normalized_path):
            return f"La ruta no es un directorio: {normalized_path}"

        # Verificar que no está ya en la lista
        if normalized_path in server_ctx.allowed_directories:
            return f"El directorio ya está en la lista: {normalized_path}"

        # Añadir a la lista
        server_ctx.allowed_directories.append(normalized_path)

        # Log para seguridad
        logging.info(f"Directorio añadido a la lista de permitidos: {normalized_path}")

        return f"Directorio añadido a la lista de permitidos: {normalized_path}"
    except Exception as e:
        return f"Error al añadir directorio: {str(e)}"


@mcp.tool()
def remove_allowed_directory(ctx: Context, directory_path: str) -> str:
    """
    Elimina un directorio de la lista de directorios permitidos.

    Args:
        directory_path: Ruta al directorio que se desea eliminar de la lista

    Returns:
        Mensaje de éxito o error
    """
    try:
        server_ctx = ctx.state

        # Normalizar y expandir la ruta
        expanded_path = expand_home(directory_path)
        absolute_path = os.path.abspath(expanded_path)
        normalized_path = normalize_path(absolute_path)

        # Verificar si está en la lista
        if normalized_path not in server_ctx.allowed_directories:
            # Intentar buscar si existe como subcadena
            matching_dirs = [d for d in server_ctx.allowed_directories if d.startswith(normalized_path) or normalized_path.startswith(d)]
            if matching_dirs:
                return f"Directorio no encontrado exactamente, pero existen coincidencias similares: {', '.join(matching_dirs)}"
            return f"El directorio no está en la lista: {normalized_path}"

        # Si el directorio de trabajo actual está dentro del directorio que se elimina,
        # resetear el directorio de trabajo actual
        if (server_ctx.current_working_directory and
                server_ctx.current_working_directory.startswith(normalized_path)):
            server_ctx.current_working_directory = None
            logging.warning(f"Se ha reseteado el directorio de trabajo actual porque estaba dentro de {normalized_path}")

        # Eliminar de la lista
        server_ctx.allowed_directories.remove(normalized_path)

        # Log para seguridad
        logging.info(f"Directorio eliminado de la lista de permitidos: {normalized_path}")

        return f"Directorio eliminado de la lista de permitidos: {normalized_path}"
    except Exception as e:
        return f"Error al eliminar directorio: {str(e)}"


@mcp.tool()
def list_dir(ctx: Context, directory_path: str = None) -> str:
    """
    Lista el contenido de un directorio, utilizando el directorio de proyecto actual
    si no se especifica uno.

    Args:
        directory_path: Ruta opcional al directorio que listar
                       (si no se proporciona, se usa el directorio de proyecto actual)

    Returns:
        Listado del directorio como texto
    """
    try:
        server_ctx = ctx.state

        # Verificar si hay directorios permitidos configurados
        if not server_ctx.allowed_directories:
            return "No hay directorios permitidos configurados. Utiliza add_allowed_directory primero."

        # Si no se proporciona ruta, usar el directorio actual del proyecto
        if directory_path is None:
            if not server_ctx.current_working_directory:
                return "No se ha establecido un directorio de proyecto. Utiliza set_project_directory primero."
            directory_path = server_ctx.current_working_directory

        # Validar la ruta
        valid_path = validate_path(directory_path, server_ctx.allowed_directories)

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