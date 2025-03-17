import json
import os
import shutil
import tempfile

import pytest

from windtools_mcp.server import list_dir


@pytest.fixture
def setup_test_directory():
    """Fixture que crea una estructura de directorios temporal para pruebas"""
    # Crear un directorio temporal
    test_dir = tempfile.mkdtemp()

    # Crear estructura de directorios y archivos para pruebas
    os.mkdir(os.path.join(test_dir, "subdir1"))
    with open(os.path.join(test_dir, "file1.txt"), "w") as f:
        f.write("contenido de prueba")
    with open(os.path.join(test_dir, "file2.txt"), "w") as f:
        f.write("otro contenido")
    with open(os.path.join(test_dir, "subdir1", "file3.txt"), "w") as f:
        f.write("más contenido de prueba")

    yield test_dir

    # Limpiar después de las pruebas
    shutil.rmtree(test_dir)


def test_list_dir_success(mocker, setup_test_directory):
    """Test que list_dir funciona correctamente cuando el directorio existe"""
    # Solo mockear los logs
    mock_log_error = mocker.patch('windtools_mcp.server.logging.error')
    mock_log_info = mocker.patch('windtools_mcp.server.logging.info')

    test_dir = setup_test_directory

    # Llamar a la función con un directorio real
    result = list_dir(test_dir)

    # Verificar el resultado
    result_json = json.loads(result)

    # Verificaciones básicas del resultado
    assert isinstance(result_json, list), "El resultado debería ser una lista"
    assert len(result_json) > 0, "El resultado no debería estar vacío"

    # Verificar que los logs se llamaron correctamente
    mock_log_info.assert_called_once_with(f"Listing directory: {test_dir}")
    mock_log_error.assert_not_called()


def test_list_dir_content_structure(mocker, setup_test_directory):
    """Test que verifica la estructura del contenido del directorio"""
    # Solo mockear los logs
    mocker.patch('windtools_mcp.server.logging.info')
    mocker.patch('windtools_mcp.server.logging.error')

    test_dir = setup_test_directory
    result = list_dir(test_dir)
    result_json = json.loads(result)

    # Verificar que es una lista
    assert isinstance(result_json, list)

    # Deberían haber al menos 3 elementos: file1.txt, file2.txt y subdir1
    assert len(result_json) >= 3

    # Verificar que existe cada tipo de elemento esperado
    file_entries = [item for item in result_json if item.get("type") == "file"]
    dir_entries = [item for item in result_json if item.get("type") == "directory"]

    # Verificar que hay archivos y directorios
    assert len(file_entries) >= 2, "Debería haber al menos 2 archivos"
    assert len(dir_entries) >= 1, "Debería haber al menos 1 directorio"

    # Verificar nombres y estructura
    file_paths = [entry.get("path") for entry in file_entries]
    all_paths = [entry.get("path") for entry in result_json]

    # Comprobar archivos específicos (usando rutas relativas o absolutas según la implementación)
    assert any("file1.txt" in path for path in file_paths), "Debería existir file1.txt"
    assert any("file2.txt" in path for path in file_paths), "Debería existir file2.txt"
    assert any("subdir1" in path for path in all_paths), "Debería existir subdir1"

    # Verificar que los archivos tienen tamaño
    for file_entry in file_entries:
        assert "size" in file_entry, "Los archivos deberían tener un tamaño"
        assert isinstance(file_entry["size"], int), "El tamaño debería ser un entero"
        assert file_entry["size"] > 0, "El tamaño debería ser mayor que cero"


def test_list_dir_nonexistent_directory(mocker):
    """Test que list_dir maneja correctamente un directorio que no existe"""
    # Solo mockear los logs
    mock_log_error = mocker.patch('windtools_mcp.server.logging.error')
    mock_log_info = mocker.patch('windtools_mcp.server.logging.info')

    # Llamar a la función con un directorio que no existe
    non_existent_dir = "/ruta/que/definitivamente/no/existe/en/el/sistema" + str(os.getpid())
    result = list_dir(non_existent_dir)

    # Verificar que devuelve un error
    result_json = json.loads(result)

    # El resultado podría ser un diccionario con error o una lista vacía
    if isinstance(result_json, dict):
        assert "error" in result_json
        assert isinstance(result_json["error"], str)
        assert len(result_json["error"]) > 0
    else:
        # Si es una lista, debería estar vacía o contener un mensaje de error
        assert len(result_json) == 0 or any("error" in item for item in result_json)

    # Verificar que los logs se llamaron correctamente
    mock_log_info.assert_called_once_with(f"Listing directory: {non_existent_dir}")
    mock_log_error.assert_called_once()


def test_list_dir_json_structure(mocker, setup_test_directory):
    """Test que verifica la estructura JSON del resultado"""
    # Solo mockear los logs
    mocker.patch('windtools_mcp.server.logging.info')

    test_dir = setup_test_directory
    result = list_dir(test_dir)

    # Verificar que es un JSON válido
    result_json = json.loads(result)

    # Regenerar el JSON con indent=2 para comparar
    formatted_json = json.dumps(result_json, indent=2)

    # El resultado original debe ser igual al JSON formateado
    assert result == formatted_json

    # Verificar la indentación
    assert "  " in result


def test_list_dir_nested_structure(mocker):
    """Test que verifica la estructura para directorios anidados más complejos"""
    # Solo mockear los logs
    mocker.patch('windtools_mcp.server.logging.info')
    mocker.patch('windtools_mcp.server.logging.error')

    # Crear estructura de directorios más compleja
    test_dir = tempfile.mkdtemp()
    try:
        # Crear varios niveles de directorios
        subdir1 = os.path.join(test_dir, "subdir1")
        subdir2 = os.path.join(subdir1, "subdir2")
        subdir3 = os.path.join(subdir1, "subdir3")
        os.makedirs(subdir2)
        os.makedirs(subdir3)

        # Crear archivos en diferentes niveles
        with open(os.path.join(test_dir, "root_file.txt"), "w") as f:
            f.write("archivo en raíz")
        with open(os.path.join(subdir1, "level1_file.txt"), "w") as f:
            f.write("archivo en nivel 1")
        with open(os.path.join(subdir2, "level2_file.txt"), "w") as f:
            f.write("archivo en nivel 2")
        with open(os.path.join(subdir3, "level3_file.txt"), "w") as f:
            f.write("archivo en nivel 3")

        # Llamar a la función
        result = list_dir(test_dir)
        result_json = json.loads(result)

        # Verificaciones básicas
        assert isinstance(result_json, list)

        # Buscar el subdirectorio en la lista
        # El subdirectorio puede estar en la lista principal o como parte de una jerarquía
        # dependiendo de cómo esté implementado _get_directory_info
        found_subdir1 = False
        for item in result_json:
            path = item.get("path", "")
            if "subdir1" in path:
                found_subdir1 = True
                break

        assert found_subdir1, "Debería existir subdir1 en el resultado"

    finally:
        # Limpiar
        shutil.rmtree(test_dir)