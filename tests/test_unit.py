import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from windtools_mcp.server import (
    ctx,  # Global context object
    codebase_search,
    get_initialization_status,
    index_repository,
    list_dir,
)

# ========== Test Fixtures ==========

@pytest.fixture
def setup_test_directory():
    """Fixture that creates a temporary directory structure for testing"""
    # Create a temporary directory
    test_dir = tempfile.mkdtemp()

    # Create directory structure and files for testing
    os.mkdir(os.path.join(test_dir, "subdir1"))
    os.mkdir(os.path.join(test_dir, "subdir2"))
    os.mkdir(os.path.join(test_dir, "subdir1", "nested"))

    # Create regular files
    with open(os.path.join(test_dir, "file1.txt"), "w") as f:
        f.write("This is a test file")
    with open(os.path.join(test_dir, "file2.txt"), "w") as f:
        f.write("Another test file with different content")
    with open(os.path.join(test_dir, "subdir1", "file3.txt"), "w") as f:
        f.write("Nested file content")

    yield test_dir

    # Clean up after tests
    import shutil
    shutil.rmtree(test_dir)


@pytest.fixture
def setup_code_directory():
    """Fixture that creates a temporary directory with sample code files"""
    test_dir = tempfile.mkdtemp()

    # Create Python files
    python_code = """
def hello_world():
    print("Hello, World!")
    return True

class TestClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"

def find_items(items, search_term):
    return [item for item in items if search_term in item]
"""

    # Create JavaScript files
    js_code = """
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

class TestClass {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return `Hello, ${this.name}!`;
    }
}

function findItems(items, searchTerm) {
    return items.filter(item => item.includes(searchTerm));
}
"""

    # Save the files
    with open(os.path.join(test_dir, "sample.py"), "w") as f:
        f.write(python_code)

    with open(os.path.join(test_dir, "sample.js"), "w") as f:
        f.write(js_code)

    # Create a subdirectory with more code
    os.mkdir(os.path.join(test_dir, "src"))
    with open(os.path.join(test_dir, "src", "utils.py"), "w") as f:
        f.write("""
def helper_function():
    return "I'm helping!"

def search_function(data, term):
    return [item for item in data if term in str(item)]
""")

    yield test_dir

    # Clean up after tests
    import shutil
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_chroma_db():
    """Fixture that mocks ChromaDB client and collection"""
    original_is_initialized = ctx.is_initialized
    original_chroma_client = ctx.chroma_client
    original_code_collection = ctx.code_collection

    # Create mock collection
    mock_collection = MagicMock()
    mock_collection.count.return_value = 50
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[
            {"file_path": "/path/to/file1.py"},
            {"file_path": "/path/to/file2.py"},
            {"file_path": "/path/to/file3.py"}
        ]],
        "documents": [[
            "def search_function(query):\n    return f\"Results for {query}\"",
            "class SearchEngine:\n    def search(self, query):\n        return []",
            "# Search utility functions\ndef preprocess_query(query):\n    return query.lower()"
        ]]
    }

    # Create mock client
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    # Set mocks in context
    ctx.is_initialized = True
    ctx.chroma_client = mock_client
    ctx.code_collection = mock_collection

    yield

    # Restore original values
    ctx.is_initialized = original_is_initialized
    ctx.chroma_client = original_chroma_client
    ctx.code_collection = original_code_collection


# ========== Tests for list_dir ==========

def test_list_dir_success(setup_test_directory):
    """Test that list_dir works correctly when the directory exists"""
    test_dir = setup_test_directory

    # Call the function with a real directory
    result = list_dir(test_dir)

    # Verify the result
    result_json = json.loads(result)

    # Basic result checks
    assert isinstance(result_json, list), "Result should be a list"
    assert len(result_json) > 0, "Result should not be empty"

    # Check for expected entries
    file_names = [item["path"] for item in result_json if item["type"] == "file"]
    dir_names = [item["path"] for item in result_json if item["type"] == "directory"]

    assert "file1.txt" in file_names, "Missing expected file file1.txt"
    assert "subdir1" in dir_names, "Missing expected directory subdir1"


def test_list_dir_nonexistent():
    """Test that list_dir handles non-existent directories properly"""
    # Create a path that definitely doesn't exist
    nonexistent_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function with the non-existent directory
    result = list_dir(nonexistent_dir)

    # Verify the result
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error message should indicate directory doesn't exist"


def test_list_dir_file_path():
    """Test that list_dir handles file paths (not directories) properly"""
    # Create a temporary file
    _, temp_file = tempfile.mkstemp()

    try:
        # Call the function with a file path instead of a directory
        result = list_dir(temp_file)

        # Verify the result
        result_json = json.loads(result)

        assert "error" in result_json, "Result should contain an error key"
        assert "not a directory" in result_json["error"], "Error message should indicate path is not a directory"
    finally:
        # Clean up
        os.unlink(temp_file)


# ========== Tests for get_initialization_status ==========

def test_get_initialization_status():
    """Test that get_initialization_status returns the current initialization status"""
    # Store original initialization state
    original_initialized = ctx.is_initialized
    original_error = ctx.initialization_error

    try:
        # Test uninitalized state
        ctx.is_initialized = False
        ctx.initialization_error = None

        result = get_initialization_status()
        result_json = json.loads(result)

        assert not result_json["is_initialized"], "Uninitialized state should report is_initialized as false"
        assert result_json["error"] is None, "Uninitialized state should have null error"

        # Test initialized state
        ctx.is_initialized = True

        result = get_initialization_status()
        result_json = json.loads(result)

        assert result_json["is_initialized"], "Initialized state should report is_initialized as true"

        # Test error state
        ctx.is_initialized = False
        ctx.initialization_error = "Test error message"

        result = get_initialization_status()
        result_json = json.loads(result)

        assert not result_json["is_initialized"], "Error state should report is_initialized as false"
        assert result_json["error"] == "Test error message", "Error message should be reported correctly"
    finally:
        # Restore original state
        ctx.is_initialized = original_initialized
        ctx.initialization_error = original_error


# ========== Tests for codebase_search ==========

def test_codebase_search_uninitialized():
    """Test that codebase_search handles uninitialized state properly"""
    # Store original initialization state
    original_initialized = ctx.is_initialized

    try:
        # Set uninitialized state
        ctx.is_initialized = False

        # Call the function
        result = codebase_search("test query")
        result_json = json.loads(result)

        assert "error" in result_json, "Uninitialized state should return an error"
        assert "not yet initialized" in result_json["error"], "Error should mention initialization"
    finally:
        # Restore original state
        ctx.is_initialized = original_initialized


def test_codebase_search_initialized(setup_code_directory, mock_chroma_db):
    """Test that codebase_search works correctly when ChromaDB is initialized"""
    # Call the function with mock ChromaDB
    result = codebase_search("search function")
    result_json = json.loads(result)

    assert "results" in result_json, "Result should contain results key"
    assert len(result_json["results"]) == 3, "Should return 3 results from mock"
    assert "search" in result_json["results"][0]["snippet"], "First result should contain search snippet"
    assert "relevance_score" in result_json["results"][0], "Results should have relevance scores"


# ========== Tests for index_repository ==========

def test_index_repository_uninitialized():
    """Test that index_repository handles uninitialized state properly"""
    # Store original initialization state
    original_initialized = ctx.is_initialized

    try:
        # Set uninitialized state
        ctx.is_initialized = False

        # Call the function
        result = index_repository(["/tmp"])
        result_json = json.loads(result)

        assert "error" in result_json, "Uninitialized state should return an error"
        assert "not yet initialized" in result_json["error"], "Error should mention initialization"
    finally:
        # Restore original state
        ctx.is_initialized = original_initialized


def test_index_repository_nonexistent_dir(mock_chroma_db):
    """Test that index_repository handles non-existent directories properly"""
    # Use a non-existent directory
    nonexistent_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function with a non-existent directory
    result = index_repository([nonexistent_dir])
    result_json = json.loads(result)

    # Should not error out with non-existent dirs, just return stats
    assert "status" in result_json, "Result should include a status field"
    assert result_json["status"] == "success", "Operation should succeed but skip non-existent dirs"
    assert "statistics" in result_json, "Result should include statistics"
    assert result_json["statistics"]["files_scanned"] == 0, "No files should be scanned"


def test_index_repository_with_force_reindex(setup_code_directory, mock_chroma_db):
    """Test index_repository with force_reindex=True"""
    # Mock the code collection's add and update methods
    orig_add = ctx.code_collection.add
    orig_update = ctx.code_collection.update
    
    try:
        ctx.code_collection.add = MagicMock()
        ctx.code_collection.update = MagicMock()
        
        # Call the function with force_reindex=True
        result = index_repository([setup_code_directory], force_reindex=True)
        result_json = json.loads(result)
        
        # Verify the result
        assert "status" in result_json, "Result should include a status field"
        assert result_json["status"] == "success", "Operation should succeed"
        
        # In force_reindex mode, we should see some calls to add/update
        assert ctx.code_collection.add.call_count > 0 or ctx.code_collection.update.call_count > 0, \
            "Should have called add or update with force_reindex=True"
    finally:
        # Restore original methods
        ctx.code_collection.add = orig_add
        ctx.code_collection.update = orig_update
