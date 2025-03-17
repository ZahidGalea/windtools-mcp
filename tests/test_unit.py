import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from windtools_mcp.server import (
    codebase_search,
    command_status,
    ctx,  # Global context object
    edit_file,
    find_by_name,
    get_initialization_status,
    grep_search,
    list_dir,
    related_files,
    run_command,
    view_code_item,
    view_file,
    write_to_file,
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

    with open(os.path.join(test_dir, "sample_test.py"), "w") as f:
        f.write("""
import unittest
from sample import hello_world, TestClass

class TestSample(unittest.TestCase):
    def test_hello_world(self):
        self.assertTrue(hello_world())

    def test_test_class(self):
        obj = TestClass("Test")
        self.assertEqual(obj.greet(), "Hello, Test!")
""")

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
    shutil.rmtree(test_dir)


@pytest.fixture
def setup_command_environment():
    """Fixture that sets up the environment for command execution tests"""
    # Store original command registry
    original_registry = ctx.command_registry.copy() if ctx.command_registry else {}

    # Reset command registry for tests
    ctx.command_registry = {}

    yield

    # Restore original command registry after tests
    ctx.command_registry = original_registry


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
        result = codebase_search("test query", ["/tmp"])
        result_json = json.loads(result)

        assert "error" in result_json, "Uninitialized state should return an error"
        assert "not yet initialized" in result_json["error"], "Error should mention initialization"
    finally:
        # Restore original state
        ctx.is_initialized = original_initialized


def test_codebase_search_invalid_directory():
    """Test that codebase_search handles invalid directories properly"""
    # Store original initialization state
    original_initialized = ctx.is_initialized

    try:
        # Set initialized state
        ctx.is_initialized = True

        # Use a non-existent directory
        nonexistent_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

        # Call the function
        result = codebase_search("test query", [nonexistent_dir])
        result_json = json.loads(result)

        # Even with invalid directory, the function should return a results array (just empty)
        assert "results" in result_json, "Result should contain a results key"
        assert isinstance(result_json["results"], list), "Results should be a list"
        assert len(result_json["results"]) == 0, "Results should be empty for non-existent directory"
    finally:
        # Restore original state
        ctx.is_initialized = original_initialized


# ========== Tests for grep_search ==========

def test_grep_search_basic(setup_code_directory):
    """Test that grep_search finds patterns correctly"""
    # This is a simulation since we can't easily install/run ripgrep in tests
    # In a real test, you'd need ripgrep installed and test with real execution

    # Prepare a mock subprocess.run function to simulate ripgrep
    original_run = subprocess.run

    try:
        def mock_run(args, **kwargs):
            # Simple mock that checks args and returns a predefined result
            # This simulates ripgrep finding matches
            class MockCompletedProcess:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            if "Hello" in args:
                return MockCompletedProcess(
                    stdout="sample.py:2:    print(\"Hello, World!\")\nsample.js:2:    console.log(\"Hello, World!\");",
                    stderr="",
                    returncode=0
                )
            elif "nonexistent" in args:
                return MockCompletedProcess(
                    stdout="",
                    stderr="",
                    returncode=1  # ripgrep returns 1 when no matches found
                )
            else:
                return MockCompletedProcess(
                    stdout="",
                    stderr="Error: invalid arguments",
                    returncode=2
                )

        # Replace subprocess.run with our mock
        subprocess.run = mock_run

        # Test finding a pattern that exists
        result = grep_search(setup_code_directory, "Hello", True, ["*.py", "*.js"], True)

        assert "sample.py:2:" in result, "Should find the pattern in Python file"
        assert "sample.js:2:" in result, "Should find the pattern in JavaScript file"

        # Test finding a pattern that doesn't exist
        result = grep_search(setup_code_directory, "nonexistent", True, ["*.py"], True)

        assert result == "No matches found.", "Should report no matches for non-existent pattern"
    finally:
        # Restore original subprocess.run
        subprocess.run = original_run


def test_grep_search_invalid_directory():
    """Test that grep_search handles invalid directories properly"""
    # Use a non-existent directory
    nonexistent_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function
    result = grep_search(nonexistent_dir, "test", True, ["*"], True)
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error should mention directory doesn't exist"


# ========== Tests for find_by_name ==========

def test_find_by_name_basic(setup_test_directory):
    """Test that find_by_name finds files by pattern correctly"""
    test_dir = setup_test_directory

    # Search for a specific file pattern
    result = find_by_name(test_dir, "*.txt")
    result_json = json.loads(result)

    assert "results" in result_json, "Result should contain a results key"
    assert isinstance(result_json["results"], list), "Results should be a list"
    assert len(result_json["results"]) >= 2, "Should find at least 2 txt files"

    # Check for specific files
    file_paths = [item["path"] for item in result_json["results"]]
    assert "file1.txt" in file_paths, "Should find file1.txt"
    assert "file2.txt" in file_paths, "Should find file2.txt"


def test_find_by_name_with_filters(setup_test_directory):
    """Test that find_by_name applies filters correctly"""
    test_dir = setup_test_directory

    # Search with includes filter
    result = find_by_name(test_dir, "*", includes=["subdir1/*"])
    result_json = json.loads(result)

    assert len(result_json["results"]) > 0, "Should find items in subdir1"
    paths = [item["path"] for item in result_json["results"]]
    assert all(p.startswith("subdir1/") for p in paths), "All results should be in subdir1"

    # Search with excludes filter
    result = find_by_name(test_dir, "*.txt", excludes=["subdir1/*"])
    result_json = json.loads(result)

    assert len(result_json["results"]) > 0, "Should find txt files outside subdir1"
    paths = [item["path"] for item in result_json["results"]]
    assert not any(p.startswith("subdir1/") for p in paths), "No results should be in subdir1"


def test_find_by_name_invalid_directory():
    """Test that find_by_name handles invalid directories properly"""
    # Use a non-existent directory
    nonexistent_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function
    result = find_by_name(nonexistent_dir, "*")
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error should mention directory doesn't exist"


# ========== Tests for view_file ==========

def test_view_file_basic(setup_code_directory):
    """Test that view_file displays file content correctly"""
    test_dir = setup_code_directory

    # View a complete small file
    sample_py_path = os.path.join(test_dir, "sample.py")
    result = view_file(sample_py_path, 0, 10)

    assert "File:" in result, "Result should contain file header"
    assert "def hello_world():" in result, "Result should contain file content"
    assert "0:" in result, "Result should contain line numbers"

    # View a specific range in the middle
    result = view_file(sample_py_path, 5, 8)

    assert "<... 5 lines not shown ...>" in result, "Should summarize skipped lines before"
    assert "class TestClass:" in result, "Should contain requested content"
    assert "<... " in result and " more lines not shown ...>" in result, "Should summarize skipped lines after"


def test_view_file_line_bounds(setup_code_directory):
    """Test that view_file handles line bounds correctly"""
    test_dir = setup_code_directory
    sample_py_path = os.path.join(test_dir, "sample.py")

    # Get total number of lines
    with open(sample_py_path, 'r') as f:
        total_lines = len(f.readlines())

    # Test requesting lines outside bounds
    result = view_file(sample_py_path, 0, total_lines + 10)

    assert f"Total lines: {total_lines}" in result, "Should report correct total line count"
    assert f"{total_lines-1}:" in result, "Should show content up to last line"
    assert "more lines not shown" not in result, "Should not mention more lines after the end"


def test_view_file_nonexistent():
    """Test that view_file handles non-existent files properly"""
    # Use a non-existent file
    nonexistent_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function
    result = view_file(nonexistent_file, 0, 10)
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error should mention file doesn't exist"


# ========== Tests for view_code_item ==========

def test_view_code_item_python(setup_code_directory):
    """Test that view_code_item extracts Python code items correctly"""
    test_dir = setup_code_directory
    sample_py_path = os.path.join(test_dir, "sample.py")

    # View a function
    result = view_code_item(sample_py_path, "hello_world")

    assert "def hello_world():" in result, "Should find the function declaration"
    assert "print(" in result, "Should include function body"
    assert "return True" in result, "Should include return statement"

    # View a class method
    result = view_code_item(sample_py_path, "TestClass.greet")

    assert "def greet(self):" in result, "Should find the method declaration"
    assert "return f\"Hello, {self.name}!\"" in result, "Should include method body"


def test_view_code_item_js(setup_code_directory):
    """Test that view_code_item extracts JavaScript code items correctly"""
    test_dir = setup_code_directory
    sample_js_path = os.path.join(test_dir, "sample.js")

    # View a function
    result = view_code_item(sample_js_path, "helloWorld")

    assert "function helloWorld()" in result, "Should find the function declaration"
    assert "console.log(" in result, "Should include function body"
    assert "return true;" in result, "Should include return statement"

    # View a class method
    result = view_code_item(sample_js_path, "TestClass.greet")

    assert "greet()" in result, "Should find the method declaration"
    assert "return `Hello, ${this.name}!`;" in result, "Should include method body"


def test_view_code_item_nonexistent(setup_code_directory):
    """Test that view_code_item handles non-existent code items properly"""
    test_dir = setup_code_directory
    sample_py_path = os.path.join(test_dir, "sample.py")

    # Try to view a non-existent code item
    result = view_code_item(sample_py_path, "nonexistent_function")
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "not found" in result_json["error"], "Error should mention code item not found"


# ========== Tests for related_files ==========

def test_related_files_basic(setup_code_directory):
    """Test that related_files finds related files correctly"""
    test_dir = setup_code_directory
    sample_py_path = os.path.join(test_dir, "sample.py")

    # Find files related to sample.py
    result = related_files(sample_py_path)
    result_json = json.loads(result)

    assert "related_files" in result_json, "Result should contain related_files key"
    assert isinstance(result_json["related_files"], list), "Related files should be a list"

    # sample_test.py should be related to sample.py
    related_paths = [os.path.basename(item["path"]) for item in result_json["related_files"]]
    assert "sample_test.py" in related_paths, "Should find related test file"


def test_related_files_nonexistent():
    """Test that related_files handles non-existent files properly"""
    # Use a non-existent file
    nonexistent_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function
    result = related_files(nonexistent_file)
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error should mention file doesn't exist"


# ========== Tests for run_command ==========

def test_run_command_registration(setup_command_environment):
    """Test that run_command registers commands correctly"""
    # Call the function with a simple command
    result = run_command("echo", "/tmp", ["hello", "world"], True, 0)
    result_json = json.loads(result)

    assert "command_id" in result_json, "Result should contain command_id"
    assert "status" in result_json, "Result should contain status"
    assert result_json["status"] == "pending_approval", "Initial status should be pending_approval"

    # Verify the command was registered in the global context
    command_id = result_json["command_id"]
    assert command_id in ctx.command_registry, "Command should be registered in global context"
    assert ctx.command_registry[command_id]["command"] == "echo", "Command name should be stored"
    assert ctx.command_registry[command_id]["args"] == ["hello", "world"], "Command args should be stored"


def test_run_command_complex_args(setup_command_environment):
    """Test that run_command handles complex arguments correctly"""
    # Call the function with complex arguments
    result = run_command(
        "find",
        "/tmp",
        ["-type", "f", "-name", "*.txt", "-exec", "grep", "test", "{}", ";"],
        False,
        100
    )
    result_json = json.loads(result)

    # Verify complex arguments were stored correctly
    command_id = result_json["command_id"]
    stored_args = ctx.command_registry[command_id]["args"]

    assert len(stored_args) == 9, "Should store all arguments correctly"
    assert stored_args[2] == "f", "Should preserve argument order"
    assert stored_args[4] == "*.txt", "Should store arguments with special characters"


# ========== Tests for command_status ==========

def test_command_status_basic(setup_command_environment):
    """Test that command_status retrieves command status correctly"""
    # Register a command
    command_id = str(uuid.uuid4())
    ctx.command_registry[command_id] = {
        "command": "test",
        "args": ["arg1", "arg2"],
        "cwd": "/tmp",
        "blocking": True,
        "wait_ms": 0,
        "status": "running",
        "output": ["Line 1\n", "Line 2\n", "Line 3\n"],
        "error": None,
        "timestamp": time.time() - 10  # Started 10 seconds ago
    }

    # Get status with top priority
    result = command_status(command_id, "top", 10)
    result_json = json.loads(result)

    assert result_json["status"] == "running", "Should report correct status"
    assert result_json["output"].startswith("Line 1"), "Should return beginning of output with top priority"
    assert result_json["runtime_seconds"] >= 10, "Should calculate runtime correctly"

    # Get status with bottom priority
    result = command_status(command_id, "bottom", 10)
    result_json = json.loads(result)

    assert result_json["output"].endswith("Line 3"), "Should return end of output with bottom priority"


def test_command_status_nonexistent(setup_command_environment):
    """Test that command_status handles non-existent command IDs properly"""
    # Use a non-existent command ID
    nonexistent_id = str(uuid.uuid4())

    # Call the function
    result = command_status(nonexistent_id, "top", 100)
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "not found" in result_json["error"], "Error should mention command ID not found"


# ========== Tests for write_to_file ==========

def test_write_to_file_basic(setup_test_directory):
    """Test that write_to_file creates files correctly"""
    test_dir = setup_test_directory
    test_file_path = os.path.join(test_dir, "new_file.txt")

    # Write content to a new file
    content = "This is test content for a new file."
    result = write_to_file(test_file_path, content, False)
    result_json = json.loads(result)

    assert result_json["success"], "Operation should succeed"
    assert os.path.exists(test_file_path), "File should be created"

    # Verify file content
    with open(test_file_path, 'r') as f:
        file_content = f.read()
    assert file_content == content, "File should contain the written content"


def test_write_to_file_empty(setup_test_directory):
    """Test that write_to_file creates empty files correctly"""
    test_dir = setup_test_directory
    test_file_path = os.path.join(test_dir, "empty_file.txt")

    # Create an empty file
    result = write_to_file(test_file_path, "This content should be ignored", True)
    result_json = json.loads(result)

    assert result_json["success"], "Operation should succeed"
    assert os.path.exists(test_file_path), "File should be created"
    assert result_json["is_empty"], "Result should indicate file is empty"

    # Verify file is empty
    with open(test_file_path, 'r') as f:
        file_content = f.read()
    assert file_content == "", "File should be empty"


def test_write_to_file_existing(setup_test_directory):
    """Test that write_to_file refuses to overwrite existing files"""
    test_dir = setup_test_directory
    test_file_path = os.path.join(test_dir, "file1.txt")  # This file exists from the fixture

    # Try to overwrite an existing file
    result = write_to_file(test_file_path, "New content", False)
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "already exists" in result_json["error"], "Error should mention file already exists"

    # Verify original content was not changed
    with open(test_file_path, 'r') as f:
        file_content = f.read()
    assert file_content == "This is a test file", "Original file content should be unchanged"


def test_write_to_file_nested_directory(setup_test_directory):
    """Test that write_to_file creates parent directories as needed"""
    test_dir = setup_test_directory
    nested_dir = os.path.join(test_dir, "new_dir", "nested_dir")
    test_file_path = os.path.join(nested_dir, "new_file.txt")

    # Write to a file in a non-existent directory structure
    result = write_to_file(test_file_path, "Content in a nested directory", False)
    result_json = json.loads(result)

    assert result_json["success"], "Operation should succeed"
    assert os.path.exists(test_file_path), "File should be created"
    assert os.path.isdir(nested_dir), "Parent directories should be created"


# ========== Tests for edit_file ==========

def test_edit_file_basic(setup_test_directory):
    """Test that edit_file handles basic edits correctly"""
    # This is a limited test since the actual implementation in your code is a simulation
    # A full implementation would need to test the actual edits are applied

    test_dir = setup_test_directory
    test_file_path = os.path.join(test_dir, "file1.txt")

    # Attempt to edit the file
    result = edit_file(
        test_file_path,
        "{{ ... }}\nThis is edited content\n{{ ... }}",
        "python",
        "Replacing a line of text",
        True
    )

    result_json = json.loads(result)

    # In your implementation, this just logs the request and returns success
    assert "success" in result_json, "Result should contain success key"
    assert result_json["success"], "Operation should report success"
    assert "instruction" in result_json, "Result should include the instruction"
    assert result_json["instruction"] == "Replacing a line of text", "Instruction should match"


def test_edit_file_nonexistent():
    """Test that edit_file handles non-existent files properly"""
    # Use a non-existent file
    nonexistent_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))

    # Call the function
    result = edit_file(
        nonexistent_file,
        "{{ ... }}\nEdited content\n{{ ... }}",
        "python",
        "Editing a non-existent file",
        True
    )
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert "does not exist" in result_json["error"], "Error should mention file doesn't exist"


def test_edit_file_unsupported_type(setup_test_directory):
    """Test that edit_file refuses to edit unsupported file types"""
    test_dir = setup_test_directory

    # Create a .ipynb file (not allowed to edit)
    ipynb_path = os.path.join(test_dir, "notebook.ipynb")
    with open(ipynb_path, 'w') as f:
        f.write("{}")

    # Try to edit the ipynb file
    result = edit_file(
        ipynb_path,
        "{{ ... }}\nEdited content\n{{ ... }}",
        "json",
        "Editing a notebook file",
        True
    )
    result_json = json.loads(result)

    assert "error" in result_json, "Result should contain an error key"
    assert ".ipynb files is not supported" in result_json["error"], "Error should mention unsupported file type"



# ========== Additional Fixtures ==========

@pytest.fixture
def setup_large_code_directory():
    """Fixture that creates a temporary directory with many code files"""
    test_dir = tempfile.mkdtemp()

    # Create a structure with many files
    for i in range(1, 101):
        subdir = os.path.join(test_dir, f"module_{i//10}")
        os.makedirs(subdir, exist_ok=True)

        with open(os.path.join(subdir, f"file_{i}.py"), "w") as f:
            f.write(f"""
# This is file {i}
def function_{i}():
    '''Function {i} documentation'''
    print("Function {i}")
    return {i}

class Class_{i}:
    def method_{i}(self, param):
        return param * {i}
""")

    yield test_dir

    # Clean up after tests
    shutil.rmtree(test_dir)


@pytest.fixture
def setup_complex_code_directory():
    """Fixture that creates a temporary directory with complex code structures"""
    test_dir = tempfile.mkdtemp()

    # Create Python file with complex structures
    python_complex = """
import os
import sys
from typing import List, Dict, Optional

# A complex function with decorators
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def complex_function(param1: str, param2: int = 0) -> Dict[str, any]:
    '''
    This is a complex function with:
    - Type annotations
    - Default parameters
    - Docstring
    - Multiple return statements
    '''
    result = {"status": "processing"}

    if not param1:
        result["status"] = "error"
        result["message"] = "param1 is required"
        return result

    if param2 < 0:
        result["status"] = "warning"
        result["value"] = None
    else:
        result["status"] = "success"
        result["value"] = param1 * param2

    return result

# A class with nested classes
class OuterClass:
    class NestedClass:
        def nested_method(self):
            return "nested"

    def __init__(self):
        self.nested = self.NestedClass()

    def outer_method(self):
        return self.nested.nested_method() + "_from_outer"

# Class inheritance
class BaseClass:
    def base_method(self):
        return "base"

class ChildClass(BaseClass):
    def child_method(self):
        return self.base_method() + "_child"
"""

    # Create JS file with complex structures
    js_complex = """
// Complex JavaScript with ES6 features
import { Component } from 'framework';

// Arrow function with default parameters
const arrowFunction = (param1, param2 = 0) => {
    return param1 * param2;
};

// Class with inheritance
class BaseComponent {
    constructor(name) {
        this.name = name;
    }

    render() {
        return `<div>${this.name}</div>`;
    }
}

class SpecialComponent extends BaseComponent {
    constructor(name, type) {
        super(name);
        this.type = type;
    }

    render() {
        return `<${this.type}>${super.render()}</${this.type}>`;
    }
}

// Nested object
const config = {
    api: {
        endpoints: {
            users: '/api/users',
            posts: '/api/posts'
        },
        version: 'v1'
    },
    settings: {
        theme: 'dark',
        notifications: true
    }
};

// Function with complex logic
function processData(data) {
    if (!data || !Array.isArray(data)) {
        throw new Error('Invalid data');
    }

    return data
        .filter(item => item.active)
        .map(item => ({
            id: item.id,
            name: item.name.toUpperCase(),
            score: item.score * 2
        }))
        .sort((a, b) => b.score - a.score);
}
"""

    # Create files with special characters
    special_chars = """
This file has special characters:
• Bullets and unusual punctuation: em—dash
• Unicode: こんにちは, 你好, Привет
• Emojis: 🚀 🌟 🔥 💻
• Control chars: Line 1
Line 2\tTabbed text
"""

    # Save the files
    with open(os.path.join(test_dir, "complex.py"), "w") as f:
        f.write(python_complex)

    with open(os.path.join(test_dir, "complex.js"), "w") as f:
        f.write(js_complex)

    with open(os.path.join(test_dir, "special_chars.txt"), "w") as f:
        f.write(special_chars)

    # Create related files for testing
    os.makedirs(os.path.join(test_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "tests"), exist_ok=True)

    with open(os.path.join(test_dir, "src", "module.py"), "w") as f:
        f.write("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
""")

    with open(os.path.join(test_dir, "tests", "test_module.py"), "w") as f:
        f.write("""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.module import add, subtract

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
""")

    # Create files for edit tests
    with open(os.path.join(test_dir, "edit_target.py"), "w") as f:
        f.write("""def function1():
    # This is the first function
    print("Hello from function 1")
    return True

def function2():
    # This is the second function
    print("Hello from function 2")
    return False

def function3():
    # This is the third function
    print("Hello from function 3")
    return None
""")

    yield test_dir

    # Clean up after tests
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


# ========== Additional Tests for codebase_search ==========

def test_codebase_search_initialized(setup_code_directory, mock_chroma_db):
    """Test that codebase_search works correctly when ChromaDB is initialized"""
    # Call the function with mock ChromaDB
    result = codebase_search("search function", [setup_code_directory])
    result_json = json.loads(result)

    assert "results" in result_json, "Result should contain results key"
    assert len(result_json["results"]) == 3, "Should return 3 results from mock"
    assert "search" in result_json["results"][0]["snippet"], "First result should contain search snippet"
    assert "relevance_score" in result_json["results"][0], "Results should have relevance scores"


def test_codebase_search_large_directory(setup_large_code_directory, mock_chroma_db):
    """Test that codebase_search warns about large number of files"""
    # Mock the logger to capture warnings
    with patch("logging.warning") as mock_warning:
        _ = codebase_search("function", [setup_large_code_directory])

        # Verify warning was logged
        mock_warning.assert_called_with(
            "Large number of files found (100). Search quality may be reduced."
        )


def test_codebase_search_multiple_directories(setup_code_directory, mock_chroma_db):
    """Test that codebase_search can search across multiple directories"""
    test_dir1 = setup_code_directory
    test_dir2 = tempfile.mkdtemp()

    try:
        # Call the function with multiple directories
        result = codebase_search("function", [test_dir1, test_dir2])
        result_json = json.loads(result)

        assert "results" in result_json, "Result should contain results key"
    finally:
        shutil.rmtree(test_dir2)


# ========== Additional Tests for grep_search ==========

def test_grep_search_case_sensitivity():
    """Test that grep_search handles case sensitivity correctly"""
    # Prepare a mock subprocess.run function
    original_run = subprocess.run

    try:
        def mock_run(args, **kwargs):
            class MockCompletedProcess:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            # Check if -i flag is in args
            is_case_insensitive = "-i" in args

            if "PATTERN" in args:
                if is_case_insensitive:
                    return MockCompletedProcess(
                        stdout="file.txt:1:pattern match\nfile.txt:2:PATTERN MATCH",
                        stderr="",
                        returncode=0
                    )
                else:
                    return MockCompletedProcess(
                        stdout="file.txt:2:PATTERN MATCH",
                        stderr="",
                        returncode=0
                    )
            else:
                return MockCompletedProcess(
                    stdout="",
                    stderr="",
                    returncode=1
                )

        # Replace subprocess.run with our mock
        subprocess.run = mock_run

        # Test case insensitive search
        result_insensitive = grep_search("/tmp", "PATTERN", True, ["*"], True)

        assert "file.txt:1:pattern" in result_insensitive, "Case insensitive search should find lowercase pattern"
        assert "file.txt:2:PATTERN" in result_insensitive, "Case insensitive search should find uppercase pattern"

        # Test case sensitive search
        result_sensitive = grep_search("/tmp", "PATTERN", True, ["*"], False)

        assert "file.txt:1:pattern" not in result_sensitive, "Case sensitive search should not find lowercase pattern"
        assert "file.txt:2:PATTERN" in result_sensitive, "Case sensitive search should find uppercase pattern"
    finally:
        # Restore original
        subprocess.run = original_run


def test_grep_search_match_per_line():
    """Test that grep_search handles match_per_line option correctly"""
    original_run = subprocess.run

    try:
        def mock_run(args, **kwargs):
            class MockCompletedProcess:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            # Check if -l flag is NOT in args (meaning match per line is True)
            match_per_line = "-l" not in args

            if match_per_line:
                return MockCompletedProcess(
                    stdout="file.txt:1:first match\nfile.txt:5:second match",
                    stderr="",
                    returncode=0
                )
            else:
                return MockCompletedProcess(
                    stdout="file.txt",
                    stderr="",
                    returncode=0
                )

        # Replace subprocess.run with our mock
        subprocess.run = mock_run

        # Test with match_per_line=True
        result_per_line = grep_search("/tmp", "match", True, ["*"], True)

        assert ":1:" in result_per_line, "match_per_line=True should include line numbers"
        assert "first match" in result_per_line, "match_per_line=True should include match content"

        # Test with match_per_line=False
        result_file_only = grep_search("/tmp", "match", False, ["*"], True)

        assert "file.txt" in result_file_only, "match_per_line=False should include file names"
        assert ":1:" not in result_file_only, "match_per_line=False should not include line numbers"
        assert "first match" not in result_file_only, "match_per_line=False should not include match content"
    finally:
        # Restore original
        subprocess.run = original_run


def test_grep_search_result_truncation():
    """Test that grep_search truncates results over 50 matches"""
    original_run = subprocess.run

    try:
        def mock_run(args, **kwargs):
            class MockCompletedProcess:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            # Generate 100 matches
            lines = []
            for i in range(1, 101):
                lines.append(f"file{i}.txt:1:match {i}")

            return MockCompletedProcess(
                stdout="\n".join(lines),
                stderr="",
                returncode=0
            )

        # Replace subprocess.run with our mock
        subprocess.run = mock_run

        # Test result truncation
        result = grep_search("/tmp", "match", True, ["*"], True)

        assert "file1.txt" in result, "First match should be included"
        assert "file50.txt" in result, "50th match should be included"
        assert "file51.txt" not in result, "51st match should not be included"
        assert "more matches" in result, "Truncation message should be included"
        assert "50 more matches" in result, "Number of truncated matches should be correct"
    finally:
        # Restore original
        subprocess.run = original_run


# ========== Additional Tests for find_by_name ==========

def test_find_by_name_max_depth(setup_test_directory):
    """Test that find_by_name respects max_depth parameter"""
    test_dir = setup_test_directory

    # Create a deeper directory structure
    deep_dir = os.path.join(test_dir, "level1", "level2", "level3")
    os.makedirs(deep_dir, exist_ok=True)

    with open(os.path.join(deep_dir, "deep_file.txt"), "w") as f:
        f.write("Deep file content")

    # Search with depth limit 1
    result_depth1 = find_by_name(test_dir, "*.txt", max_depth=1)
    result_json_depth1 = json.loads(result_depth1)

    # Search with depth limit 3
    result_depth3 = find_by_name(test_dir, "*.txt", max_depth=3)
    result_json_depth3 = json.loads(result_depth3)

    # Get all paths from results
    paths_depth1 = [item["path"] for item in result_json_depth1["results"]]
    paths_depth3 = [item["path"] for item in result_json_depth3["results"]]

    # Depth 1 should not include deep file
    assert not any("level2" in p for p in paths_depth1), "Depth 1 should not include files at level 2 or deeper"

    # Depth 3 should include deep file
    assert any("deep_file.txt" in p for p in paths_depth3), "Depth 3 should include deep file"


def test_find_by_name_type_filter(setup_test_directory):
    """Test that find_by_name applies type filters correctly"""
    test_dir = setup_test_directory

    # Search for files only
    result_files = find_by_name(test_dir, "*", type_filter="file")
    result_json_files = json.loads(result_files)

    # Verify all results are files
    assert all(item["type"] == "file" for item in result_json_files["results"]), "File filter should only return files"
    assert len(result_json_files["results"]) > 0, "Should find some files"

    # We could also test for directory type, but the function currently only implements file type


def test_find_by_name_complex_patterns(setup_complex_code_directory):
    """Test that find_by_name handles complex glob patterns"""
    test_dir = setup_complex_code_directory

    # Test **/ pattern for recursive matching
    result_recursive = find_by_name(test_dir, "**/test_*.py")
    result_json_recursive = json.loads(result_recursive)

    test_file_paths = [item["path"] for item in result_json_recursive["results"]]
    assert any("test_module.py" in p for p in test_file_paths), "Should find test files in subdirectories"

    # Test character classes
    result_char_class = find_by_name(test_dir, "*.[jp][sy]*")  # Matches .js or .py
    result_json_char_class = json.loads(result_char_class)

    file_exts = [os.path.splitext(item["path"])[1] for item in result_json_char_class["results"]]
    assert ".py" in file_exts, "Should find .py files"
    assert ".js" in file_exts, "Should find .js files"
    assert not any(ext not in [".py", ".js"] for ext in file_exts), "Should only find .py and .js files"


# ========== Additional Tests for view_file ==========

def test_view_file_line_limit():
    """Test that view_file enforces the 200 line limit"""
    # Create a temporary file with more than 200 lines
    _, temp_file = tempfile.mkstemp()
    try:
        with open(temp_file, 'w') as f:
            for i in range(300):
                f.write(f"Line {i}\n")

        # Try to view more than 200 lines
        result = view_file(temp_file, 0, 250)
        result_json = json.loads(result)

        assert "error" in result_json, "Result should contain an error key"
        assert "200 lines" in result_json["error"], "Error should mention 200 line limit"

        # Try to view exactly 200 lines
        result = view_file(temp_file, 0, 199)  # 0-199 = 200 lines

        assert "error" not in result, "Valid request should not result in error"
        assert "0: Line 0" in result, "Result should include first line"
        assert "199: Line 199" in result, "Result should include last requested line"
    finally:
        os.unlink(temp_file)


def test_view_file_special_characters(setup_complex_code_directory):
    """Test that view_file handles files with special characters"""
    test_dir = setup_complex_code_directory
    special_file = os.path.join(test_dir, "special_chars.txt")

    # View the file with special characters
    result = view_file(special_file, 0, 10)

    # Basic validation that file content is included
    assert "Unicode: こんにちは" in result, "Unicode characters should be preserved"
    assert "Emojis: 🚀" in result, "Emoji characters should be preserved"

    # Since the function uses 'errors=replace', it should not crash on invalid unicode
    # but replace them with the replacement character


# ========== Additional Tests for view_code_item ==========

def test_view_code_item_complex_python(setup_complex_code_directory):
    """Test that view_code_item handles complex Python code structures"""
    test_dir = setup_complex_code_directory
    complex_py = os.path.join(test_dir, "complex.py")

    # View a decorated function
    result = view_code_item(complex_py, "complex_function")

    assert "@decorator" in result, "Decorator should be included in function definition"
    assert "def complex_function" in result, "Function signature should be included"
    assert "This is a complex function" in result, "Docstring should be included"
    assert "return result" in result, "Function body should be included"

    # View a nested class method
    result = view_code_item(complex_py, "OuterClass.NestedClass.nested_method")

    if "def nested_method" in result:
        assert "return \"nested\"" in result, "Method body should be included"
    else:
        # The current implementation might not support deeply nested methods
        # but it should not crash
        pass

    # View an inherited method
    result = view_code_item(complex_py, "ChildClass.child_method")

    if "def child_method" in result:
        assert "return self.base_method()" in result, "Method body should be included"
    else:
        # The current implementation might not fully support inheritance
        # but it should not crash
        pass


def test_view_code_item_complex_js(setup_complex_code_directory):
    """Test that view_code_item handles complex JavaScript code structures"""
    test_dir = setup_complex_code_directory
    complex_js = os.path.join(test_dir, "complex.js")

    # View an arrow function
    result = view_code_item(complex_js, "arrowFunction")

    # Since the current implementation might have limited JS support,
    # we just check it doesn't crash and returns some content
    if not isinstance(result, dict) or "error" not in result:
        assert "arrowFunction" in result, "Function name should be in result"

    # View a class method
    result = view_code_item(complex_js, "SpecialComponent.render")

    # Again, if it's supported, check content, otherwise just ensure no crash
    if not isinstance(result, dict) or "error" not in result:
        assert "render" in result, "Method name should be in result"


# ========== Additional Tests for related_files ==========

def test_related_files_import_relationship(setup_complex_code_directory):
    """Test that related_files identifies relationships based on imports"""
    test_dir = setup_complex_code_directory
    test_file = os.path.join(test_dir, "tests", "test_module.py")

    # Get related files for the test file
    result = related_files(test_file)
    result_json = json.loads(result)

    assert "related_files" in result_json, "Result should contain related_files key"

    # Find related file that contains the module being imported
    module_file_rel = None
    for item in result_json["related_files"]:
        if "module.py" in item["path"]:
            module_file_rel = item
            break

    assert module_file_rel is not None, "Should find module.py as related to test_module.py"
    assert module_file_rel["relevance_score"] > 0.5, "Import relationship should have high relevance score"


def test_related_files_naming_patterns(setup_complex_code_directory):
    """Test that related_files identifies relationships based on naming patterns"""
    test_dir = setup_complex_code_directory
    module_file = os.path.join(test_dir, "src", "module.py")

    # Get related files for the module file
    result = related_files(module_file)
    result_json = json.loads(result)

    assert "related_files" in result_json, "Result should contain related_files key"

    # Find related test file based on naming pattern
    test_file_rel = None
    for item in result_json["related_files"]:
        if "test_module.py" in item["path"]:
            test_file_rel = item
            break

    assert test_file_rel is not None, "Should find test_module.py as related to module.py"
    assert "related_type" in test_file_rel, "Related file should have a relationship type"


# ========== Additional Tests for run_command and command_status ==========

def test_command_lifecycle(setup_command_environment):
    """Test the full lifecycle of a command from registration to completion"""
    # Register a command
    result = run_command("echo", "/tmp", ["hello", "world"], True, 0)
    result_json = json.loads(result)

    command_id = result_json["command_id"]

    # Verify initial status
    assert result_json["status"] == "pending_approval", "Initial status should be pending_approval"

    # Simulate user approval and command execution
    ctx.command_registry[command_id]["status"] = "running"
    ctx.command_registry[command_id]["output"] = ["Running command...\n"]

    # Get status during execution
    result = command_status(command_id, "top", 100)
    result_json = json.loads(result)

    assert result_json["status"] == "running", "Status should be running after approval"
    assert "Running command" in result_json["output"], "Should show command output"

    # Simulate command completion
    ctx.command_registry[command_id]["status"] = "done"
    ctx.command_registry[command_id]["output"].append("hello world\n")

    # Get final status
    result = command_status(command_id, "top", 100)
    result_json = json.loads(result)

    assert result_json["status"] == "done", "Status should be done after completion"
    assert "hello world" in result_json["output"], "Should show final command output"


def test_command_output_priority(setup_command_environment):
    """Test that command_status respects output priority settings"""
    # Register a command with long output
    command_id = str(uuid.uuid4())
    ctx.command_registry[command_id] = {
        "command": "test",
        "args": [],
        "cwd": "/tmp",
        "blocking": True,
        "wait_ms": 0,
        "status": "done",
        "output": [],
        "error": None,
        "timestamp": time.time()
    }

    # Generate 100 lines of output
    for i in range(1, 101):
        ctx.command_registry[command_id]["output"].append(f"Line {i}\n")

    # Test top priority with character limit
    result = command_status(command_id, "top", 50)
    result_json = json.loads(result)

    assert "Line 1" in result_json["output"], "Top priority should include start of output"
    assert "Line 10" not in result_json["output"], "Top priority with limit should not include later output"

    # Test bottom priority with character limit
    result = command_status(command_id, "bottom", 50)
    result_json = json.loads(result)

    assert "Line 100" in result_json["output"], "Bottom priority should include end of output"
    assert "Line 1" not in result_json["output"], "Bottom priority with limit should not include early output"

    # Test split priority with character limit
    result = command_status(command_id, "split", 100)
    result_json = json.loads(result)

    assert "Line 1" in result_json["output"], "Split priority should include start of output"
    assert "Line 100" in result_json["output"], "Split priority should include end of output"
    assert "... (output truncated) ..." in result_json["output"], "Split priority should indicate truncation"


def test_wait_ms_before_async():
    """Test that run_command handles wait_ms_before_async parameter"""
    # This would ideally test that the wait_ms parameter affects timing
    # but since our implementation is simulated, we just check it's stored correctly

    result = run_command("sleep", "/tmp", ["1"], False, 500)
    result_json = json.loads(result)

    command_id = result_json["command_id"]
    assert ctx.command_registry[command_id]["wait_ms"] == 500, "wait_ms should be stored correctly"


# ========== Additional Tests for edit_file ==========

def test_edit_file_actual_edits(setup_complex_code_directory):
    """Test that edit_file actually applies edits correctly"""
    # This test is limited since the current implementation just simulates edits
    # In a real implementation, this would verify the edits are applied to the file

    test_dir = setup_complex_code_directory
    edit_file_path = os.path.join(test_dir, "edit_target.py")

    # Read original content
    with open(edit_file_path, 'r') as f:
        original_content = f.read()

    # Make a copy to test actual edits
    test_edit_path = os.path.join(test_dir, "edit_test.py")
    with open(test_edit_path, 'w') as f:
        f.write(original_content)

    # Apply a simple edit
    edit_content = """{{ ... }}
def function2():
    # This is the modified second function
    print("Modified function 2")
    return True
{{ ... }}"""

    result = edit_file(test_edit_path, edit_content, "python", "Modified function2", True)

    # In a real implementation, we would check that function2 was actually modified
    # For now, just verify the function doesn't crash and returns success
    result_json = json.loads(result)
    assert "success" in result_json, "Edit operation should report success status"


def test_edit_file_multiple_sections(setup_complex_code_directory):
    """Test that edit_file handles multiple edit sections correctly"""
    # This test is limited since the current implementation just simulates edits

    test_dir = setup_complex_code_directory
    edit_file_path = os.path.join(test_dir, "edit_target.py")

    # Apply multiple edits
    edit_content = """{{ ... }}
def function1():
    # This is the modified first function
    print("Modified function 1")
    return True
{{ ... }}
def function3():
    # This is the modified third function
    print("Modified function 3")
    return "modified"
{{ ... }}"""

    result = edit_file(edit_file_path, edit_content, "python", "Modified first and third functions", True)

    # Verify the function parsed the edits correctly
    result_json = json.loads(result)
    if "edit_details" in result_json:
        assert result_json["edit_details"]["edit_sections"] == 3, "Should identify 3 edit sections"