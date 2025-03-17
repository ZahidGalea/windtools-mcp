import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid

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