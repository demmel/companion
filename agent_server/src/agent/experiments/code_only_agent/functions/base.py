"""Base function implementations for the agent."""

import os
from datetime import datetime
from typing import TYPE_CHECKING

from agent.experiments.code_only_agent.functions.definitions import FunctionDef

if TYPE_CHECKING:
    from agent.experiments.code_only_agent.state import State


def create_speak_function(state: "State"):
    """Create a speak function that captures messages in state."""

    def speak(message: str) -> str:
        """
        Send a message to the user.

        Args:
            message: The message to send to the user

        Returns:
            Confirmation that message was sent
        """
        state.speak_messages.append(message)
        print(f"[SPEAK] {message}")
        return f"Message sent to user: {message}"

    return speak


def create_read_file_function():
    """Create a read_file function."""

    def read_file(path: str) -> str:
        """
        Read contents of a file at the given path.

        Args:
            path: Path to the file to read

        Returns:
            Contents of the file, or error message if reading fails
        """
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"

    return read_file


def create_get_time_function():
    """Create a get_time function."""

    def get_time() -> str:
        """
        Get the current date and time.

        Returns:
            Current date and time in format 'YYYY-MM-DD HH:MM:SS'
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return get_time


def create_list_files_function():
    """Create a list_files function."""

    def list_files(directory: str) -> list[str]:
        """
        List all files in a directory.

        Args:
            directory: Path to the directory to list

        Returns:
            List of filenames in the directory, or error message if listing fails
        """
        try:
            return os.listdir(directory)
        except Exception as e:
            return [f"Error listing directory: {type(e).__name__}: {e}"]

    return list_files


def create_find_functions_function(state: "State"):
    """Create a find_functions function that searches the function tree."""

    def find_functions(in_dir: str = "", name: str = "") -> str:
        """
        Search for functions in the function tree.

        Args:
            in_dir: Directory path to search (e.g., "filesystem", "system/time", "" for root)
            name: Function name to search for (supports fuzzy matching)

        Returns:
            Formatted string showing matching functions
        """
        results = state.function_tree.find_functions(in_dir, name)

        if not results:
            return f"No functions found matching '{name}' in '{in_dir or 'root'}'"

        # Add found functions to cache so they're loaded for use
        for func_def in results:
            state.function_cache.add(func_def)

        lines = []
        for func_def in results:
            lines.append(f"{func_def.signature}")
            lines.append(f"  {func_def.description}")

        return "\n".join(lines)

    return find_functions


def initialize_base_functions(state: "State") -> None:
    """Initialize base functions in the function tree and cache."""
    # Create functions
    speak_func = create_speak_function(state)
    read_file_func = create_read_file_function()
    get_time_func = create_get_time_function()
    list_files_func = create_list_files_function()
    find_functions_func = create_find_functions_function(state)

    # Define function metadata with paths
    function_defs = [
        ("communication", FunctionDef(
            name="speak",
            func=speak_func,
            signature="speak(message: str) -> str",
            description="""Send a message to the user.

Args:
    message: The message to send to the user

Returns:
    Confirmation that message was sent""",
            category="communication",
            path="communication",
        )),
        ("filesystem", FunctionDef(
            name="read_file",
            func=read_file_func,
            signature="read_file(path: str) -> str",
            description="""Read contents of a file at the given path.

Args:
    path: Path to the file to read

Returns:
    Contents of the file, or error message if reading fails""",
            category="filesystem",
            path="filesystem",
        )),
        ("system/time", FunctionDef(
            name="get_time",
            func=get_time_func,
            signature="get_time() -> str",
            description="""Get the current date and time.

Returns:
    Current date and time in format 'YYYY-MM-DD HH:MM:SS'""",
            category="system",
            path="system/time",
        )),
        ("filesystem", FunctionDef(
            name="list_files",
            func=list_files_func,
            signature="list_files(directory: str) -> list[str]",
            description="""List all files in a directory.

Args:
    directory: Path to the directory to list

Returns:
    List of filenames in the directory, or error message if listing fails""",
            category="filesystem",
            path="filesystem",
        )),
        ("system", FunctionDef(
            name="find_functions",
            func=find_functions_func,
            signature="find_functions(in_dir: str = '', name: str = '') -> str",
            description="""Search for functions in the function tree.

Args:
    in_dir: Directory path to search (e.g., "filesystem", "system/time", "" for root)
    name: Function name to search for (supports fuzzy matching)

Returns:
    Formatted string showing matching functions""",
            category="system",
            path="system",
        )),
    ]

    # Add all functions to tree and cache
    for path, func_def in function_defs:
        state.function_tree.add_function(func_def, path)
        state.function_cache.add(func_def)
