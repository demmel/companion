"""Base function implementations for the agent."""

import os
from datetime import datetime
from typing import TYPE_CHECKING

from agent.experiments.code_only_agent.functions.definitions import FunctionDef

if TYPE_CHECKING:
    from agent.experiments.code_only_agent.state import State


def create_speak_function(state: "State"):
    """Create a speak function that captures messages in state."""

    def speak(message: str):
        """Send a message to the user."""
        state.speak_messages.append(message)
        print(f"[SPEAK] {message}")

    return speak


def create_read_file_function():
    """Create a read_file function."""

    def read_file(path: str) -> str:
        """Read contents of a file at the given path."""
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"

    return read_file


def create_get_time_function():
    """Create a get_time function."""

    def get_time() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return get_time


def create_list_files_function():
    """Create a list_files function."""

    def list_files(directory: str) -> list[str]:
        """List all files in a directory."""
        try:
            return os.listdir(directory)
        except Exception as e:
            return [f"Error listing directory: {type(e).__name__}: {e}"]

    return list_files


def initialize_base_functions(state: "State") -> None:
    """Initialize base functions in the function tree and cache."""
    # Create functions
    speak_func = create_speak_function(state)
    read_file_func = create_read_file_function()
    get_time_func = create_get_time_function()
    list_files_func = create_list_files_function()

    # Define function metadata
    functions = [
        FunctionDef(
            name="speak",
            func=speak_func,
            signature="speak(message: str)",
            description="Send a message to the user",
            category="communication",
        ),
        FunctionDef(
            name="read_file",
            func=read_file_func,
            signature="read_file(path: str) -> str",
            description="Read contents of a file at the given path",
            category="filesystem",
        ),
        FunctionDef(
            name="get_time",
            func=get_time_func,
            signature="get_time() -> str",
            description="Get the current date and time",
            category="system",
        ),
        FunctionDef(
            name="list_files",
            func=list_files_func,
            signature="list_files(directory: str) -> list[str]",
            description="List all files in a directory",
            category="filesystem",
        ),
    ]

    # Add all functions to tree and cache
    for func_def in functions:
        state.function_tree.add_function(func_def)
        state.function_cache.add(func_def)
