"""Base function implementations for the agent."""

import os
from datetime import datetime

from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.experiments.code_only_agent.functions.definitions import FunctionDef
from agent.experiments.code_only_agent.execution import ExecutionState, SpeakMessage
from agent.experiments.code_only_agent.state import State
from agent.state import Priority


def create_speak_function(exec_state: ExecutionState):
    """Create a speak function that appends to the execution state."""

    def speak(message: str) -> str:
        """
        Send a message to the user.

        Args:
            message: The message to send to the user

        Returns:
            Confirmation that message was sent
        """
        exec_state.current_iteration_messages.append(SpeakMessage(content=message))
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


def create_find_functions_function(state: State):
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


def create_priority_functions(state: State):
    """Create priority management functions."""

    def add_priority(priority: str) -> str:
        """
        Add a new priority to my current priorities.

        Args:
            priority: Description of the new priority

        Returns:
            Confirmation message with the priority ID
        """
        priority_id = f"p{state.next_priority_id}"
        state.current_priorities.append(Priority(id=priority_id, content=priority))
        state.next_priority_id += 1
        return f"Added priority {priority_id}: {priority}"

    def remove_priority(priority_id: str) -> str:
        """
        Remove a priority from my current priorities.

        Args:
            priority_id: ID of the priority to remove (e.g., 'p1', 'p2')

        Returns:
            Confirmation message or error if not found
        """
        for i, p in enumerate(state.current_priorities):
            if p.id == priority_id:
                removed = state.current_priorities.pop(i)
                return f"Removed priority {priority_id}: {removed.content}"
        return f"Priority {priority_id} not found"

    def list_priorities() -> str:
        """
        List my current priorities.

        Returns:
            Formatted list of current priorities
        """
        if not state.current_priorities:
            return "No current priorities"
        lines = []
        for p in state.current_priorities:
            lines.append(f"{p.id}: {p.content}")
        return "\n".join(lines)

    return add_priority, remove_priority, list_priorities


def create_call_llm_function(llm: LLM, model: SupportedModel):
    """Create a call_llm function."""

    def call_llm(prompt: str) -> str:
        """
        Call an LLM with a prompt and get a response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response text
        """
        try:
            response = llm.generate_complete(
                model=model, prompt=prompt, caller="code_agent_user_call"
            )
            return response
        except Exception as e:
            return f"Error calling LLM: {type(e).__name__}: {e}"

    return call_llm


def initialize_base_functions(
    state: State, llm: LLM, model: SupportedModel, exec_state: ExecutionState
) -> None:
    """Initialize base functions in the function tree and cache."""
    # Create speak function with reference to ExecutionState
    speak_func = create_speak_function(exec_state)

    # Create other functions
    read_file_func = create_read_file_function()
    get_time_func = create_get_time_function()
    list_files_func = create_list_files_function()
    find_functions_func = create_find_functions_function(state)
    add_priority_func, remove_priority_func, list_priorities_func = (
        create_priority_functions(state)
    )
    call_llm_func = create_call_llm_function(llm, model)

    # Add speak to exec_globals
    exec_state.exec_globals["speak"] = speak_func

    # Define function metadata with paths
    function_defs = [
        (
            "communication",
            FunctionDef(
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
            ),
        ),
        (
            "filesystem",
            FunctionDef(
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
            ),
        ),
        (
            "system/time",
            FunctionDef(
                name="get_time",
                func=get_time_func,
                signature="get_time() -> str",
                description="""Get the current date and time.

Returns:
    Current date and time in format 'YYYY-MM-DD HH:MM:SS'""",
                category="system",
                path="system/time",
            ),
        ),
        (
            "filesystem",
            FunctionDef(
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
            ),
        ),
        (
            "system",
            FunctionDef(
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
            ),
        ),
        (
            "personality",
            FunctionDef(
                name="add_priority",
                func=add_priority_func,
                signature="add_priority(priority: str) -> str",
                description="""Add a new priority to my current priorities.

Args:
    priority: Description of the new priority

Returns:
    Confirmation message with the priority ID""",
                category="personality",
                path="personality",
            ),
        ),
        (
            "personality",
            FunctionDef(
                name="remove_priority",
                func=remove_priority_func,
                signature="remove_priority(priority_id: str) -> str",
                description="""Remove a priority from my current priorities.

Args:
    priority_id: ID of the priority to remove (e.g., 'p1', 'p2')

Returns:
    Confirmation message or error if not found""",
                category="personality",
                path="personality",
            ),
        ),
        (
            "personality",
            FunctionDef(
                name="list_priorities",
                func=list_priorities_func,
                signature="list_priorities() -> str",
                description="""List my current priorities.

Returns:
    Formatted list of current priorities""",
                category="personality",
                path="personality",
            ),
        ),
        (
            "llm",
            FunctionDef(
                name="call_llm",
                func=call_llm_func,
                signature="call_llm(prompt: str) -> str",
                description="""Call an LLM with a prompt and get a response.

Args:
    prompt: The prompt to send to the LLM

Returns:
    The LLM's response text""",
                category="llm",
                path="llm",
            ),
        ),
    ]

    # Add all functions to tree and cache
    for path, func_def in function_defs:
        state.function_tree.add_function(func_def, path)
        state.function_cache.add(func_def)
