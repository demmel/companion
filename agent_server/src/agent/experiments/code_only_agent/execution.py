"""Code parsing and execution for code-only agent."""

import re
import io
import sys
from typing import Callable, Any
from dataclasses import dataclass
from abc import ABC


class OutputMessage(ABC):
    """Base class for execution output messages."""

    pass


@dataclass
class SpeakMessage(OutputMessage):
    """Message sent to user via speak()."""

    content: str


@dataclass
class StdoutMessage(OutputMessage):
    """Standard output from code execution."""

    content: str


@dataclass
class ErrorMessage(OutputMessage):
    """Error from code execution."""

    error: str


class ExecutionState:
    """Execution state for a turn (persists across iterations)."""

    def __init__(self):
        self.exec_globals: dict[str, Any] = {}
        self.current_iteration_messages: list[OutputMessage] = []

    def reset_iteration_messages(self) -> None:
        """Clear messages for a new iteration while preserving exec_globals."""
        self.current_iteration_messages = []


def format_output_message(output: OutputMessage) -> str:
    """Format an output message for display."""
    if isinstance(output, SpeakMessage):
        return f"[SPEAK] {output.content}"
    elif isinstance(output, StdoutMessage):
        return f"[STDOUT] {output.content}"
    elif isinstance(output, ErrorMessage):
        return f"[ERROR] {output.error}"
    return str(output)


class MessageCollector(io.StringIO):
    """Custom stdout that collects StdoutMessages in chronological order."""

    def __init__(self, exec_state: ExecutionState):
        super().__init__()
        self.exec_state = exec_state
        self._buffer = ""

    def write(self, text: str) -> int:
        # Accumulate text in buffer
        self._buffer += text

        # If we have a complete line (ends with newline), flush it
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            # Process all complete lines
            for line in lines[:-1]:
                if line:  # Don't add empty lines
                    self.exec_state.current_iteration_messages.append(
                        StdoutMessage(content=line)
                    )
            # Keep the incomplete line in buffer
            self._buffer = lines[-1]

        return super().write(text)

    def flush_remaining(self):
        """Flush any remaining buffer content."""
        if self._buffer:
            self.exec_state.current_iteration_messages.append(
                StdoutMessage(content=self._buffer)
            )
            self._buffer = ""


def get_safe_builtins() -> dict[str, Callable]:
    """
    Get a curated set of safe Python builtins.

    Excludes dangerous functions like eval, exec, __import__, open, etc.
    Includes basic data manipulation and type conversion functions.
    """
    safe_builtins = {
        # Type functions
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        # Collection operations
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "reversed": reversed,
        "sum": sum,
        "min": min,
        "max": max,
        "any": any,
        "all": all,
        # String operations
        "chr": chr,
        "ord": ord,
        # Functional
        "map": map,
        "filter": filter,
        # Debugging/inspection
        "type": type,
        "isinstance": isinstance,
        # Iteration
        "iter": iter,
        "next": next,
        # Other safe utilities
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
    }
    return safe_builtins


def parse_code_blocks(text: str) -> tuple[str, list[str]]:
    """
    Parse LLM response into reasoning and code blocks.

    Returns:
        (reasoning, code_blocks) where reasoning is all non-code text,
        and code_blocks is a list of code extracted from ```python blocks
    """
    # Find all python code blocks
    pattern = r"```python\n(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)

    # Remove code blocks to get reasoning
    reasoning = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return reasoning, code_blocks


def execute_code(
    code: str,
    exec_state: ExecutionState,
) -> None:
    """
    Execute code in a restricted environment and collect output messages.

    Args:
        code: The Python code to execute
        exec_state: ExecutionState containing exec_globals and message collection
    """
    # Capture stdout with MessageCollector
    old_stdout = sys.stdout
    sys.stdout = MessageCollector(exec_state)

    try:
        exec(code, exec_state.exec_globals, exec_state.exec_globals)
        # Flush any remaining buffered output
        sys.stdout.flush_remaining()
    except Exception as e:
        exec_state.current_iteration_messages.append(
            ErrorMessage(error=f"{type(e).__name__}: {e}")
        )
    finally:
        sys.stdout = old_stdout
