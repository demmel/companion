"""Code parsing and execution for code-only agent."""

import re
import io
import sys
from typing import Callable, Any


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


def execute_code(code: str, available_functions: dict[str, Callable]) -> str:
    """
    Execute code in a restricted environment.

    Args:
        code: The Python code to execute
        available_functions: Dict of functions to make available in the execution environment

    Returns the captured output as a string.
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        # Create execution environment with safe builtins and provided functions
        exec_globals: dict[str, Any] = {"__builtins__": get_safe_builtins()}
        if available_functions:
            exec_globals.update(available_functions)

        exec(code, exec_globals, {})
        output = captured_output.getvalue()
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
