"""Function definition dataclass."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class FunctionDef:
    """Definition of a function available to the agent."""

    name: str
    func: Callable
    signature: str
    description: str
    category: str
    path: str = ""  # Path in function tree (e.g., "filesystem", "system/time")
