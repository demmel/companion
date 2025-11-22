"""LRU cache for functions currently loaded for the agent."""

from collections import OrderedDict
from typing import Optional, Callable

from agent.experiments.code_only_agent.functions.definitions import FunctionDef


class FunctionCache:
    """LRU cache for functions the agent currently has loaded."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.cache: OrderedDict[str, FunctionDef] = OrderedDict()

    def add(self, func_def: FunctionDef):
        """Add a function to the cache."""
        if func_def.name in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(func_def.name)
        else:
            self.cache[func_def.name] = func_def
            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def get(self, name: str) -> Optional[FunctionDef]:
        """Get a function from cache."""
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        return None

    def get_all_funcs(self) -> dict[str, Callable]:
        """Get all cached functions as a dict for execution."""
        return {name: func_def.func for name, func_def in self.cache.items()}

    def format_for_prompt(self) -> str:
        """Format cached functions for the LLM prompt."""
        if not self.cache:
            return "(no functions loaded)"

        lines = []
        for func_def in self.cache.values():
            lines.append(f"{func_def.signature}")
            lines.append(f"  {func_def.description}")
        return "\n".join(lines)
