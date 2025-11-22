"""Function tree for organizing available functions."""

from typing import Optional

from agent.experiments.code_only_agent.functions.definitions import FunctionDef


class FunctionTree:
    """Tree structure of available functions organized by category."""

    def __init__(self):
        self.functions: dict[str, dict[str, FunctionDef]] = {}

    def add_function(self, func_def: FunctionDef):
        """Add a function to the tree under its category."""
        if func_def.category not in self.functions:
            self.functions[func_def.category] = {}
        self.functions[func_def.category][func_def.name] = func_def

    def get_function(self, name: str) -> Optional[FunctionDef]:
        """Get a function by name from any category."""
        for category_funcs in self.functions.values():
            if name in category_funcs:
                return category_funcs[name]
        return None

    def list_all_functions(self) -> list[FunctionDef]:
        """Get all functions as a flat list."""
        result = []
        for category_funcs in self.functions.values():
            result.extend(category_funcs.values())
        return result

    def format_as_tree(self) -> str:
        """Format the function tree for display to the agent."""
        if not self.functions:
            return "(empty)"

        lines = []
        for category, funcs in sorted(self.functions.items()):
            lines.append(f"{category}/")
            for name, func_def in sorted(funcs.items()):
                lines.append(f"  {func_def.signature}")
                lines.append(f"    {func_def.description}")
        return "\n".join(lines)
