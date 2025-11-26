"""LRU cache for functions currently loaded for the agent."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Callable

from agent.experiments.code_only_agent.functions.definitions import FunctionDef
from agent.experiments.code_only_agent.functions.tree import FunctionTree


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

    def format_for_prompt(self, state) -> str:
        """
        Format cached functions as a tree using Unix tree-style box-drawing chars.

        Shows cached functions expanded, uncached paths with "(X more)".

        Args:
            state: State object to access function_tree for counting uncached functions
        """
        if not self.cache:
            return "(no functions loaded)"

        # Build tree structure from cached functions
        cached_names = set(self.cache.keys())

        # Group cached functions by their paths
        tree_structure = {}  # path -> list of FunctionDef
        for func_def in self.cache.values():
            path = func_def.path or ""
            if path not in tree_structure:
                tree_structure[path] = []
            tree_structure[path].append(func_def)

        # Build output using tree format
        lines = ["."]
        self._format_tree_recursive(
            state.function_tree,
            "",
            tree_structure,
            cached_names,
            lines,
            prefix="",
            is_last_list=[],
        )

        return "\n".join(lines)

    def _format_tree_recursive(
        self,
        tree: FunctionTree,
        current_path: str,
        tree_structure: dict[str, list[FunctionDef]],
        cached_names: set[str],
        lines: list[str],
        prefix: str,
        is_last_list: list[bool],
    ) -> None:
        """Recursively format tree with box-drawing characters."""
        # Get children at this path
        children = tree.get_child_paths(current_path)

        # Get functions at this path
        functions_here = tree_structure.get(current_path, [])

        # Calculate total items (functions + child dirs)
        total_items = len(functions_here) + len(children)

        # Display functions at this level
        for i, func_def in enumerate(sorted(functions_here, key=lambda f: f.name)):
            is_last = (i == len(functions_here) - 1) and len(children) == 0

            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{func_def.signature}")

            # Add description with proper indentation for multi-line descriptions
            desc_prefix = prefix + ("    " if is_last else "│   ")
            desc_lines = func_def.description.split("\n")
            for desc_line in desc_lines:
                lines.append(f"{desc_prefix}  {desc_line}")

        # Display child directories
        for i, child_name in enumerate(sorted(children)):
            is_last_child = i == len(children) - 1
            connector = "└── " if is_last_child else "├── "

            child_path = f"{current_path}/{child_name}" if current_path else child_name

            # Count uncached functions in this subtree
            total_funcs = tree.count_functions_at_path(child_path)
            cached_in_subtree = sum(
                1
                for name in cached_names
                if any(
                    f.path.startswith(child_path)
                    for f in self.cache.values()
                    if f.name == name
                )
            )
            uncached_count = total_funcs - cached_in_subtree

            # Show directory with uncached count if applicable
            dir_label = f"{child_name}/"
            if uncached_count > 0:
                dir_label += f" ({uncached_count} more)"

            lines.append(f"{prefix}{connector}{dir_label}")

            # Recurse into child
            new_prefix = prefix + ("    " if is_last_child else "│   ")
            self._format_tree_recursive(
                tree,
                child_path,
                tree_structure,
                cached_names,
                lines,
                new_prefix,
                is_last_list + [is_last_child],
            )
