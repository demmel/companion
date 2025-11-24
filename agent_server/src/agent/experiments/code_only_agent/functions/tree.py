"""Function tree for organizing available functions."""

from typing import Optional
from dataclasses import dataclass, field

from agent.experiments.code_only_agent.functions.definitions import FunctionDef


@dataclass
class TreeNode:
    """A node in the function tree."""

    name: str
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    functions: list[FunctionDef] = field(default_factory=list)

    def add_child(self, name: str) -> "TreeNode":
        """Add or get a child node."""
        if name not in self.children:
            self.children[name] = TreeNode(name=name)
        return self.children[name]

    def get_child(self, name: str) -> Optional["TreeNode"]:
        """Get a child node by name."""
        return self.children.get(name)

    def add_function(self, func_def: FunctionDef):
        """Add a function to this node."""
        self.functions.append(func_def)


class FunctionTree:
    """Hierarchical tree of available functions organized by path."""

    def __init__(self):
        self.root = TreeNode(name="")

    def add_function(self, func_def: FunctionDef, path: str = ""):
        """
        Add a function to the tree at the specified path.

        Args:
            func_def: Function definition to add
            path: Path in tree (e.g., "filesystem/read", "communication")
                  Empty string adds to root.
        """
        # Navigate to the target node, creating intermediate nodes as needed
        node = self.root
        if path:
            parts = path.split("/")
            for part in parts:
                node = node.add_child(part)

        # Add function to the target node
        node.add_function(func_def)

    def get_function(self, name: str) -> Optional[FunctionDef]:
        """Get a function by name from anywhere in the tree."""
        return self._search_function_recursive(self.root, name)

    def _search_function_recursive(
        self, node: TreeNode, name: str
    ) -> Optional[FunctionDef]:
        """Recursively search for a function by name."""
        # Check functions in this node
        for func_def in node.functions:
            if func_def.name == name:
                return func_def

        # Search children
        for child in node.children.values():
            result = self._search_function_recursive(child, name)
            if result:
                return result

        return None

    def find_functions(self, in_dir: str, name: str) -> list[FunctionDef]:
        """
        Find functions matching the name in the specified directory.

        Args:
            in_dir: Directory path to search (e.g., "filesystem", "filesystem/read", "" for root)
            name: Function name to search for (supports fuzzy matching)

        Returns:
            List of matching FunctionDef objects
        """
        # Navigate to the target directory
        node = self.root
        if in_dir:
            parts = in_dir.split("/")
            for part in parts:
                node = node.get_child(part)
                if node is None:
                    return []  # Directory doesn't exist

        # Search recursively from this node
        results = []
        self._find_functions_recursive(node, name, results)
        return results

    def _find_functions_recursive(
        self, node: TreeNode, name: str, results: list[FunctionDef]
    ):
        """Recursively find functions matching the name."""
        name_lower = name.lower()

        # Check functions in this node
        for func_def in node.functions:
            # Exact match
            if func_def.name == name:
                results.append(func_def)
            # Fuzzy match: substring of name or description
            elif name_lower in func_def.name.lower() or name_lower in func_def.description.lower():
                results.append(func_def)

        # Search children
        for child in node.children.values():
            self._find_functions_recursive(child, name, results)

    def list_all_functions(self) -> list[FunctionDef]:
        """Get all functions as a flat list."""
        result = []
        self._collect_functions_recursive(self.root, result)
        return result

    def _collect_functions_recursive(self, node: TreeNode, result: list[FunctionDef]):
        """Recursively collect all functions."""
        result.extend(node.functions)
        for child in node.children.values():
            self._collect_functions_recursive(child, result)

    def count_functions_at_path(self, path: str) -> int:
        """
        Count total functions at and under the given path.

        Args:
            path: Directory path (e.g., "filesystem", "system/time", "" for root)

        Returns:
            Number of functions at and under this path
        """
        node = self.root
        if path:
            parts = path.split("/")
            for part in parts:
                node = node.get_child(part)
                if node is None:
                    return 0

        return self._count_functions_recursive(node)

    def _count_functions_recursive(self, node: TreeNode) -> int:
        """Recursively count all functions under a node."""
        count = len(node.functions)
        for child in node.children.values():
            count += self._count_functions_recursive(child)
        return count

    def get_child_paths(self, path: str) -> list[str]:
        """
        Get all child directory names at the given path.

        Args:
            path: Directory path (e.g., "filesystem", "" for root)

        Returns:
            List of child directory names
        """
        node = self.root
        if path:
            parts = path.split("/")
            for part in parts:
                node = node.get_child(part)
                if node is None:
                    return []

        return list(node.children.keys())

    def format_as_tree(self, node: Optional[TreeNode] = None, indent: int = 0) -> str:
        """Format the function tree for display to the agent."""
        if node is None:
            node = self.root

        lines = []

        # Show functions in this node
        for func_def in sorted(node.functions, key=lambda f: f.name):
            lines.append("  " * indent + f"{func_def.signature}")
            lines.append("  " * indent + f"  {func_def.description}")

        # Show children
        for child_name, child_node in sorted(node.children.items()):
            lines.append("  " * indent + f"{child_name}/")
            lines.append(self.format_as_tree(child_node, indent + 1))

        return "\n".join(lines) if lines else "(empty)"
