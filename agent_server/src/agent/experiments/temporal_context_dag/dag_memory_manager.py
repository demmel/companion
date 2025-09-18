"""
DAG Memory Manager - Standalone memory management component for DAG-based context system.

This manager encapsulates all DAG memory functionality and provides a clean interface
for memory formation, context management, and persistence without requiring integration
with the main agent system.
"""

import logging

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.llm import LLM, SupportedModel
from agent.state import State
from pydantic import BaseModel

from .models import ContextElement, ContextGraph, MemoryGraph
from .experiment_helpers import (
    calculate_context_budget,
    create_initial_context_subgraph,
    create_initial_graph,
)
from .io import save_dag_to_json

logger = logging.getLogger(__name__)


class ContextElementData(BaseModel):
    memory_id: str
    tokens: int


class ContextGraphData(BaseModel):
    elements: list[ContextElementData]
    edges: list[int]


class DagMemoryData(BaseModel):
    memory: MemoryGraph
    context: ContextGraphData


class DagMemoryManager:
    """
    Standalone DAG memory management system.

    Manages memory formation, connections, and context extraction using the DAG-based
    approach. Can be used independently or integrated into an agent system.
    """

    def __init__(self, memory_graph: MemoryGraph, context_graph: ContextGraph):
        """
        Initialize the DAG memory manager.

        Args:
            llm: LLM instance for memory extraction and connections
            model: Model to use for memory operations
            token_budget: Total token budget for context management
        """
        self.memory_graph = memory_graph
        self.context_graph = context_graph

    @classmethod
    def create(
        cls, initial_state: State, backstory: str, token_budget: int
    ) -> "DagMemoryManager":
        """
        Initialize the memory system from an initial trigger (character definition).

        Args:
            initial_trigger: The initial character definition trigger
        """
        # Derive initial state and create initial graph
        memory_graph = create_initial_graph(initial_state, backstory)

        # Create initial context subgraph
        context_budget = calculate_context_budget(token_budget, initial_state)
        context_graph = create_initial_context_subgraph(memory_graph, context_budget)

        return cls(memory_graph, context_graph)

    def process_trigger(
        self,
        trigger: TriggerHistoryEntry,
        state: State,
        llm: LLM,
        model: SupportedModel,
        token_budget: int,
        update_state: bool = False,
    ) -> None:
        """
        Process a trigger and update the memory graph and context.

        Args:
            trigger: The trigger history entry to process
            current_state: Current agent state (used for context, may be mutated)
            update_state: Whether to update the manager's internal state (for retroactive building)
        """

        # Use the existing process_trigger function from experiment.py
        from .experiment import process_trigger

        self.memory_graph, self.context_graph, self.state = process_trigger(
            graph=self.memory_graph,
            context=self.context_graph,
            state=state,
            trigger=trigger,
            llm=llm,
            model=model,
            token_budget=token_budget,
            update_state=update_state,
        )

    def get_current_context(self) -> ContextGraph:
        """
        Get the current context graph (working memory).

        Returns:
            The current context graph containing relevant memories within token budget
        """
        return self.context_graph

    def get_memory_graph(self) -> MemoryGraph:
        """
        Get the complete memory graph.

        Returns:
            The complete memory graph containing all memories and connections
        """
        return self.memory_graph

    def save_to_file(self, filepath: str) -> None:
        """
        Save the current memory graph to a JSON file.

        Args:
            filepath: Path to save the memory graph
        """

        data = DagMemoryData(
            memory=self.memory_graph,
            context=ContextGraphData(
                elements=[
                    ContextElementData(
                        memory_id=elem.memory.id,
                        tokens=elem.tokens,
                    )
                    for elem in self.context_graph.elements
                ],
                edges=[
                    self.memory_graph.edges.index(edge)
                    for edge in self.context_graph.edges
                ],
            ),
        )
        with open(filepath, "w") as f:
            f.write(data.model_dump_json(indent=2))

    def load_from_file(self, filepath: str) -> None:
        """
        Load a memory graph from a JSON file.

        Args:
            filepath: Path to load the memory graph from
        """

        with open(filepath, "r") as f:
            data = DagMemoryData.model_validate_json(f.read())
        self.memory_graph = data.memory
        self.context_graph = ContextGraph(
            elements=[
                ContextElement(
                    memory=self.memory_graph.elements[elem.memory_id],
                    tokens=elem.tokens,
                )
                for elem in data.context.elements
            ],
            edges=[
                self.memory_graph.edges[edge_idx] for edge_idx in data.context.edges
            ],
        )
