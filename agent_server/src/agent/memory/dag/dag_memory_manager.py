"""
Action-based DAG Memory Manager that uses the reducer+action pattern for observability.

This manager emits actions instead of directly mutating state, enabling time-travel
debugging and complete replay of memory graph evolution.
"""

import logging
from agent.memory.dag.context_formatting import format_context
from agent.memory.memory import IMemory, MemoryQueries, RetrievedMemories
from agent.timeit import timeit
from typing import Sequence

from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage import ITriggerHistory
from agent.memory.dag.memory_formation import (
    extract_memories_as_actions,
)
from agent.llm import LLM, SupportedModel
from agent.state import State
from pydantic import BaseModel

from .models import ContextElement, ContextGraph, MemoryGraph
from .actions import (
    AddToContextAction,
    ApplyTokenDecayAction,
    MemoryAction,
    CheckpointAction,
)
from .action_log import MemoryActionLog
from .context_management import (
    prune_context_to_budget_as_actions,
)
from .reducer import apply_action

logger = logging.getLogger(__name__)


class ContextElementData(BaseModel):
    memory_id: str
    tokens: int


class ContextGraphData(BaseModel):
    elements: list[ContextElementData]
    edges: list[str]  # Edge IDs instead of indices


class DagMemoryData(BaseModel):
    memory: MemoryGraph
    context: ContextGraphData


class DagMemoryManager(IMemory):
    """
    Action-based DAG memory management system with full observability.

    Uses reducer+action pattern where all state changes are recorded as actions
    that can be replayed to reconstruct any historical state.
    """

    def __init__(
        self,
        memory_graph: MemoryGraph,
        context_graph: ContextGraph,
        trigger_history: ITriggerHistory,
        use_individual_formatting: bool,
    ):
        """Initialize with existing graph state and empty action log."""
        self.memory_graph = memory_graph
        self.context_graph = context_graph
        self.trigger_history = trigger_history
        self.action_log = MemoryActionLog()
        self.use_individual_formatting = use_individual_formatting

    @classmethod
    def create(
        cls,
        trigger_history: ITriggerHistory,
        use_individual_formatting: bool,
    ) -> "DagMemoryManager":
        """
        Create a new manager with empty state. Memories will be added via postprocess_trigger.

        Args:
            trigger_history: Trigger history for replay functionality
            use_individual_formatting: Whether to use individual memory formatting
        """
        # Start with completely empty state - memories will be added via postprocess_trigger
        manager = cls(
            MemoryGraph(),
            ContextGraph(elements=[], edges=[]),
            trigger_history,
            use_individual_formatting,
        )

        # Record that we're starting with empty state
        manager.action_log.add_checkpoint(
            label="creation_start",
            description="Starting with empty memory graph - initial exchange will be processed via postprocess_trigger",
        )

        return manager

    def query(
        self,
        memory_queries: MemoryQueries,
        llm: LLM,
        model: SupportedModel,
    ) -> RetrievedMemories:
        """
        Retrieve memories relevant to the given queries.

        Pure read: scores/expands candidates and returns the retrieved memories (formatted
        for prompts, plus the context actions retrieval produced). Does NOT mutate the
        working context — folding the recall into context is a separate, deliberate step
        (`reinforce`). Returns empty `RetrievedMemories` if nothing matched.
        """
        from .retrieval_integration import retrieve_relevant_memories_as_actions

        with timeit("Memory Retrieval"):
            retrieval_actions = retrieve_relevant_memories_as_actions(
                queries=memory_queries.queries,
                memory_graph=self.memory_graph,
                context_graph=self.context_graph,
                max_retrieved_memories=5,
                min_similarity_threshold=0.4,
            )

        if not retrieval_actions:
            return RetrievedMemories()

        elements = [
            self.memory_graph.elements[action.memory_id]
            for action in retrieval_actions
            if isinstance(action, AddToContextAction)
            and action.memory_id in self.memory_graph.elements
        ]
        return RetrievedMemories(elements=elements, actions=list(retrieval_actions))

    def reinforce(
        self,
        retrieved: RetrievedMemories,
        budget: int,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """
        Fold a recall's results into the persistent working context, then prune to budget.

        The deliberate mutation half of recall: dispatches the retrieval actions (recorded in
        the memory action log) and prunes so a recall can't push the context over budget.
        """
        if not retrieved.actions:
            return

        self.dispatch_actions(retrieved.actions)
        self.add_checkpoint(
            label="memories_reinforced",
            description=f"Reinforced {len(retrieved.actions)} retrieved memory actions into context",
        )
        self._prune_to_budget(budget, llm, model)

    def prune(
        self,
        budget: int,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """
        Per-turn context maintenance: apply token decay, then prune to budget.

        Runs the decay + prune housekeeping that used to live in the now-removed
        preprocess step, with no retrieval. Called once per trigger before reasoning so
        the working context stays within budget even when the agent never recalls.
        """
        logger.info("Pruning working context (decay + prune to budget)")
        self.add_checkpoint(
            label="prune_start",
            description="Starting per-turn context maintenance (decay + prune)",
        )

        # STEP 1: Apply token decay to existing context memories
        self._apply_token_decay()

        # STEP 2: Prune context to budget
        self._prune_to_budget(budget, llm, model)

        logger.info(
            f"Prune complete - Context: {len(self.context_graph.elements)} elements, "
            f"{len(self.context_graph.edges)} edges"
        )

    def _prune_to_budget(
        self,
        budget: int,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """Prune the working context to fit within the given token budget."""
        with timeit("Context Pruning"):
            pruning_actions = prune_context_to_budget_as_actions(
                graph=self.memory_graph,
                context=self.context_graph,
                budget=budget,
                use_individual_formatting=self.use_individual_formatting,
                llm=llm,
                model=model,
            )

        if pruning_actions:
            self.dispatch_actions(pruning_actions)
            self.add_checkpoint(
                label="context_pruned",
                description=f"Pruned context to fit budget of {budget} tokens",
            )

    def get_formatted_context(self) -> str:
        """
        Return the current working context formatted for prompts.

        Pure read: no retrieval, no decay, no mutation. Used by the reasoning loop to feed
        the already-accumulated context into situational analysis and as the per-turn
        return value.
        """
        return format_context(
            self.context_graph, self.memory_graph, self.use_individual_formatting
        )

    def store(
        self,
        trigger_history_entry: TriggerHistoryEntry,
        state: State,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        self.postprocess_trigger(
            trigger=trigger_history_entry,
            state=state,
            llm=llm,
            model=model,
        )

    def dispatch_actions(self, actions: Sequence[MemoryAction]) -> None:
        """
        Dispatch a list of actions to update the memory graph and context.

        Args:
            actions: List of actions to apply
        """
        for action in actions:
            self.action_log.add_action(action)
            apply_action(
                self.trigger_history, self.memory_graph, self.context_graph, action
            )

    def add_checkpoint(self, label: str, description: str) -> CheckpointAction:
        """Add a checkpoint to the action log."""
        return self.action_log.add_checkpoint(label, description)

    def postprocess_trigger(
        self,
        trigger: TriggerHistoryEntry,
        state: State,
        llm: LLM,
        model: SupportedModel,
    ) -> None:
        """
        Postprocess trigger by extracting memories from the completed reasoning.

        This is called AFTER the reasoning loop completes to extract and store
        new memories from the agent's reasoning and actions.

        Args:
            trigger: The completed trigger history entry
            state: Current agent state
            llm: LLM instance for memory operations
            model: Model to use for decisions
            token_budget: Token budget for context management
            action_registry: Action registry for budget calculation
        """
        logger.info(f"Processing trigger {trigger.entry_id} with action-based approach")

        # Checkpoint: Start of trigger processing
        self.add_checkpoint(
            label=f"trigger_start_{trigger.entry_id}",
            description=f"Starting processing of trigger {trigger.entry_id}",
        )

        # Extract memories and connections as actions
        with timeit("Memory Extraction"):
            memory_actions = extract_memories_as_actions(
                trigger, state, self.context_graph, llm, model, self.memory_graph
            )

        if memory_actions:
            # Dispatch memory and connection actions
            self.dispatch_actions(memory_actions)

            # Checkpoint: Memories extracted
            self.add_checkpoint(
                label=f"memories_extracted_{trigger.entry_id}",
                description=f"Extracted memories and connections for {trigger.entry_id}",
            )

            # Checkpoint: Trigger processing complete
            self.add_checkpoint(
                label=f"trigger_complete_{trigger.entry_id}",
                description=f"Completed processing trigger {trigger.entry_id}",
            )

            logger.info(
                f"Completed trigger {trigger.entry_id} - "
                f"Graph: {len(self.memory_graph.elements)} memories, "
                f"Context: {len(self.context_graph.elements)} elements"
            )
        else:
            logger.info(f"No significant memories extracted from {trigger.entry_id}")

    def get_current_context(self) -> ContextGraph:
        """Get the current context graph."""
        return self.context_graph

    def get_memory_graph(self) -> MemoryGraph:
        """Get the complete memory graph."""
        return self.memory_graph

    def get_action_log(self) -> MemoryActionLog:
        """Get the action log for replay and analysis."""
        return self.action_log

    def replay_to_checkpoint(self, checkpoint_label: str) -> "DagMemoryManager":
        """
        Create a new manager instance by replaying actions up to a checkpoint.

        Args:
            checkpoint_label: Label of the checkpoint to replay to

        Returns:
            New manager instance with state at the specified checkpoint
        """
        # Replay actions to get graph state at checkpoint
        memory_graph, context_graph = self.action_log.replay_to_checkpoint(
            self.trigger_history, checkpoint_label
        )

        # Create new manager with replayed state
        new_manager = DagMemoryManager(
            memory_graph,
            context_graph,
            self.trigger_history,
            self.use_individual_formatting,
        )

        # Copy the action log up to the checkpoint
        checkpoint_idx = self.action_log.find_checkpoint_index(checkpoint_label)
        if checkpoint_idx is not None:
            new_manager.action_log.actions = self.action_log.actions[
                : checkpoint_idx + 1
            ]

        return new_manager

    def to_data(self) -> DagMemoryData:
        """
        Serialize the current memory and context graphs to a data object.

        Returns:
            A DagMemoryData object containing the memory and context graphs
        """

        return DagMemoryData(
            memory=self.memory_graph,
            context=ContextGraphData(
                elements=[
                    ContextElementData(
                        memory_id=elem.memory.id,
                        tokens=elem.tokens,
                    )
                    for elem in self.context_graph.elements
                ],
                edges=[edge.id for edge in self.context_graph.edges],
            ),
        )

    @classmethod
    def from_data(
        cls,
        data: DagMemoryData,
        trigger_history: ITriggerHistory,
        use_individual_formatting: bool,
    ) -> "DagMemoryManager":
        """
        Create a DagMemoryManager from a serialized data object.

        Args:
            data: A DagMemoryData object containing the memory and context graphs
            trigger_history: Trigger history for replay functionality
            use_individual_formatting: Whether to use individual memory formatting
        Returns:
            A DagMemoryManager instance initialized with the provided data
        """

        memory_graph = data.memory

        context_graph = ContextGraph(
            elements=[
                ContextElement(
                    memory=memory_graph.elements[elem.memory_id],
                    tokens=elem.tokens,
                )
                for elem in data.context.elements
            ],
            edges=[memory_graph.edges[edge_id] for edge_id in data.context.edges],
        )
        return cls(
            memory_graph, context_graph, trigger_history, use_individual_formatting
        )

    def save_to_file(self, filepath: str) -> None:
        """
        Save the current memory graph to a JSON file.

        Args:
            filepath: Path to save the memory graph
        """

        data = self.to_data()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2))

    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        trigger_history: ITriggerHistory,
        use_individual_formatting: bool,
    ) -> "DagMemoryManager":
        """
        Load a memory graph from a JSON file.

        Args:
            filepath: Path to load the memory graph from
            trigger_history: Trigger history for replay functionality
            use_individual_formatting: Whether to use individual memory formatting
        """

        with open(filepath, "r", encoding="utf-8") as f:
            data = DagMemoryData.model_validate_json(f.read())
        return cls.from_data(data, trigger_history, use_individual_formatting)

    def save_action_log(self, filepath: str) -> None:
        """Save the action log to a file."""
        self.action_log.save_to_file(filepath)

    @classmethod
    def load_from_action_log(
        cls,
        filepath: str,
        trigger_history: ITriggerHistory,
        use_individual_formatting: bool,
    ) -> "DagMemoryManager":
        """
        Create a manager by replaying an action log from file.

        Args:
            filepath: Path to the action log file
            trigger_history: Trigger history for replay functionality
            use_individual_formatting: Whether to use individual memory formatting

        Returns:
            Manager instance with state replayed from the action log
        """
        action_log = MemoryActionLog.load_from_file(filepath)
        memory_graph, context_graph = action_log.replay_from_empty(trigger_history)

        manager = cls(
            memory_graph, context_graph, trigger_history, use_individual_formatting
        )
        manager.action_log = action_log

        return manager

    def _apply_token_decay(self) -> None:
        """
        Apply token decay to all existing context memories.

        Each turn, memories naturally lose some token value to simulate aging.
        This makes memories more likely to be pruned if they're not being
        reinforced by retrieval or relevance.

        Args:
            decay_amount: Number of tokens to subtract from each memory (default: 2)
        """
        if not self.context_graph.elements:
            return

        decay_amount = 1

        logger.debug(
            f"Applying token decay of {decay_amount} to {len(self.context_graph.elements)} context memories"
        )

        self.dispatch_actions([ApplyTokenDecayAction(decay_amount=decay_amount)])
