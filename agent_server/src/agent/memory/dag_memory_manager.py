"""
Action-based DAG Memory Manager that uses the reducer+action pattern for observability.

This manager emits actions instead of directly mutating state, enabling time-travel
debugging and complete replay of memory graph evolution.
"""

import logging
from agent.timeit import timeit
from typing import Sequence

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.memory_formation import (
    extract_memories_as_actions,
)
from agent.llm import LLM, SupportedModel
from agent.state import State
from pydantic import BaseModel

from .models import ContextElement, ContextGraph, MemoryGraph
from .actions import ApplyTokenDecayAction, MemoryAction, CheckpointAction
from .action_log import MemoryActionLog
from .context_management import (
    prune_context_to_budget_as_actions,
)
from .reducer import apply_action

logger = logging.getLogger(__name__)


def calculate_context_budget(
    token_budget: int,
    state: State,
    action_registry: ActionRegistry,
) -> int:
    """
    Calculate available token budget for context after accounting for prompt overhead.

    Args:
        token_budget: Total token budget
        state: Current agent state
        action_registry: Action registry for building sample prompt

    Returns:
        Available tokens for context
    """
    from agent.chain_of_action.prompts import build_situational_analysis_prompt
    from agent.chain_of_action.trigger import UserInputTrigger

    sa_prompt = build_situational_analysis_prompt(
        state=state,
        trigger=UserInputTrigger(content="sample", user_name="User"),
        trigger_history=TriggerHistory(),
        registry=action_registry,
        dag_memory_manager=DagMemoryManager(
            memory_graph=MemoryGraph(),
            context_graph=ContextGraph(),
            trigger_history=TriggerHistory(),
        ),
    )
    prompt_tokens = int(len(sa_prompt) / 3.4)

    context_budget = token_budget - prompt_tokens

    logger.info(
        f"Context budget calculation: total={token_budget}, prompt={prompt_tokens} => context budget={context_budget}"
    )

    return context_budget


class ContextElementData(BaseModel):
    memory_id: str
    tokens: int


class ContextGraphData(BaseModel):
    elements: list[ContextElementData]
    edges: list[str]  # Edge IDs instead of indices


class DagMemoryData(BaseModel):
    memory: MemoryGraph
    context: ContextGraphData


class DagMemoryManager:
    """
    Action-based DAG memory management system with full observability.

    Uses reducer+action pattern where all state changes are recorded as actions
    that can be replayed to reconstruct any historical state.
    """

    def __init__(
        self,
        memory_graph: MemoryGraph,
        context_graph: ContextGraph,
        trigger_history: TriggerHistory,
    ):
        """Initialize with existing graph state and empty action log."""
        self.memory_graph = memory_graph
        self.context_graph = context_graph
        self.trigger_history = trigger_history
        self.action_log = MemoryActionLog()

    @classmethod
    def create(
        cls,
        initial_state: State,
        token_budget: int,
        action_registry: ActionRegistry,
        trigger_history: TriggerHistory,
    ) -> "DagMemoryManager":
        """
        Create a new manager with initial state creation recorded as actions.

        Args:
            initial_state: Initial agent state
            backstory: Agent backstory for initial memories
            token_budget: Token budget for context management
            action_registry: Action registry for budget calculation
            trigger_history: Trigger history
            llm: LLM instance to use
            model: Model to use for semantic connection extraction
        """
        # Start with completely empty state - memories will be added via postprocess_trigger
        manager = cls(
            MemoryGraph(), ContextGraph(elements=[], edges=[]), trigger_history
        )

        # Record that we're starting with empty state
        manager.action_log.add_checkpoint(
            label="creation_start",
            description="Starting with empty memory graph - initial exchange will be processed via postprocess_trigger",
        )

        return manager

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

    def preprocess_trigger(
        self,
        trigger: Trigger,
        state: State,
        llm: LLM,
        model: SupportedModel,
        token_budget: int,
        action_registry: ActionRegistry,
    ) -> None:
        """
        Preprocess trigger by retrieving relevant memories and pruning context.

        This is called BEFORE the reasoning loop to ensure the agent has access
        to relevant retrieved memories during its reasoning process.

        Args:
            trigger: The incoming trigger (not yet processed)
            state: Current agent state
            llm: LLM instance for memory operations
            model: Model to use for decisions
            token_budget: Token budget for context management
            action_registry: Action registry for budget calculation
        """
        logger.info(f"Preprocessing trigger for memory retrieval")

        # Checkpoint: Start of preprocessing
        self.add_checkpoint(
            label=f"preprocess_start",
            description=f"Starting preprocessing for incoming trigger",
        )

        # STEP 1: Apply token decay to existing context memories
        self._apply_token_decay()

        # STEP 2: Memory retrieval based on incoming trigger
        from .retrieval_integration import (
            retrieve_relevant_memories_as_actions,
        )

        with timeit("Memory Retrieval"):
            retrieval_actions = retrieve_relevant_memories_as_actions(
                memory_graph=self.memory_graph,
                context_graph=self.context_graph,
                state=state,
                trigger=trigger,
                llm=llm,
                model=model,
                max_retrieved_memories=5,
                max_queries=6,
                min_similarity_threshold=0.4,
            )

        if retrieval_actions:
            self.dispatch_actions(retrieval_actions)

            # Checkpoint: Memories retrieved
            self.add_checkpoint(
                label=f"memories_retrieved_preprocess",
                description=f"Retrieved {len(retrieval_actions)} relevant memories during preprocessing",
            )

        # STEP 3: Prune context to budget BEFORE reasoning
        with timeit("Context Pruning"):
            context_budget = calculate_context_budget(
                token_budget, state, action_registry
            )
            pruning_actions = prune_context_to_budget_as_actions(
                self.memory_graph, self.context_graph, context_budget
            )

        if pruning_actions:
            self.dispatch_actions(pruning_actions)

            # Checkpoint: Context pruned
            self.add_checkpoint(
                label=f"context_pruned_preprocess",
                description=f"Pruned context to fit budget of {context_budget} tokens",
            )

        logger.info(
            f"Preprocessing complete - Context: {len(self.context_graph.elements)} elements, "
            f"{len(self.context_graph.edges)} edges"
        )

    def postprocess_trigger(
        self,
        trigger: TriggerHistoryEntry,
        state: State,
        llm: LLM,
        model: SupportedModel,
        token_budget: int,
        action_registry: ActionRegistry,
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
            memory_graph, context_graph, self.trigger_history
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
        cls, data: DagMemoryData, trigger_history: TriggerHistory
    ) -> "DagMemoryManager":
        """
        Create a DagMemoryManager from a serialized data object.

        Args:
            data: A DagMemoryData object containing the memory and context graphs
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
        return cls(memory_graph, context_graph, trigger_history)

    def save_to_file(self, filepath: str) -> None:
        """
        Save the current memory graph to a JSON file.

        Args:
            filepath: Path to save the memory graph
        """

        data = self.to_data()
        with open(filepath, "w") as f:
            f.write(data.model_dump_json(indent=2))

    @classmethod
    def load_from_file(
        cls, filepath: str, trigger_history: TriggerHistory
    ) -> "DagMemoryManager":
        """
        Load a memory graph from a JSON file.

        Args:
            filepath: Path to load the memory graph from
        """

        with open(filepath, "r") as f:
            data = DagMemoryData.model_validate_json(f.read())
        return cls.from_data(data, trigger_history)

    def save_action_log(self, filepath: str) -> None:
        """Save the action log to a file."""
        self.action_log.save_to_file(filepath)

    @classmethod
    def load_from_action_log(
        cls, filepath: str, trigger_history: TriggerHistory
    ) -> "DagMemoryManager":
        """
        Create a manager by replaying an action log from file.

        Args:
            filepath: Path to the action log file

        Returns:
            Manager instance with state replayed from the action log
        """
        action_log = MemoryActionLog.load_from_file(filepath)
        memory_graph, context_graph = action_log.replay_from_empty(trigger_history)

        manager = cls(memory_graph, context_graph, trigger_history)
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
