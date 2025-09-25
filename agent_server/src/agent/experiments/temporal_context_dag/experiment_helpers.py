"""
Helper functions for the DAG memory experiment.
"""

import logging

from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.prompts import build_situational_analysis_prompt
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.state import State

from .models import (
    ContextGraph,
    MemoryContainer,
    MemoryGraph,
)

logger = logging.getLogger(__name__)


def calculate_context_budget(
    token_budget: int,
    state: State,
    action_registry: ActionRegistry,
) -> int:
    sa_prompt = build_situational_analysis_prompt(
        state=state,
        trigger=UserInputTrigger(content="sample", user_name="User"),
        trigger_history=TriggerHistory(),
        relevant_memories=[],
        registry=action_registry,
        dag_context=ContextGraph(),
    )
    prompt_tokens = int(len(sa_prompt) / 3.4)

    context_budget = token_budget - prompt_tokens

    logger.info(
        f"Context budget calculation: total={token_budget}, prompt={prompt_tokens} => context budget={context_budget}"
    )

    return context_budget


def create_initial_graph(
    state: State, backstory: str, initial_exchange: TriggerHistoryEntry
) -> MemoryGraph:
    """
    Create initial memory graph with memories extracted from backstory.

    Args:
        state: Initial agent state
        backstory: Backstory text to extract memories from

    Returns:
        MemoryGraph with backstory memories and their connections
    """
    from .models import MemoryGraph, ContextGraph
    from .memory_formation import (
        extract_memories_from_interaction,
        add_memory_container_to_graph,
    )
    from .connection_system import add_connections_to_graph
    from agent.chain_of_action.trigger_history import TriggerHistoryEntry
    from agent.chain_of_action.trigger import UserInputTrigger
    from agent.llm import create_llm
    from agent.llm import SupportedModel
    from datetime import datetime

    graph = MemoryGraph()

    # Extract memories from backstory using existing system
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    empty_context = ContextGraph()

    memories, edges = extract_memories_from_interaction(
        initial_exchange, state, empty_context, llm, model
    )

    if memories:
        # Create container for backstory memories
        container = MemoryContainer(
            trigger=initial_exchange, element_ids=[m.memory.id for m in memories]
        )

        # Add to graph
        graph = add_memory_container_to_graph(
            graph, container, [m.memory for m in memories]
        )

        # Add connections extracted during memory formation
        graph, _ = add_connections_to_graph(graph, edges)

    return graph


def create_initial_context_subgraph(graph: MemoryGraph, budget: int) -> ContextGraph:
    """
    Create initial context subgraph with proper token assignment based on emotional significance.
    """
    from .models import ContextElement

    return ContextGraph(
        elements=[
            ContextElement(memory=m, tokens=int(m.emotional_significance * 100))
            for m in graph.elements.values()
        ],
        edges=[e for e in graph.edges.values()],
    )
