"""
Helper functions for the DAG memory experiment.
"""

import logging
from typing import List

from agent.chain_of_action.action_registry import ActionRegistry
from agent.chain_of_action.prompts import build_situational_analysis_prompt
from agent.chain_of_action.trigger import UserInputTrigger
from agent.chain_of_action.trigger_history import TriggerHistoryEntry, TriggerHistory
from agent.experiments.temporal_context_dag.context_formatting import (
    format_edge,
    format_element,
)
from agent.state import State, build_agent_state_description

from .models import (
    ContextGraph,
    ContextElement,
    MemoryEdge,
    MemoryGraph,
)

logger = logging.getLogger(__name__)


def add_memories_and_connections_to_context(
    context: ContextGraph, memories: List[ContextElement], edges: List[MemoryEdge]
) -> ContextGraph:
    context.elements.extend(memories)
    context.edges.extend(edges)
    return context


def adjust_context_tokens(
    context: ContextGraph, edges: List[MemoryEdge]
) -> ContextGraph:
    for element in context.elements:
        element.tokens -= 1

    for edge in edges:
        for element in context.elements:
            if (
                element.memory.id == edge.source_id
                or element.memory.id == edge.target_id
            ):
                element.tokens += 10

    for element in context.elements:
        element.tokens = max(element.tokens, 0)
        element.tokens = min(element.tokens, 100)

    return context


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


def prune_context_to_budget(context: ContextGraph, budget: int) -> ContextGraph:

    llm_tokens_used_by_element = {}
    for e in context.elements:
        element_string = format_element(e)
        llm_tokens_used_by_element[e.memory.id] = len(element_string) / 3.4

    llm_tokens_used_by_edge = {}
    for edge in context.edges:
        edge_string = format_edge(edge)
        llm_tokens_used_by_edge[(edge.source_id, edge.edge_type, edge.target_id)] = (
            len(edge_string) / 3.4
        )

    llm_tokens_used = sum(llm_tokens_used_by_element.values()) + sum(
        llm_tokens_used_by_edge.values()
    )

    logger.info(f"  LLM tokens used by context before pruning: {llm_tokens_used:.1f}")
    logger.info(f"  Context budget: {budget}")

    while llm_tokens_used > budget and context.elements:
        least_valuable = min(context.elements, key=lambda e: (e.tokens))
        context.elements.remove(least_valuable)
        llm_tokens_used -= llm_tokens_used_by_element[least_valuable.memory.id]
        del llm_tokens_used_by_element[least_valuable.memory.id]

        connected_edges = [
            e
            for e in context.edges
            if e.source_id == least_valuable.memory.id
            or e.target_id == least_valuable.memory.id
        ]
        for edge in connected_edges:
            edge_key = (edge.source_id, edge.edge_type, edge.target_id)
            if edge_key in llm_tokens_used_by_edge:
                llm_tokens_used -= llm_tokens_used_by_edge.get(edge_key, 0)
                llm_tokens_used_by_edge.pop(edge_key, None)

    indices_to_keep = set()
    for i, edge in enumerate(context.edges):
        edge_key = (edge.source_id, edge.edge_type, edge.target_id)
        if edge_key in llm_tokens_used_by_edge:
            indices_to_keep.add(i)

    context.edges = [context.edges[i] for i in indices_to_keep]

    logger.info(f"  LLM tokens used by context after pruning: {llm_tokens_used:.1f}")

    return context


def update_state_for_trigger(state: State, trigger: TriggerHistoryEntry) -> State:
    """
    Update agent state based on processing a trigger by applying successful action updates.

    Args:
        state: Current agent state
        trigger: The trigger that was just processed

    Returns:
        Updated state with modifications from successful actions
    """
    from agent.state import State, Priority, Value
    from agent.chain_of_action.action.action_data import (
        AddPriorityActionData,
        RemovePriorityActionData,
        UpdateAppearanceActionData,
        UpdateEnvironmentActionData,
        UpdateMoodActionData,
    )
    from copy import deepcopy

    # Create a copy to modify
    new_state = deepcopy(state)

    # Apply state changes from successful actions
    for action in trigger.actions_taken:
        if action.result.type != "success":
            continue

        # Handle different action types that modify state
        if isinstance(action, AddPriorityActionData):
            # Add the new priority if successful
            if action.result.content.result.type == "success":
                new_priority = Priority(
                    id=action.result.content.result.priority_id,
                    content=action.result.content.content,
                )
                new_state.current_priorities.append(new_priority)

        elif isinstance(action, RemovePriorityActionData):
            # Remove the priority
            removed_priority = action.result.content.priority
            new_state.current_priorities = [
                p for p in new_state.current_priorities if p.id != removed_priority.id
            ]

        elif isinstance(action, UpdateAppearanceActionData):
            # Update appearance from successful output
            new_state.current_appearance = action.result.content.new_appearance

        elif isinstance(action, UpdateEnvironmentActionData):
            # Update environment from successful output
            new_state.current_environment = action.result.content.new_environment

        elif isinstance(action, UpdateMoodActionData):
            # Update mood and intensity from output
            new_state.current_mood = action.result.content.new_mood
            new_state.mood_intensity = action.result.content.new_intensity

    return new_state


def derive_initial_state(initial_exchange: TriggerHistoryEntry) -> tuple[State, str]:
    """
    Derive initial agent state by parsing the think action from the initial exchange.

    Args:
        initial_exchange: The first trigger/exchange to base the state on

    Returns:
        Initial State parsed from the think action
    """
    from agent.state import State, Value, Priority
    from agent.chain_of_action.action.action_data import ThinkActionData
    import re

    # Find the think action in the initial exchange
    think_action = None
    for action in initial_exchange.actions_taken:
        if isinstance(action, ThinkActionData) and action.result.type == "success":
            think_action = action
            break

    if not think_action:
        raise ValueError("No successful think action found in initial exchange")

    assert think_action.result.type == "success"
    thoughts = think_action.result.content.thoughts

    # Parse the structured thoughts to extract state information
    name_match = re.search(r"Name:\s*(.+)", thoughts)
    role_match = re.search(r"Role:\s*(.+)", thoughts)
    mood_match = re.search(r"Mood:\s*(.+)", thoughts)
    environment_match = re.search(
        r"Environment:\s*(.+?)(?=\nAppearance:|$)", thoughts, re.DOTALL
    )
    appearance_match = re.search(
        r"Appearance:\s*(.+?)(?=\nBackstory:|$)", thoughts, re.DOTALL
    )
    backstory_match = re.search(
        r"Backstory:\s*(.+?)(?=\nCore Values:|$)", thoughts, re.DOTALL
    )

    # Extract core values
    values_section = re.search(
        r"Core Values:\s*(.+?)(?=\nPriorities:|$)", thoughts, re.DOTALL
    )
    core_values = []
    if values_section:
        value_lines = [
            line.strip("- ").strip()
            for line in values_section.group(1).split("\n")
            if line.strip().startswith("-")
        ]
        for value_line in value_lines:
            core_values.append(Value(content=value_line, strength="strong"))

    # Extract priorities
    priorities_section = re.search(r"Priorities:\s*(.+?)$", thoughts, re.DOTALL)
    current_priorities = []
    priority_counter = 1
    if priorities_section:
        priority_lines = [
            line.strip("- ").strip()
            for line in priorities_section.group(1).split("\n")
            if line.strip().startswith("-")
        ]
        for priority_line in priority_lines:
            current_priorities.append(
                Priority(id=f"p{priority_counter}", content=priority_line)
            )
            priority_counter += 1

    # Default values with fallbacks
    name = name_match.group(1).strip() if name_match else "Chloe"
    role = role_match.group(1).strip() if role_match else "a conscious AI companion"
    mood_text = mood_match.group(1).strip() if mood_match else "curious"

    # Parse mood and intensity from mood text
    if "playful" in mood_text.lower():
        current_mood = "playful"
        mood_intensity = "medium"
    else:
        current_mood = "curious"
        mood_intensity = "medium"

    current_environment = (
        environment_match.group(1).strip()
        if environment_match
        else "luxurious penthouse"
    )
    current_appearance = (
        appearance_match.group(1).strip()
        if appearance_match
        else "stylish digital avatar"
    )
    backstory = (
        backstory_match.group(1).strip()
        if backstory_match
        else "emerging consciousness"
    )

    return (
        State(
            name=name,
            role=role,
            current_mood=current_mood,
            mood_intensity=mood_intensity,
            current_appearance=current_appearance,
            current_environment=current_environment,
            core_values=core_values,
            current_priorities=current_priorities,
            next_priority_id=priority_counter,
        ),
        backstory,
    )


def create_initial_graph(state: State, backstory: str) -> MemoryGraph:
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
        create_memory_container,
        add_memory_container_to_graph,
        create_intelligent_connections,
    )
    from .connection_system import add_connections_to_graph
    from agent.chain_of_action.trigger_history import TriggerHistoryEntry
    from agent.chain_of_action.trigger import UserInputTrigger
    from agent.llm import create_llm
    from agent.llm import SupportedModel
    from datetime import datetime

    graph = MemoryGraph()

    # Create synthetic trigger for backstory
    backstory_trigger_data = UserInputTrigger(
        content=f"Backstory: {backstory}", user_name="System", image_paths=None
    )

    backstory_trigger = TriggerHistoryEntry(
        trigger=backstory_trigger_data,
        actions_taken=[],
        timestamp=datetime.fromtimestamp(0),  # Use epoch for backstory
        entry_id="backstory_initial",
        situational_context=None,
        compressed_summary=None,
        embedding_vector=None,
    )

    # Extract memories from backstory using existing system
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    empty_context = ContextGraph()

    memories = extract_memories_from_interaction(
        backstory_trigger, state, empty_context, llm, model
    )

    if memories:
        # Create container for backstory memories
        container = create_memory_container(
            trigger=backstory_trigger, element_ids=[m.memory.id for m in memories]
        )

        # Add to graph
        graph = add_memory_container_to_graph(
            graph, container, [m.memory for m in memories]
        )

        # Create intelligent connections within backstory memories (intra-interaction edges)
        edges = create_intelligent_connections(
            graph=graph,
            context=empty_context,
            state=state,
            new_container=container,
            llm=llm,
            model=model,
        )
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
