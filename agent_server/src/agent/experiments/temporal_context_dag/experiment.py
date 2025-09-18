import logging
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.conversation_persistence import AgentData, ConversationPersistence
from agent.experiments.temporal_context_dag.connection_system import (
    add_connections_to_graph,
)
from agent.experiments.temporal_context_dag.io import save_dag_to_json
from agent.experiments.temporal_context_dag.memory_formation import (
    add_memory_container_to_graph,
    create_memory_container,
    extract_memories_from_interaction,
)
from agent.experiments.temporal_context_dag.models import ContextGraph, MemoryGraph
from agent.experiments.temporal_context_dag.experiment_helpers import (
    add_memories_and_connections_to_context,
    adjust_context_tokens,
    calculate_context_budget,
    prune_context_to_budget,
    update_state_for_trigger,
    derive_initial_state,
    create_initial_graph,
    create_initial_context_subgraph,
)
from agent.llm import LLM, SupportedModel, create_llm
from agent.state import State

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_baseline_data(prefix: str) -> AgentData:
    persistence = ConversationPersistence()
    return persistence.load_agent_data(prefix)


def process_trigger(
    graph: MemoryGraph,
    context: ContextGraph,
    state: State,
    trigger: TriggerHistoryEntry,
    llm: LLM,
    model: SupportedModel,
    token_budget: int,
    update_state: bool = True,
) -> tuple[MemoryGraph, ContextGraph, State]:
    """
    Process a single trigger with current graph context, adding new memories and connections.

    Args:
        graph: Current memory graph state
        trigger: The trigger to process
        llm: LLM instance for memory extraction and connections
        model: Model to use for decisions
        context_budget: Token budget for context memories

    Returns:
        Updated memory graph with new memories and connections
    """
    from .memory_formation import create_intelligent_connections

    # Extract memories from this interaction with full context awareness
    memories = extract_memories_from_interaction(trigger, state, context, llm, model)

    if memories:
        # Create container for this interaction
        container = create_memory_container(
            trigger=trigger, element_ids=[m.memory.id for m in memories]
        )

        # Add to graph (creates temporal "follows" edges)
        graph = add_memory_container_to_graph(
            graph, container, [m.memory for m in memories]
        )

        logger.info(
            f"  Added {len(memories)} memories and connections for {trigger.entry_id}"
        )

        # Create intelligent connections between new memories and existing context
        edges = create_intelligent_connections(
            graph=graph,
            context=context,
            state=state,
            new_container=container,
            llm=llm,
            model=model,
        )
        graph = add_connections_to_graph(graph, edges)

        logger.info(f"  Created {len(edges)} new connections for {trigger.entry_id}")

        context = add_memories_and_connections_to_context(context, memories, edges)
        context = adjust_context_tokens(context, edges)
        context_budget = calculate_context_budget(token_budget, state)

        logger.info(
            f"  Context now has {len(context.elements)} elements and {len(context.edges)} edges"
        )

        context = prune_context_to_budget(context, context_budget)

        logger.info(
            f"  After pruning, context has {len(context.elements)} elements and {len(context.edges)} edges"
        )

        if update_state:
            state = update_state_for_trigger(state, trigger)

    else:
        logger.info(f"  No significant memories extracted from {trigger.entry_id}")

    return graph, context, state


def build_graph_from_triggers(
    data: AgentData, llm: LLM, model: SupportedModel, token_budget: int
) -> MemoryGraph:
    assert data.initial_exchange is not None, "Initial exchange is required"
    state, backstory = derive_initial_state(data.initial_exchange)
    graph = create_initial_graph(state, backstory)
    triggers = data.trigger_history.get_all_entries()
    context_budget = calculate_context_budget(token_budget, state)
    context = create_initial_context_subgraph(graph, context_budget)

    for i, trigger in enumerate(triggers):
        logger.info(
            f"\n=== Processing trigger {i+1}/{len(triggers)}: {trigger.entry_id} ==="
        )
        graph, context, state = process_trigger(
            graph,
            context,
            state,
            trigger,
            llm,
            model,
            token_budget,
        )
        save_dag_to_json(graph, "output_dag.json")

    return graph


def main():
    data = load_baseline_data("baseline")
    graph = MemoryGraph()
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    token_budget = 32000
    graph = build_graph_from_triggers(data, llm, model, token_budget)


if __name__ == "__main__":
    main()
