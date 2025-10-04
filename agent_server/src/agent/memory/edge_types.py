"""
Consolidated edge type definitions with clean hierarchy:
ReversibleEdgeType ⊃ MemoryEdgeType ⊃ ConnectionType
"""

from enum import Enum


class EdgeType(str, Enum):
    """All possible edge types including forward and reverse forms."""

    EXPLAINS = "explains"
    EXPLAINED_BY = "explained_by"

    CAUSED = "caused"
    CAUSED_BY = "caused_by"

    CONTRADICTS = "contradicts"
    CONTRADICTED_BY = "contradicted_by"

    CLARIFIES = "clarifies"
    CLARIFIED_BY = "clarified_by"

    RETRACTS = "retracts"
    RETRACTED_BY = "retracted_by"

    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"

    CORRECTS = "corrects"
    CORRECTED_BY = "corrected_by"


class GraphEdgeType(str, Enum):
    """Edge types that can exist in memory graph - subset of ReversibleEdgeType."""

    EXPLAINED_BY = EdgeType.EXPLAINED_BY.value
    EXPLAINS = EdgeType.EXPLAINS.value
    CAUSED = EdgeType.CAUSED.value
    CONTRADICTED_BY = EdgeType.CONTRADICTED_BY.value
    CLARIFIED_BY = EdgeType.CLARIFIED_BY.value
    RETRACTED_BY = EdgeType.RETRACTED_BY.value
    SUPERSEDED_BY = EdgeType.SUPERSEDED_BY.value
    CORRECTED_BY = EdgeType.CORRECTED_BY.value


class AgentControlledEdgeType(str, Enum):
    """Edge types agents can create - subset of MemoryEdgeType."""

    EXPLAINED_BY = GraphEdgeType.EXPLAINED_BY.value
    EXPLAINS = GraphEdgeType.EXPLAINS.value
    CAUSED = GraphEdgeType.CAUSED.value
    CONTRADICTED_BY = GraphEdgeType.CONTRADICTED_BY.value
    CLARIFIED_BY = GraphEdgeType.CLARIFIED_BY.value
    RETRACTED_BY = GraphEdgeType.RETRACTED_BY.value
    SUPERSEDED_BY = GraphEdgeType.SUPERSEDED_BY.value
    CORRECTED_BY = GraphEdgeType.CORRECTED_BY.value


def get_prompt_description(edge_type: AgentControlledEdgeType) -> str:
    """Get prompt description for edge type - statically exhaustive."""
    match edge_type:
        case AgentControlledEdgeType.EXPLAINED_BY:
            return "Existing memory provides context/explanation for new memory"
        case AgentControlledEdgeType.EXPLAINS:
            return "Existing memory is explained/given context by new memory"
        case AgentControlledEdgeType.CAUSED:
            return "Existing memory caused/led to new memory"
        case AgentControlledEdgeType.CONTRADICTED_BY:
            return "Existing memory is definitively false, contradicted by new memory"
        case AgentControlledEdgeType.CLARIFIED_BY:
            return "Existing memory was a misunderstanding, clarified by new memory"
        case AgentControlledEdgeType.RETRACTED_BY:
            return "Existing memory is completely withdrawn/retracted by new memory"
        case AgentControlledEdgeType.SUPERSEDED_BY:
            return "Existing memory is superseded/overridden by new memory"
        case AgentControlledEdgeType.CORRECTED_BY:
            return "Existing memory is corrected/updated by new memory"


def get_context_description(edge_type: EdgeType) -> str:
    """Get context description for edge type - statically exhaustive."""
    match edge_type:
        case EdgeType.EXPLAINS:
            return "This memory provides context, background, or reasoning for another memory"
        case EdgeType.EXPLAINED_BY:
            return "This memory is given context, background, or reasoning by another memory"
        case EdgeType.CAUSED:
            return "This memory directly caused, triggered, or led to another memory"
        case EdgeType.CAUSED_BY:
            return "This memory was directly caused, triggered, or resulted from another memory"
        case EdgeType.CONTRADICTS:
            return "This memory definitively contradicts another memory as false"
        case EdgeType.CONTRADICTED_BY:
            return "This memory is definitively false, contradicted by another memory"
        case EdgeType.CLARIFIES:
            return "This memory clarifies a misunderstanding in another memory"
        case EdgeType.CLARIFIED_BY:
            return "This memory was a misunderstanding, clarified by another memory"
        case EdgeType.RETRACTS:
            return "This memory completely withdraws/retracts another memory"
        case EdgeType.RETRACTED_BY:
            return "This memory is completely withdrawn/retracted by another memory"
        case EdgeType.SUPERSEDES:
            return "This memory supersedes/overrides another memory"
        case EdgeType.SUPERSEDED_BY:
            return "This memory is superseded/overridden by another memory"
        case EdgeType.CORRECTS:
            return "This memory corrects/updates another memory"
        case EdgeType.CORRECTED_BY:
            return "This memory is corrected/updated by another memory"


REVERSALS = [
    (EdgeType.EXPLAINS, EdgeType.EXPLAINED_BY),
    (EdgeType.CAUSED, EdgeType.CAUSED_BY),
    (EdgeType.CONTRADICTS, EdgeType.CONTRADICTED_BY),
    (EdgeType.CLARIFIES, EdgeType.CLARIFIED_BY),
    (EdgeType.RETRACTS, EdgeType.RETRACTED_BY),
    (EdgeType.SUPERSEDES, EdgeType.SUPERSEDED_BY),
    (EdgeType.CORRECTS, EdgeType.CORRECTED_BY),
]

REVERSE_MAPPING = {a: b for a, b in REVERSALS}
REVERSE_MAPPING.update({b: a for a, b in REVERSALS})
assert set(REVERSE_MAPPING.keys()) == set(
    EdgeType
), "Reverse mapping must cover all edge types"


def get_prompt_edge_type_list():
    """Get comma-separated list for prompts."""
    return ", ".join(e.value for e in AgentControlledEdgeType)


def get_edge_type_memory_formation_descriptions():
    """Get connection descriptions for prompts."""
    lines = ["Connection types (existing memory → new memory):"]
    for t in AgentControlledEdgeType:
        lines.append(f"- {t.value}: {get_prompt_description(t)}")
    return "\n".join(lines)


def get_edge_type_context_descrioptions():
    """Get context documentation lines."""
    lines = ["**Connection Types:**"]
    # Forward descriptions
    for t in EdgeType:
        lines.append(f"- `{t.value}`: {get_context_description(t)}")
    return lines


def validate_hierarchy():
    """Validate the enum hierarchy: ReversibleEdgeType ⊃ MemoryEdgeType ⊃ ConnectionType."""
    reversible_values = {e.value for e in EdgeType}
    memory_values = {e.value for e in GraphEdgeType}
    connection_values = {e.value for e in AgentControlledEdgeType}

    # Check MemoryEdgeType ⊆ ReversibleEdgeType
    invalid_memory = memory_values - reversible_values
    if invalid_memory:
        raise ValueError(
            f"MemoryEdgeType contains values not in ReversibleEdgeType: {invalid_memory}"
        )

    # Check ConnectionType ⊆ MemoryEdgeType
    invalid_connection = connection_values - memory_values
    if invalid_connection:
        raise ValueError(
            f"ConnectionType contains values not in MemoryEdgeType: {invalid_connection}"
        )


# Validate hierarchy on import
validate_hierarchy()
