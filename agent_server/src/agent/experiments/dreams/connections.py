"""Connection discovery for Connect dreams."""

import json
import logging

from agent.llm import LLM, SupportedModel, Message
from agent.memory.dag.models import MemoryElement

from .models import DiscoveredConnection


logger = logging.getLogger(__name__)


# Valid edge types the agent can create
VALID_EDGE_TYPES = [
    "EXPLAINS",
    "CAUSED",
    "CLARIFIED_BY",
    "CONTRADICTED_BY",
]


def discover_connections(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> list[DiscoveredConnection]:
    """
    Use LLM to discover potential connections between memories.

    Analyzes the memories and identifies relationships that could
    become new edges in the memory graph.

    Args:
        memories: List of memories to analyze for connections
        llm: LLM instance
        model: Model to use

    Returns:
        List of discovered connections
    """
    if len(memories) < 2:
        return []

    # Build context with memory IDs and content
    memory_context = "\n".join(
        [
            f"[{i+1}] ID: {mem.id}\n    Content: {mem.content}"
            for i, mem in enumerate(memories)
        ]
    )

    prompt = f"""Analyze these memories and identify meaningful connections between them.

MEMORIES:
{memory_context}

Find connections where one memory relates to another. Valid relationship types:
- EXPLAINS: Memory A explains or provides context for Memory B
- CAUSED: Memory A led to or caused Memory B
- CLARIFIED_BY: Memory A is clarified or expanded by Memory B
- CONTRADICTED_BY: Memory A contradicts or conflicts with Memory B

Only identify connections that are genuinely meaningful - not every pair is connected.

Respond with a JSON array of connections. Each connection has:
- source_id: The ID of the source memory
- target_id: The ID of the target memory
- edge_type: One of EXPLAINS, CAUSED, CLARIFIED_BY, CONTRADICTED_BY
- reasoning: Brief explanation of why this connection exists

Example response:
[
  {{
    "source_id": "abc123...",
    "target_id": "def456...",
    "edge_type": "EXPLAINS",
    "reasoning": "The first memory provides context for understanding the second"
  }}
]

If no meaningful connections exist, return an empty array: []

Respond ONLY with the JSON array, no other text."""

    messages = [
        Message(
            role="system",
            content="You are analyzing memories to discover connections between them.",
        ),
        Message(role="user", content=prompt),
    ]

    response = llm.chat(model, messages, caller="dream_discover_connections")

    # Parse the response
    connections: list[DiscoveredConnection] = []

    try:
        # Extract JSON from response
        content = response.strip()

        # Handle potential markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Find the JSON content between the code block markers
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        parsed = json.loads(content)

        if not isinstance(parsed, list):
            logger.warning("Connection discovery returned non-list response")
            return []

        # Build a set of valid memory IDs
        valid_ids = {mem.id for mem in memories}

        for item in parsed:
            if not isinstance(item, dict):
                continue

            source_id = item.get("source_id", "")
            target_id = item.get("target_id", "")
            edge_type = item.get("edge_type", "")
            reasoning = item.get("reasoning", "")

            # Validate
            if source_id not in valid_ids:
                logger.debug(
                    f"Skipping connection: source_id {source_id} not in memories"
                )
                continue
            if target_id not in valid_ids:
                logger.debug(
                    f"Skipping connection: target_id {target_id} not in memories"
                )
                continue
            if edge_type not in VALID_EDGE_TYPES:
                logger.debug(f"Skipping connection: invalid edge_type {edge_type}")
                continue
            if source_id == target_id:
                logger.debug("Skipping connection: source and target are the same")
                continue

            connections.append(
                DiscoveredConnection(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    reasoning=reasoning,
                )
            )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse connection discovery response: {e}")
    except Exception as e:
        logger.warning(f"Error processing connection discovery: {e}")

    return connections
