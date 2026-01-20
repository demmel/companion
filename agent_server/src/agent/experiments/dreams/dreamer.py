"""Main dream generation orchestration."""

from typing import Optional

from agent.llm import LLM, SupportedModel
from agent.memory.dag.models import MemoryGraph

from datetime import datetime

from .models import (
    Dream,
    DreamConfig,
    DreamMode,
    DiscoveredConnection,
    SeedSelection,
    TraversalStrategy,
    NarrativeStyle,
)
from .seed_selection import select_seed
from .traversal import traverse
from .narrative import generate_narrative, extract_themes


class Dreamer:
    """
    Orchestrates dream generation from memory graphs.

    Combines seed selection, graph traversal, and narrative generation
    to create dream experiences from memories.
    """

    def __init__(self, memory_graph: MemoryGraph, llm: LLM, model: SupportedModel):
        """
        Initialize the Dreamer.

        Args:
            memory_graph: The memory graph to dream from
            llm: LLM instance for narrative generation
            model: Model to use for generation
        """
        self.memory_graph = memory_graph
        self.llm = llm
        self.model = model

    def dream(self, config: DreamConfig, fixed_seed_id: Optional[str] = None) -> Dream:
        """
        Generate a dream based on configuration.

        Args:
            config: Configuration for dream generation
            fixed_seed_id: If provided, use this seed instead of selecting one

        Returns:
            Generated Dream object
        """
        # Step 1: Select seed memory
        seed_id = select_seed(
            self.memory_graph, config.seed_selection.value, fixed_seed_id
        )

        if seed_id is None:
            raise ValueError(
                "No seed memory could be selected - memory graph may be empty"
            )

        # Step 2: Traverse the graph
        traversal_path, edges_used = traverse(
            self.memory_graph, seed_id, config.depth, config.traversal_strategy.value
        )

        # Step 3: Collect memories from traversal
        memories = [
            self.memory_graph.elements[mem_id]
            for mem_id in traversal_path
            if mem_id in self.memory_graph.elements
        ]

        if not memories:
            raise ValueError("No memories found in traversal path")

        # Step 4: Generate narrative
        narrative = generate_narrative(
            memories, config.narrative_style.value, self.llm, self.model
        )

        # Step 5: Extract themes
        themes = extract_themes(memories, self.llm, self.model)

        # Step 6: Create Dream object
        dream = Dream(
            seed_memory_id=seed_id,
            traversal_path=traversal_path,
            edges_used=edges_used,
            narrative=narrative,
            duration_memories=len(traversal_path),
            themes_emerged=themes,
            config=config,
        )

        return dream

    def dream_with_fixed_path(
        self, traversal_path: list[str], narrative_style: NarrativeStyle
    ) -> Dream:
        """
        Generate a dream from a fixed traversal path.

        Useful for comparing narrative styles with the same memories.

        Args:
            traversal_path: Pre-determined path through memories
            narrative_style: Style to use for narrative generation

        Returns:
            Generated Dream object
        """
        # Collect memories from path
        memories = [
            self.memory_graph.elements[mem_id]
            for mem_id in traversal_path
            if mem_id in self.memory_graph.elements
        ]

        if not memories:
            raise ValueError("No memories found in traversal path")

        # Generate narrative
        narrative = generate_narrative(
            memories, narrative_style.value, self.llm, self.model
        )

        # Extract themes
        themes = extract_themes(memories, self.llm, self.model)

        # Create config representing the fixed path
        config = DreamConfig(
            seed_selection=SeedSelection.RANDOM,  # Not actually used
            traversal_strategy=TraversalStrategy.RANDOM_JUMP,  # Not actually used
            depth=len(traversal_path),
            narrative_style=narrative_style,
        )

        # Create Dream object
        dream = Dream(
            seed_memory_id=traversal_path[0],
            traversal_path=traversal_path,
            edges_used=[],  # No edges tracked for fixed path
            narrative=narrative,
            duration_memories=len(traversal_path),
            themes_emerged=themes,
            config=config,
        )

        return dream

    def dream_mode(
        self,
        mode: DreamMode,
        since_timestamp: datetime | None = None,
        depth: int = 5,
    ) -> Dream:
        """
        Generate a dream using a purpose-driven mode.

        Args:
            mode: The type of dream to generate (TODAY, BIZARRE, CONNECT)
            since_timestamp: For TODAY mode, only use memories after this time
            depth: Number of memories to visit

        Returns:
            Generated Dream object (with discovered_connections for CONNECT mode)
        """
        if mode == DreamMode.TODAY:
            return self._dream_today(since_timestamp, depth)
        elif mode == DreamMode.BIZARRE:
            return self._dream_bizarre(depth)
        elif mode == DreamMode.CONNECT:
            return self._dream_connect(depth)
        else:
            raise ValueError(f"Unknown dream mode: {mode}")

    def _dream_today(self, since_timestamp: datetime | None, depth: int) -> Dream:
        """
        Dream about today - consolidate memories since last sleep.

        Uses semantic drift to stay thematically connected within today's memories.
        """
        # Filter memories to those since the timestamp
        if since_timestamp is None:
            # Default to all memories if no timestamp given
            memory_ids = list(self.memory_graph.elements.keys())
        else:
            memory_ids = [
                mem_id
                for mem_id, mem in self.memory_graph.elements.items()
                if mem.timestamp >= since_timestamp
            ]

        if not memory_ids:
            raise ValueError("No memories found since the given timestamp")

        # Pick a random seed from today's memories
        import random

        seed_id = random.choice(memory_ids)

        # Use semantic drift to stay thematically connected
        # But constrain traversal to today's memories only
        traversal_path = [seed_id]
        visited = {seed_id}

        while len(traversal_path) < depth and len(visited) < len(memory_ids):
            current_id = traversal_path[-1]
            current_mem = self.memory_graph.elements[current_id]

            # Find most similar unvisited memory from today
            best_id = None
            best_similarity = -2.0

            if current_mem.embedding_vector is not None:
                from agent.embedding_service import EmbeddingService

                for mem_id in memory_ids:
                    if mem_id in visited:
                        continue
                    mem = self.memory_graph.elements[mem_id]
                    if mem.embedding_vector is None:
                        continue

                    similarity = EmbeddingService.cosine_similarity(
                        current_mem.embedding_vector, mem.embedding_vector
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_id = mem_id

            if best_id is None:
                # Fall back to random from today
                candidates = [m for m in memory_ids if m not in visited]
                if not candidates:
                    break
                best_id = random.choice(candidates)

            traversal_path.append(best_id)
            visited.add(best_id)

        # Generate narrative
        memories = [self.memory_graph.elements[mem_id] for mem_id in traversal_path]

        narrative = generate_narrative(
            memories,
            NarrativeStyle.STREAM.value,  # Stream for consolidation feels right
            self.llm,
            self.model,
        )

        themes = extract_themes(memories, self.llm, self.model)

        config = DreamConfig(
            seed_selection=SeedSelection.RECENT,
            traversal_strategy=TraversalStrategy.SEMANTIC_DRIFT,
            depth=depth,
            narrative_style=NarrativeStyle.STREAM,
        )

        return Dream(
            seed_memory_id=seed_id,
            traversal_path=traversal_path,
            edges_used=[],
            narrative=narrative,
            duration_memories=len(traversal_path),
            themes_emerged=themes,
            config=config,
            mode=DreamMode.TODAY,
        )

    def _dream_bizarre(self, depth: int) -> Dream:
        """
        Bizarre dream - surreal, contrast-seeking dream like humans have.

        Uses contrast seeking for jarring transitions and fragment style.
        """
        config = DreamConfig(
            seed_selection=SeedSelection.RANDOM,
            traversal_strategy=TraversalStrategy.CONTRAST_SEEKING,
            depth=depth,
            narrative_style=NarrativeStyle.FRAGMENT,
        )

        dream = self.dream(config)
        # Add the mode to the returned dream
        dream.mode = DreamMode.BIZARRE
        return dream

    def _dream_connect(self, depth: int) -> Dream:
        """
        Connect the dots - find hidden connections between memories.

        After generating the dream, uses LLM to discover potential
        connections that could become new edges in the memory graph.
        """
        from .connections import discover_connections

        # Use random jump to get diverse memories
        config = DreamConfig(
            seed_selection=SeedSelection.RANDOM,
            traversal_strategy=TraversalStrategy.RANDOM_JUMP,
            depth=depth,
            narrative_style=NarrativeStyle.STREAM,
        )

        # Generate the base dream
        seed_id = select_seed(self.memory_graph, config.seed_selection.value)

        if seed_id is None:
            raise ValueError("No seed memory could be selected")

        traversal_path, edges_used = traverse(
            self.memory_graph, seed_id, config.depth, config.traversal_strategy.value
        )

        memories = [
            self.memory_graph.elements[mem_id]
            for mem_id in traversal_path
            if mem_id in self.memory_graph.elements
        ]

        if not memories:
            raise ValueError("No memories found in traversal path")

        # Generate narrative
        narrative = generate_narrative(
            memories, config.narrative_style.value, self.llm, self.model
        )

        themes = extract_themes(memories, self.llm, self.model)

        # Discover connections between the memories
        connections = discover_connections(memories, self.llm, self.model)

        return Dream(
            seed_memory_id=seed_id,
            traversal_path=traversal_path,
            edges_used=edges_used,
            narrative=narrative,
            duration_memories=len(traversal_path),
            themes_emerged=themes,
            config=config,
            mode=DreamMode.CONNECT,
            discovered_connections=connections,
        )


def create_dreamer(
    memory_graph: MemoryGraph,
    llm: LLM,
    model: SupportedModel = SupportedModel.MISTRAL_SMALL_3_2_Q4,
) -> Dreamer:
    """
    Create a Dreamer instance.

    Args:
        memory_graph: The memory graph to dream from
        llm: LLM instance for narrative generation
        model: Model to use (defaults to Mistral Small 3.2)

    Returns:
        Configured Dreamer instance
    """
    return Dreamer(memory_graph, llm, model)
