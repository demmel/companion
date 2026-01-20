"""Knowledge Graph infrastructure for fair attribute-aware retrieval testing.

This module provides:
1. Entity resolution - clustering mentions to canonical entities
2. Attribute normalization - mapping raw attributes to canonical schema
3. Knowledge graph - storing facts with proper structure
4. Query interface - resolving natural language queries to KG lookups
"""

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from agent.embedding_service import EmbeddingService


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Entity:
    """A canonical entity in the knowledge graph."""

    id: str  # "entity_david"
    canonical_name: str  # "David"
    aliases: set[str] = field(default_factory=set)  # {"david", "my husband"}
    entity_type: str = "unknown"  # "person", "place", "topic"


@dataclass
class AttributeSchema:
    """Schema for an attribute type."""

    id: str  # "attr_mood"
    canonical_name: str  # "mood"
    aliases: set[str] = field(
        default_factory=set
    )  # {"current_mood", "emotional_state"}
    attribute_type: str = "replacement"  # "replacement" or "additive"


@dataclass
class KGFact:
    """A fact in the knowledge graph."""

    entity_id: str  # Resolved entity ID
    attribute_id: str  # Normalized attribute ID
    value: str  # The actual value
    source_memory_id: str  # Which memory this came from
    timestamp: int  # For temporal ordering


# =============================================================================
# Entity Resolver
# =============================================================================


class EntityResolver:
    """Resolves mentions to canonical entities using exact match + embedding similarity."""

    def __init__(
        self, embedding_service: EmbeddingService, similarity_threshold: float = 0.7
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.entities: dict[str, Entity] = {}
        self.alias_to_entity: dict[str, str] = {}  # lowercase alias -> entity_id
        self._entity_counter = 0

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip()

    def _generate_id(self, name: str) -> str:
        """Generate a unique entity ID."""
        self._entity_counter += 1
        base = self._normalize(name).replace(" ", "_")
        return f"entity_{base}_{self._entity_counter}"

    def add_entity(
        self,
        canonical_name: str,
        aliases: set[str] | None = None,
        entity_type: str = "unknown",
    ) -> str:
        """Add a new entity to the resolver."""
        entity_id = self._generate_id(canonical_name)
        all_aliases = {self._normalize(canonical_name)}
        if aliases:
            all_aliases.update(self._normalize(a) for a in aliases)

        entity = Entity(
            id=entity_id,
            canonical_name=canonical_name,
            aliases=all_aliases,
            entity_type=entity_type,
        )
        self.entities[entity_id] = entity

        for alias in all_aliases:
            self.alias_to_entity[alias] = entity_id

        return entity_id

    def resolve(self, mention: str) -> str | None:
        """Resolve a mention to an entity ID.

        1. Try exact alias match
        2. Try embedding similarity to known aliases
        3. Return None if no match (caller decides whether to create new)
        """
        normalized = self._normalize(mention)

        # Exact match
        if normalized in self.alias_to_entity:
            return self.alias_to_entity[normalized]

        # Embedding similarity
        if not self.entities:
            return None

        mention_emb = np.array(self.embedding_service.encode(mention))

        best_score = 0.0
        best_entity_id = None

        for entity_id, entity in self.entities.items():
            for alias in entity.aliases:
                alias_emb = np.array(self.embedding_service.encode(alias))
                score = float(np.dot(mention_emb, alias_emb))
                if score > best_score:
                    best_score = score
                    best_entity_id = entity_id

        if best_score >= self.similarity_threshold:
            # Add as new alias
            if best_entity_id:
                self.entities[best_entity_id].aliases.add(normalized)
                self.alias_to_entity[normalized] = best_entity_id
            return best_entity_id

        return None

    def resolve_or_create(self, mention: str, entity_type: str = "unknown") -> str:
        """Resolve a mention, creating a new entity if needed."""
        entity_id = self.resolve(mention)
        if entity_id is None:
            entity_id = self.add_entity(mention, entity_type=entity_type)
        return entity_id


# =============================================================================
# Attribute Normalizer
# =============================================================================


class AttributeNormalizer:
    """Normalizes raw attributes to canonical schema."""

    def __init__(
        self, embedding_service: EmbeddingService, similarity_threshold: float = 0.7
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.schema: dict[str, AttributeSchema] = {}
        self.alias_to_attr: dict[str, str] = {}  # lowercase alias -> attr_id
        self._init_default_schema()

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip().replace("_", " ")

    def _init_default_schema(self) -> None:
        """Initialize with common attribute types for companion agent."""
        # Replacement attributes (most recent wins)
        replacement_attrs = [
            ("mood", ["current_mood", "emotional_state", "feeling", "emotions"]),
            ("location", ["current_location", "where", "place"]),
            (
                "appearance",
                ["current_appearance", "wearing", "outfit", "dressed", "clothes"],
            ),
            ("activity", ["current_activity", "doing", "action"]),
            ("status", ["current_status", "state"]),
            ("desire", ["want", "wants", "wanting", "craving"]),
        ]

        # Additive attributes (accumulates)
        additive_attrs = [
            (
                "preferences",
                ["likes", "dislikes", "favorite", "preference", "food_preference"],
            ),
            ("relationships", ["knows", "relationship", "friend", "family"]),
            ("experiences", ["experienced", "did", "went", "has_done"]),
            ("knowledge", ["learned", "knows_about", "understands"]),
            ("goals", ["goal", "priority", "priorities", "wants_to"]),
            ("traits", ["personality", "trait", "characteristic"]),
        ]

        for canonical, aliases in replacement_attrs:
            self._add_to_schema(canonical, aliases, "replacement")

        for canonical, aliases in additive_attrs:
            self._add_to_schema(canonical, aliases, "additive")

    def _add_to_schema(
        self, canonical: str, aliases: list[str], attr_type: str
    ) -> None:
        """Add an attribute to the schema."""
        attr_id = f"attr_{canonical}"
        all_aliases = {self._normalize(canonical)}
        all_aliases.update(self._normalize(a) for a in aliases)

        self.schema[attr_id] = AttributeSchema(
            id=attr_id,
            canonical_name=canonical,
            aliases=all_aliases,
            attribute_type=attr_type,
        )

        for alias in all_aliases:
            self.alias_to_attr[alias] = attr_id

    def normalize(self, raw_attribute: str) -> tuple[str | None, str]:
        """Normalize a raw attribute to canonical ID.

        Returns: (attr_id or None, attribute_type)
        """
        normalized = self._normalize(raw_attribute)

        # Exact match
        if normalized in self.alias_to_attr:
            attr_id = self.alias_to_attr[normalized]
            return attr_id, self.schema[attr_id].attribute_type

        # Embedding similarity
        if not self.schema:
            return None, "replacement"

        attr_emb = np.array(self.embedding_service.encode(raw_attribute))

        best_score = 0.0
        best_attr_id = None

        for attr_id, attr_schema in self.schema.items():
            for alias in attr_schema.aliases:
                alias_emb = np.array(self.embedding_service.encode(alias))
                score = float(np.dot(attr_emb, alias_emb))
                if score > best_score:
                    best_score = score
                    best_attr_id = attr_id

        if best_score >= self.similarity_threshold and best_attr_id:
            # Add as new alias
            self.schema[best_attr_id].aliases.add(normalized)
            self.alias_to_attr[normalized] = best_attr_id
            return best_attr_id, self.schema[best_attr_id].attribute_type

        # No match - return None, default to replacement
        return None, "replacement"

    def get_attribute_type(self, attr_id: str) -> str:
        """Get the type (replacement/additive) for an attribute."""
        if attr_id in self.schema:
            return self.schema[attr_id].attribute_type
        return "replacement"


# =============================================================================
# Knowledge Graph
# =============================================================================


class KnowledgeGraph:
    """Knowledge graph with entity resolution and attribute normalization."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.entity_resolver = EntityResolver(embedding_service)
        self.attribute_normalizer = AttributeNormalizer(embedding_service)
        self.facts: list[KGFact] = []

    def add_fact(
        self,
        raw_entity: str,
        raw_attribute: str,
        value: str,
        source_memory_id: str,
        timestamp: int,
    ) -> KGFact | None:
        """Add a fact to the KG, resolving entity and normalizing attribute."""
        # Resolve entity
        entity_id = self.entity_resolver.resolve_or_create(raw_entity)

        # Normalize attribute
        attr_id, attr_type = self.attribute_normalizer.normalize(raw_attribute)
        if attr_id is None:
            # Unknown attribute - create a new one
            attr_id = f"attr_{raw_attribute.lower().replace(' ', '_')}"
            self.attribute_normalizer.schema[attr_id] = AttributeSchema(
                id=attr_id,
                canonical_name=raw_attribute,
                aliases={raw_attribute.lower()},
                attribute_type=attr_type,
            )
            self.attribute_normalizer.alias_to_attr[raw_attribute.lower()] = attr_id

        fact = KGFact(
            entity_id=entity_id,
            attribute_id=attr_id,
            value=value,
            source_memory_id=source_memory_id,
            timestamp=timestamp,
        )
        self.facts.append(fact)
        return fact

    def get_entity_facts(self, entity_id: str) -> list[KGFact]:
        """Get all facts for an entity."""
        return [f for f in self.facts if f.entity_id == entity_id]

    def get_current_state(self, entity_id: str) -> list[KGFact]:
        """Get current state for an entity.

        For replacement attrs: only most recent fact
        For additive attrs: all facts
        """
        entity_facts = self.get_entity_facts(entity_id)

        # Group by attribute
        by_attr: dict[str, list[KGFact]] = defaultdict(list)
        for fact in entity_facts:
            by_attr[fact.attribute_id].append(fact)

        result: list[KGFact] = []
        for attr_id, attr_facts in by_attr.items():
            attr_type = self.attribute_normalizer.get_attribute_type(attr_id)
            if attr_type == "replacement":
                # Only most recent
                most_recent = max(attr_facts, key=lambda f: f.timestamp)
                result.append(most_recent)
            else:
                # All facts
                result.extend(attr_facts)

        return result

    def get_attribute_facts(self, entity_id: str, attribute_id: str) -> list[KGFact]:
        """Get facts for a specific entity.attribute."""
        attr_facts = [
            f
            for f in self.facts
            if f.entity_id == entity_id and f.attribute_id == attribute_id
        ]

        attr_type = self.attribute_normalizer.get_attribute_type(attribute_id)
        if attr_type == "replacement":
            if attr_facts:
                return [max(attr_facts, key=lambda f: f.timestamp)]
            return []
        else:
            return attr_facts

    def resolve_entity_from_query(self, query: str) -> str | None:
        """Find which entity a query is asking about.

        Uses embedding similarity between query and entity names/aliases.
        """
        if not self.entity_resolver.entities:
            return None

        query_emb = np.array(self.embedding_service.encode(query))

        best_score = 0.0
        best_entity_id = None

        for entity_id, entity in self.entity_resolver.entities.items():
            # Compare to canonical name and aliases
            for name in [entity.canonical_name] + list(entity.aliases):
                name_emb = np.array(self.embedding_service.encode(name))
                score = float(np.dot(query_emb, name_emb))
                if score > best_score:
                    best_score = score
                    best_entity_id = entity_id

        return best_entity_id

    def resolve_attribute_from_query(self, query: str, entity_id: str) -> str | None:
        """Find which attribute a query is asking about.

        Returns None if it seems like an entity overview query.
        """
        # Get attributes this entity has
        entity_facts = self.get_entity_facts(entity_id)
        entity_attrs = set(f.attribute_id for f in entity_facts)

        if not entity_attrs:
            return None

        query_emb = np.array(self.embedding_service.encode(query))

        best_score = 0.0
        best_attr_id = None

        for attr_id in entity_attrs:
            if attr_id not in self.attribute_normalizer.schema:
                continue
            attr_schema = self.attribute_normalizer.schema[attr_id]

            # Compare to canonical name and aliases
            for name in [attr_schema.canonical_name] + list(attr_schema.aliases):
                name_emb = np.array(self.embedding_service.encode(name))
                score = float(np.dot(query_emb, name_emb))
                if score > best_score:
                    best_score = score
                    best_attr_id = attr_id

        # Only return if there's a strong match (otherwise it's an overview query)
        if best_score >= 0.5:
            return best_attr_id

        return None


# =============================================================================
# KG-based Retrieval
# =============================================================================


def retrieve_kg_aware(
    query: str,
    kg: KnowledgeGraph,
    top_k: int = 5,
) -> list[str]:
    """Fair KG-based retrieval - no hardcoding.

    1. Find entity in query using embedding similarity
    2. Determine if specific attribute or overview
    3. Return relevant memory IDs
    """
    # Find entity
    entity_id = kg.resolve_entity_from_query(query)
    if not entity_id:
        return []

    # Try to find specific attribute
    attribute_id = kg.resolve_attribute_from_query(query, entity_id)

    if attribute_id:
        # Specific attribute query
        facts = kg.get_attribute_facts(entity_id, attribute_id)
    else:
        # Entity overview query
        facts = kg.get_current_state(entity_id)

    # Get unique memory IDs
    memory_ids = list(dict.fromkeys(f.source_memory_id for f in facts))
    return memory_ids[:top_k]
