#!/usr/bin/env python3
"""
Relationship Schema Bank

Maintains a registry of relationship schemas with semantic roles to avoid
synonymous duplicates and provide consistency in N-ary relationship processing.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Tuple
import logging
import json
import numpy as np

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State
from agent.memory.embedding_service import get_embedding_service
from agent.experiments.knowledge_graph.knn_entity_search import (
    IKNNEntity,
    KNNEntitySearch,
)

logger = logging.getLogger(__name__)


@dataclass
class RelationshipSchemaInfo:
    name: str
    usage_count: int
    description: str
    semantic_roles: List[str]
    category: str


@dataclass
class RelationshipBankStats:
    total_schemas: int
    total_usage: int
    most_used_schema: Optional[str] = None
    most_used_count: int = 0
    schemas: Optional[List[RelationshipSchemaInfo]] = None


class RelationshipSchemaMatch(BaseModel):
    """Result of relationship schema matching"""

    reasoning: str = Field(description="Why this choice was made")
    use_existing: bool = Field(
        description="Whether to use an existing relationship schema"
    )
    relationship_type: str = Field(description="The relationship type to use")
    semantic_roles: List[str] = Field(
        description="The semantic roles for this relationship"
    )
    role_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from proposed roles to existing schema roles if different",
    )


class RelationshipSchemaEntry(BaseModel, IKNNEntity):
    """Entry in the relationship schema bank"""

    name: str
    description: str
    semantic_roles: List[str] = Field(
        description="List of semantic roles for this relationship"
    )
    category: str = Field(
        description="Semantic category: state, action, transfer, comparative, etc."
    )
    examples: List[str]
    usage_count: int
    created_at: datetime
    last_used: datetime
    embedding: Optional[List[float]] = None

    def get_id(self) -> str:
        """Return unique identifier for KNN search"""
        return self.name

    def get_embedding(self) -> np.ndarray:
        """Return embedding for KNN search"""
        return np.array(self.embedding)

    def get_text(self) -> str:
        """Return text representation for KNN search"""
        roles_text = ", ".join(self.semantic_roles)
        return f"{self.name} ({self.category}): {roles_text} - {self.description}"


class RelationshipSchemaBankFileData(BaseModel):
    """Schema for saving/loading the entire relationship schema bank"""

    relationship_schemas: Dict[str, RelationshipSchemaEntry]
    last_updated: datetime


class RelationshipSchemaBank:
    """
    Maintains a registry of relationship schemas to avoid duplicates and ensure consistency.
    Stores relationship types with their semantic roles and categories.
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        state: State,
        bank_file: str = "relationship_schema_bank.json",
    ):
        self.llm = llm
        self.model = model
        self.state = state
        self.bank_file = bank_file
        self.embedding_service = get_embedding_service()

        # Registry of relationship schemas
        self.relationship_schemas: Dict[str, RelationshipSchemaEntry] = {}

        # KNN search for schema similarity
        self.schema_search = KNNEntitySearch[RelationshipSchemaEntry]()

        # Load existing bank if it exists
        self.load_bank()

        # Pre-populate with core schemas if bank is empty
        if not self.relationship_schemas:
            self._initialize_core_schemas()

    def get_or_create_relationship_schema(
        self,
        proposed_type: str,
        proposed_roles: List[str],
        description: str,
        category: str,
        examples: List[str],
        context: str = "",
    ) -> RelationshipSchemaMatch:
        """
        Get existing relationship schema or create new one if appropriate.

        Args:
            proposed_type: The relationship type the LLM wants to use
            proposed_roles: List of semantic roles for this relationship
            description: Description of what this relationship means
            category: Semantic category (state, action, transfer, comparative, etc.)
            examples: Example usage of this relationship
            context: Additional context about the relationship

        Returns:
            RelationshipSchemaMatch with decision and details
        """

        if not self.relationship_schemas:
            # First relationship schema - just create it
            self._create_new_relationship_schema(
                proposed_type, proposed_roles, description, category, examples
            )
            return RelationshipSchemaMatch(
                use_existing=False,
                relationship_type=proposed_type,
                semantic_roles=proposed_roles,
                reasoning="First relationship schema in the bank",
            )

        # Check against existing schemas
        match = self._find_matching_relationship_schema(
            proposed_type, proposed_roles, description, category, examples, context
        )

        if match.use_existing:
            # Use existing schema
            if match.relationship_type in self.relationship_schemas:
                existing_entry = self.relationship_schemas[match.relationship_type]
                existing_entry.usage_count += 1
                existing_entry.last_used = datetime.now()

                # Add new examples if they're different
                for example in examples:
                    if example not in existing_entry.examples:
                        existing_entry.examples.append(example)

                self.save_bank()

                logger.info(
                    f"Using existing relationship schema: {match.relationship_type} (reason: {match.reasoning})"
                )
            else:
                logger.warning(
                    f"LLM suggested using existing schema '{match.relationship_type}' but it doesn't exist. Creating new schema."
                )
                self._create_new_relationship_schema(
                    match.relationship_type,
                    match.semantic_roles,
                    description,
                    category,
                    examples,
                )

            return match
        else:
            # Create new schema
            logger.info(
                f"Creating new relationship schema: {match.relationship_type} (reason: {match.reasoning})"
            )
            self._create_new_relationship_schema(
                match.relationship_type,
                match.semantic_roles,
                description,
                category,
                examples,
            )
            return match

    def _find_matching_relationship_schema(
        self,
        proposed_type: str,
        proposed_roles: List[str],
        description: str,
        category: str,
        examples: List[str],
        context: str,
    ) -> RelationshipSchemaMatch:
        """Find matching relationship schema using both embedding and LLM analysis"""

        # First try embedding-based similarity
        embedding_match = self._find_similar_schema_by_embedding(
            proposed_type, proposed_roles, description, category
        )
        if embedding_match:
            return embedding_match

        # Fall back to LLM-based matching
        return self._llm_find_matching_schema(
            proposed_type, proposed_roles, description, category, examples, context
        )

    def _llm_find_matching_schema(
        self,
        proposed_type: str,
        proposed_roles: List[str],
        description: str,
        category: str,
        examples: List[str],
        context: str,
    ) -> RelationshipSchemaMatch:
        """Use LLM to find matching relationship schema"""

        schemas_list = []
        for name, entry in self.relationship_schemas.items():
            examples_text = "; ".join(entry.examples[:3])
            roles_text = ", ".join(entry.semantic_roles)
            schemas_list.append(
                f"- {name} ({entry.category}): {roles_text} - {entry.description}. Examples: {examples_text}"
            )

        schemas_text = "\n".join(schemas_list)
        examples_text = "; ".join(examples)
        roles_text = ", ".join(proposed_roles)

        prompt = f"""I want to use a relationship type "{proposed_type}" with roles [{roles_text}] in category "{category}".
Description: {description}
Examples: {examples_text}
Context: {context}

EXISTING SCHEMAS:
{schemas_text}

Should I use an existing schema or create a new one?

SCHEMA MATCHING CRITERIA:
1. **Category Match**: Same semantic category is strongly preferred (state vs action vs transfer vs comparative)
2. **Role Structure**: Similar number and types of semantic roles 
3. **Semantic Meaning**: Similar relationship meaning within the category
4. **Role Compatibility**: Can the proposed roles map reasonably to existing roles?

Examples of good matches:
- prefers(agent, preferred, compared_to) ↔ favors(experiencer, favored, over) [same preference category, similar roles]
- gave(agent, object, beneficiary) ↔ sent(sender, item, recipient) [same transfer category, mappable roles]

Examples of bad matches:
- prefers(agent, preferred, compared_to) ↔ chooses(agent, selected, alternative) [preference state vs action]
- likes(agent, object) ↔ prefers(agent, preferred, compared_to) [simple vs comparative structure]

If you choose an existing schema, provide role mapping if the role names differ.
Choose the best relationship schema to use and explain why."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=RelationshipSchemaMatch,
                model=self.model,
                llm=self.llm,
                caller="relationship_schema_matching",
                temperature=0.1,
            )
            return result
        except Exception as e:
            logger.error(f"Schema matching failed: {e}")
            # Fallback to creating new schema
            return RelationshipSchemaMatch(
                use_existing=False,
                relationship_type=proposed_type,
                semantic_roles=proposed_roles,
                reasoning="Schema matching failed, creating new schema",
            )

    def _find_similar_schema_by_embedding(
        self,
        proposed_type: str,
        proposed_roles: List[str],
        description: str,
        category: str,
    ) -> Optional[RelationshipSchemaMatch]:
        """Find similar schema using KNN search"""

        if not self.relationship_schemas:
            return None

        # Create text representation for KNN search
        roles_text = ", ".join(proposed_roles)
        proposed_text = f"{proposed_type} ({category}): {roles_text} - {description}"

        try:
            # Use KNN search to find best match
            best_match = self.schema_search.find_best_match(proposed_text)

            if not best_match:
                return None

            similarity = best_match.similarity
            existing_entry = best_match.t

            # High similarity threshold for auto-acceptance
            if similarity >= 0.85:
                logger.info(
                    f"Schema KNN match: '{proposed_type}' → '{existing_entry.name}' (similarity: {similarity:.3f})"
                )
                return RelationshipSchemaMatch(
                    use_existing=True,
                    relationship_type=existing_entry.name,
                    semantic_roles=existing_entry.semantic_roles,
                    reasoning=f"High KNN similarity ({similarity:.3f}) indicates same concept",
                )

            # Very low similarity threshold for auto-rejection
            if similarity < 0.3:
                logger.info(
                    f"Schema KNN reject: '{proposed_type}' best match '{existing_entry.name}' (similarity: {similarity:.3f}) - creating new"
                )
                return RelationshipSchemaMatch(
                    use_existing=False,
                    relationship_type=proposed_type,
                    semantic_roles=proposed_roles,
                    reasoning=f"Low KNN similarity ({similarity:.3f}) indicates new concept",
                )

            # Medium similarity - log for LLM fallback
            logger.info(
                f"Schema KNN uncertain: '{proposed_type}' vs '{existing_entry.name}' (similarity: {similarity:.3f}) - using LLM"
            )

        except Exception as e:
            logger.warning(f"Schema KNN search failed: {e}")

        # Medium similarity - let LLM decide
        return None

    def _create_new_relationship_schema(
        self,
        schema_type: str,
        roles: List[str],
        description: str,
        category: str,
        examples: List[str],
    ) -> None:
        """Create and store a new relationship schema"""

        entry = RelationshipSchemaEntry(
            name=schema_type,
            description=description,
            semantic_roles=roles,
            category=category,
            examples=examples,
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
        )

        # Generate embedding
        self._generate_schema_embedding(entry)

        # Store in registry
        self.relationship_schemas[schema_type] = entry

        # Add to KNN search index
        self.schema_search.add_entity(entry)

        # Save to file
        self.save_bank()

        logger.info(
            f"Created new relationship schema: {schema_type} with roles {roles}"
        )

    def _generate_schema_embedding(self, schema_entry: RelationshipSchemaEntry) -> None:
        """Generate embedding for a relationship schema entry"""

        roles_text = ", ".join(schema_entry.semantic_roles)
        text_for_embedding = f"{schema_entry.name} ({schema_entry.category}): {roles_text} - {schema_entry.description}"

        try:
            embedding = self.embedding_service.encode(text_for_embedding)
            if embedding:
                schema_entry.embedding = embedding
                logger.debug(f"Generated embedding for schema '{schema_entry.name}'")
            else:
                logger.warning(
                    f"Failed to generate embedding for schema '{schema_entry.name}'"
                )
        except Exception as e:
            logger.warning(
                f"Error generating embedding for schema '{schema_entry.name}': {e}"
            )

    # Cosine similarity method removed - now using KNN search

    def get_relationship_stats(self) -> RelationshipBankStats:
        """Get statistics about relationship schemas"""

        if not self.relationship_schemas:
            return RelationshipBankStats(total_schemas=0, total_usage=0)

        total_usage = sum(
            entry.usage_count for entry in self.relationship_schemas.values()
        )
        most_used_entry = max(
            self.relationship_schemas.values(), key=lambda x: x.usage_count
        )

        schemas_info = []
        for entry in self.relationship_schemas.values():
            schemas_info.append(
                RelationshipSchemaInfo(
                    name=entry.name,
                    usage_count=entry.usage_count,
                    description=entry.description,
                    semantic_roles=entry.semantic_roles,
                    category=entry.category,
                )
            )

        return RelationshipBankStats(
            total_schemas=len(self.relationship_schemas),
            total_usage=total_usage,
            most_used_schema=most_used_entry.name,
            most_used_count=most_used_entry.usage_count,
            schemas=schemas_info,
        )

    def save_bank(self):
        """Save relationship schema bank to file"""

        try:
            data = RelationshipSchemaBankFileData(
                relationship_schemas=self.relationship_schemas,
                last_updated=datetime.now(),
            )

            with open(self.bank_file, "w") as f:
                f.write(data.model_dump_json(indent=2))

            logger.debug(f"Saved relationship schema bank to {self.bank_file}")
        except Exception as e:
            logger.error(f"Error saving relationship schema bank: {e}")

    def load_bank(self):
        """Load relationship schema bank from file"""

        try:
            with open(self.bank_file, "r") as f:
                data_dict = json.load(f)

            # Convert back to objects
            self.relationship_schemas = {}
            for name, entry_dict in data_dict.get("relationship_schemas", {}).items():
                # Parse datetime strings
                if "created_at" in entry_dict and isinstance(
                    entry_dict["created_at"], str
                ):
                    entry_dict["created_at"] = datetime.fromisoformat(
                        entry_dict["created_at"].replace("Z", "+00:00")
                    )
                if "last_used" in entry_dict and isinstance(
                    entry_dict["last_used"], str
                ):
                    entry_dict["last_used"] = datetime.fromisoformat(
                        entry_dict["last_used"].replace("Z", "+00:00")
                    )

                entry = RelationshipSchemaEntry(**entry_dict)
                self.relationship_schemas[name] = entry

                # Add to KNN search index
                self.schema_search.add_entity(entry)

            logger.info(
                f"Loaded {len(self.relationship_schemas)} relationship schemas from {self.bank_file}"
            )

        except FileNotFoundError:
            logger.info(
                f"No existing relationship schema bank found at {self.bank_file}, starting fresh"
            )
        except Exception as e:
            logger.error(f"Error loading relationship schema bank: {e}")
            logger.info("Starting with empty relationship schema bank")

    def _initialize_core_schemas(self) -> None:
        """Initialize core relationship schemas that we know we'll need"""

        core_schemas = [
            # Experience-Entity involvement (high confidence structural relationship)
            {
                "name": "involves",
                "description": "An experience or event involves a participant entity",
                "semantic_roles": ["experiencer", "entity"],
                "category": "action",
                "examples": [
                    "involves(experiencer=experience, entity=person)",
                    "involves(experiencer=experience, entity=concept)",
                    "involves(experiencer=experience, entity=emotion)",
                ],
            },
            # Temporal relationships (perfect confidence from timestamps)
            {
                "name": "happened_before",
                "description": "One event occurred before another in time",
                "semantic_roles": ["earlier", "later"],
                "category": "temporal",
                "examples": ["happened_before(earlier=event1, later=event2)"],
            },
            # Common high-confidence relationships
            {
                "name": "caused",
                "description": "One entity causes or triggers another entity or state",
                "semantic_roles": ["cause", "effect"],
                "category": "causal",
                "examples": ["caused(cause=person, effect=emotion)"],
            },
            {
                "name": "prefers",
                "description": "An entity has a preference for one thing over another",
                "semantic_roles": ["agent", "preferred", "compared_to"],
                "category": "state",
                "examples": [
                    "prefers(agent=person, preferred=option1, compared_to=option2)"
                ],
            },
            {
                "name": "creates",
                "description": "An entity creates or produces another entity",
                "semantic_roles": ["creator", "created"],
                "category": "action",
                "examples": ["creates(creator=person, created=object)"],
            },
        ]

        for schema_def in core_schemas:
            entry = RelationshipSchemaEntry(
                name=schema_def["name"],
                description=schema_def["description"],
                semantic_roles=schema_def["semantic_roles"],
                category=schema_def["category"],
                examples=schema_def["examples"],
                usage_count=0,  # Will increment when used
                created_at=datetime.now(),
                last_used=datetime.now(),
            )

            # Generate embedding for the schema
            self._generate_schema_embedding(entry)

            self.relationship_schemas[entry.name] = entry

            # Add to KNN search index
            self.schema_search.add_entity(entry)

        logger.info(f"Initialized {len(core_schemas)} core relationship schemas")

        # Save the bank with core schemas
        self.save_bank()

    def get_builtin_schema(self, schema_name: str) -> Optional[RelationshipSchemaMatch]:
        """
        Get a builtin schema directly without expensive matching process.

        Use this for high-confidence relationships where we know exactly
        what schema we want to avoid similarity matching overhead.
        """
        if schema_name in self.relationship_schemas:
            entry = self.relationship_schemas[schema_name]

            # Update usage statistics
            entry.usage_count += 1
            entry.last_used = datetime.now()

            return RelationshipSchemaMatch(
                use_existing=True,
                relationship_type=entry.name,
                semantic_roles=entry.semantic_roles,
                role_mapping=None,  # No mapping needed for direct access
                reasoning=f"Direct access to builtin schema '{schema_name}'",
            )

        logger.warning(f"Builtin schema '{schema_name}' not found")
        return None
