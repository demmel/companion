#!/usr/bin/env python3
"""
Relationship Type Bank

Maintains a registry of relationship types with descriptions to avoid
synonymous duplicates and provide consistency in relationship naming.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Tuple
import logging

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State
from agent.memory.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class RelationshipTypeMatch(BaseModel):
    """Result of relationship type matching"""

    use_existing: bool = Field(
        description="Whether to use an existing relationship type"
    )
    relationship_type: str = Field(description="The relationship type to use")
    flip_direction: bool = Field(
        default=False,
        description="Whether to flip the source and target entities for this relationship",
    )
    reasoning: str = Field(description="Why this choice was made")


class RelationshipTypeEntry(BaseModel):
    """Entry in the relationship type bank"""

    name: str
    description: str
    examples: List[str]
    usage_count: int
    created_at: datetime
    last_used: datetime
    embedding: Optional[List[float]] = None


class RelationshipTypeBankFileData(BaseModel):
    """Schema for saving/loading the entire relationship type bank"""

    relationship_types: Dict[str, RelationshipTypeEntry]
    last_updated: datetime


class RelationshipTypeBank:
    """
    Maintains a registry of relationship types to avoid duplicates and ensure consistency.
    Starts empty and learns relationship types as the agent creates them.
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        state: State,
        bank_file: str = "relationship_type_bank.json",
    ):
        self.llm = llm
        self.model = model
        self.state = state
        self.bank_file = bank_file
        self.embedding_service = get_embedding_service()

        # Registry of relationship types
        self.relationship_types: Dict[str, RelationshipTypeEntry] = {}

        # Load existing bank if it exists
        self.load_bank()

    def get_or_create_relationship_type(
        self,
        proposed_type: str,
        description: str,
        source_entity: str,
        target_entity: str,
        context: str = "",
    ) -> Tuple[str, bool]:
        """
        Get existing relationship type or create new one if appropriate.

        Args:
            proposed_type: The relationship type the LLM wants to use
            description: Description of what this relationship means
            source_entity: Name of source entity
            target_entity: Name of target entity
            context: Additional context about the relationship

        Returns:
            Tuple of (relationship_type, should_flip_direction)
        """

        if not self.relationship_types:
            # First relationship type - just create it
            self._create_new_relationship_type(
                proposed_type, description, source_entity, target_entity
            )
            return (proposed_type, False)

        # Check against existing types
        match = self._find_matching_relationship_type(
            proposed_type, description, source_entity, target_entity, context
        )

        if match.use_existing:
            # Use existing type (with safety check and directional fallback)
            if match.relationship_type not in self.relationship_types:
                # Check for common directional opposites
                opposite_type = self._find_directional_opposite(match.relationship_type)
                if opposite_type and opposite_type in self.relationship_types:
                    logger.info(
                        f"LLM suggested '{match.relationship_type}' but found directional opposite '{opposite_type}'. Using with flipped direction."
                    )
                    # Use the opposite type with flipped direction
                    existing_entry = self.relationship_types[opposite_type]
                    existing_entry.usage_count += 1
                    existing_entry.last_used = datetime.now()
                    return (opposite_type, not match.flip_direction)  # Flip the flip
                else:
                    logger.warning(
                        f"LLM suggested using existing type '{match.relationship_type}' but it doesn't exist. Creating new type."
                    )
                    self._create_new_relationship_type(
                        match.relationship_type,
                        description,
                        source_entity,
                        target_entity,
                    )
                    return (match.relationship_type, match.flip_direction)

            existing_entry = self.relationship_types[match.relationship_type]
            existing_entry.usage_count += 1
            existing_entry.last_used = datetime.now()

            # Handle direction flipping if needed
            if match.flip_direction:
                # Flip the entities in the example
                new_example = (
                    f"{target_entity} --[{match.relationship_type}]--> {source_entity}"
                )
                logger.info(
                    f"Using existing relationship type with flipped direction: {match.relationship_type} "
                    f"({source_entity} -> {target_entity} becomes {target_entity} -> {source_entity})"
                )
            else:
                new_example = (
                    f"{source_entity} --[{match.relationship_type}]--> {target_entity}"
                )

            if new_example not in existing_entry.examples:
                existing_entry.examples.append(new_example)
                existing_entry.examples = existing_entry.examples[
                    -5:
                ]  # Keep last 5 examples

            logger.info(
                f"Using existing relationship type: {match.relationship_type} (reason: {match.reasoning})"
            )
            return (match.relationship_type, match.flip_direction)
        else:
            # Create new type
            logger.info(
                f"Creating new relationship type: {match.relationship_type} (reason: {match.reasoning})"
            )
            self._create_new_relationship_type(
                match.relationship_type, description, source_entity, target_entity
            )
            return (match.relationship_type, False)

    def _find_matching_relationship_type(
        self,
        proposed_type: str,
        description: str,
        source_entity: str,
        target_entity: str,
        context: str,
    ) -> RelationshipTypeMatch:
        """Use embedding pre-filtering then LLM to find matching relationship type"""

        # First try embedding-based similarity matching
        embedding_match = self._find_similar_relationship_by_embedding(
            proposed_type, description
        )

        if embedding_match:
            return embedding_match

        # Fall back to LLM matching for uncertain cases
        return self._llm_find_matching_relationship_type(
            proposed_type, description, source_entity, target_entity, context
        )

    def _llm_find_matching_relationship_type(
        self,
        proposed_type: str,
        description: str,
        source_entity: str,
        target_entity: str,
        context: str,
    ) -> RelationshipTypeMatch:
        """Use LLM to find matching relationship type or decide to create new one"""

        # Build existing types summary
        existing_types = []
        for name, entry in self.relationship_types.items():
            existing_types.append(f"- {name}: {entry.description}")
            if entry.examples:
                existing_types.append(f"  Examples: {', '.join(entry.examples[:3])}")

        existing_types_text = (
            "\n".join(existing_types) if existing_types else "No existing types"
        )

        prompt = f"""I am {self.state.name}. I'm deciding whether to use an existing relationship type or create a new one.

PROPOSED RELATIONSHIP:
Type: {proposed_type}
Description: {description}
Usage: {source_entity} --[{proposed_type}]--> {target_entity}
Context: {context}

EXISTING RELATIONSHIP TYPES:
{existing_types_text}

Should I use an existing relationship type or create a new one? Consider:
1. Are there existing types that mean essentially the same thing?
2. Would using an existing type maintain consistency?
3. Is the proposed type meaningfully different from existing ones?
4. Would a slight modification of existing type work better?
5. DIRECTIONAL RELATIONSHIPS: For directional relationships (like "causes"/"caused_by", "owns"/"owned_by", "creates"/"created_by"), check if the existing relationship means the same thing but in the opposite direction. If so, you can use the existing type but set flip_direction=true.

IMPORTANT: If you choose an existing relationship type that has the opposite direction from what's needed, set flip_direction=true. For example:
- If I want "A causes B" but existing type is "caused_by" with examples like "X caused_by Y", then use relationship_type="caused_by" with flip_direction=true
- If I want "A gifted_by B" but existing type is "gifts" with examples like "X gifts Y", then use relationship_type="gifts" with flip_direction=true

Choose the best relationship type to use, whether to flip direction, and explain why."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=RelationshipTypeMatch,
                model=self.model,
                llm=self.llm,
                caller="relationship_type_matching",
            )

            return result

        except Exception as e:
            logger.error(f"Relationship type matching failed: {e}")
            # Fallback: create new type
            return RelationshipTypeMatch(
                use_existing=False,
                relationship_type=proposed_type,
                reasoning="LLM matching failed, creating new type as fallback",
            )

    def _create_new_relationship_type(
        self, rel_type: str, description: str, source_entity: str, target_entity: str
    ) -> None:
        """Create a new relationship type entry"""

        example = f"{source_entity} --[{rel_type}]--> {target_entity}"

        # Generate a general conceptual description instead of using the specific instance
        general_description = self._generate_general_description(
            rel_type, description, source_entity, target_entity
        )

        entry = RelationshipTypeEntry(
            name=rel_type,
            description=general_description,
            examples=[example],
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
        )

        # Generate embedding for the new relationship type
        self._generate_relationship_embedding(entry)

        self.relationship_types[rel_type] = entry
        self.save_bank()

    def _generate_general_description(
        self,
        rel_type: str,
        specific_description: str,
        source_entity: str,
        target_entity: str,
    ) -> str:
        """Generate a general, reusable description for a relationship type"""

        prompt = f"""I need to create a general description for the relationship type "{rel_type}".

SPECIFIC EXAMPLE:
{source_entity} --[{rel_type}]--> {target_entity}
Context: {specific_description}

Create a general, reusable description that explains what this relationship type means conceptually, not just this specific instance.

IMPORTANT: Avoid agent-perspective or person-specific language. Relationship types should be neutral and generalizable.

Good examples:
- "causes": "One entity causes, triggers, or brings about another entity or state"
- "owns": "Someone possesses or has ownership of something"  
- "enables": "One thing allows, permits, or makes possible another thing"
- "expresses": "One thing conveys, shows, or represents another thing"

Bad examples (too specific):
- "causes": "David's presence caused me to feel excitement"
- "owns": "David owns the penthouse"

Bad examples (agent-perspective):
- "makes_me_feel": "Something that makes me feel an emotion" (should be "affects" or "causes")
- "David_prefers": "What David prefers" (should be "prefers")
- "my_goal_is_to": "My goal to do something" (grammatically broken)

Write a neutral, general description for "{rel_type}" that would apply to other similar relationships."""

        try:
            from pydantic import BaseModel

            class GeneralDescription(BaseModel):
                description: str

            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=GeneralDescription,
                model=self.model,
                llm=self.llm,
                caller="relationship_type_description",
            )
            return result.description
        except Exception as e:
            logger.warning(
                f"Failed to generate general description for {rel_type}: {e}"
            )
            # Fallback: try to generalize the specific description
            return (
                f"A relationship where one entity {rel_type.replace('_', ' ')} another"
            )

    def _find_similar_relationship_by_embedding(
        self, proposed_type: str, description: str
    ) -> Optional[RelationshipTypeMatch]:
        """Use embedding similarity to find matching relationship types before LLM call"""

        if not self.relationship_types:
            return None

        # Generate embedding for the proposed relationship
        proposed_text = f"{proposed_type}: {description}"
        try:
            proposed_embedding = self.embedding_service.encode(proposed_text)
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for proposed relationship: {e}"
            )
            return None

        # Calculate similarities with existing relationship types
        similarities = []
        for rel_name, rel_entry in self.relationship_types.items():
            if rel_entry.embedding is None:
                # Generate missing embedding
                self._generate_relationship_embedding(rel_entry)

            if rel_entry.embedding:
                try:
                    similarity = self.embedding_service.cosine_similarity(
                        proposed_embedding, rel_entry.embedding
                    )
                    similarities.append((similarity, rel_name, rel_entry))
                except Exception:
                    continue

        if not similarities:
            return None

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[0], reverse=True)
        best_similarity, best_name, best_entry = similarities[0]

        # High similarity threshold for auto-matching
        if best_similarity >= 0.95:
            logger.info(
                f"High embedding similarity ({best_similarity:.3f}) - auto-matching '{proposed_type}' to '{best_name}'"
            )
            return RelationshipTypeMatch(
                use_existing=True,
                relationship_type=best_name,
                flip_direction=False,
                reasoning=f"High embedding similarity ({best_similarity:.3f}) indicates same concept",
            )

        # Very low similarity threshold for auto-rejection
        if best_similarity < 0.3:
            logger.info(
                f"Low embedding similarity ({best_similarity:.3f}) - creating new type '{proposed_type}'"
            )
            return RelationshipTypeMatch(
                use_existing=False,
                relationship_type=proposed_type,
                flip_direction=False,
                reasoning=f"Low embedding similarity ({best_similarity:.3f}) indicates new concept",
            )

        # Medium similarity - let LLM decide
        return None

    def _generate_relationship_embedding(
        self, rel_entry: RelationshipTypeEntry
    ) -> None:
        """Generate embedding for a relationship type entry"""
        try:
            embedding_text = f"{rel_entry.name}: {rel_entry.description}"
            rel_entry.embedding = self.embedding_service.encode(embedding_text)
            logger.debug(f"Generated embedding for relationship type: {rel_entry.name}")
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {rel_entry.name}: {e}")

    def _find_directional_opposite(self, relationship_type: str) -> Optional[str]:
        """Find the directional opposite of a relationship type if it exists"""

        # Common directional opposites
        directional_pairs = {
            "caused_by": "caused",
            "caused": "caused_by",
            "created_by": "created",
            "created": "created_by",
            "owned_by": "owned",
            "owned": "owned_by",
            "given_by": "given",
            "given": "given_by",
            "gifted_by": "gifted_to_me",
            "gifted_to_me": "gifted_by",
            "enabled_by": "enabled",
            "enabled": "enabled_by",
            "triggered_by": "triggered",
            "triggered": "triggered_by",
        }

        return directional_pairs.get(relationship_type)

    def get_relationship_stats(self) -> Dict[str, Any]:
        """Get statistics about relationship types"""
        if not self.relationship_types:
            return {"total_types": 0, "total_usage": 0}

        total_usage = sum(
            entry.usage_count for entry in self.relationship_types.values()
        )
        most_used = max(self.relationship_types.values(), key=lambda x: x.usage_count)

        return {
            "total_types": len(self.relationship_types),
            "total_usage": total_usage,
            "most_used_type": most_used.name,
            "most_used_count": most_used.usage_count,
            "types": [
                {
                    "name": entry.name,
                    "usage_count": entry.usage_count,
                    "description": (
                        entry.description[:50] + "..."
                        if len(entry.description) > 50
                        else entry.description
                    ),
                }
                for entry in sorted(
                    self.relationship_types.values(),
                    key=lambda x: x.usage_count,
                    reverse=True,
                )
            ],
        }

    def save_bank(self):
        """Save relationship type bank to file"""
        try:
            data = RelationshipTypeBankFileData(
                relationship_types=self.relationship_types,
                last_updated=datetime.now(),
            )

            with open(self.bank_file, "w") as f:
                f.write(data.model_dump_json(indent=2))

            logger.debug(
                f"Saved relationship bank with {len(self.relationship_types)} types"
            )

        except Exception as e:
            logger.error(f"Failed to save relationship bank: {e}")

    def load_bank(self):
        """Load relationship type bank from file"""
        try:
            with open(self.bank_file, "r") as f:
                data = f.read()
                data = RelationshipTypeBankFileData.model_validate_json(data)

            self.relationship_types = data.relationship_types

            logger.info(
                f"Loaded relationship bank with {len(self.relationship_types)} types"
            )

        except FileNotFoundError:
            logger.info("No existing relationship bank found, starting empty")
        except Exception as e:
            logger.error(f"Failed to load relationship bank: {e}")


if __name__ == "__main__":
    # Test the relationship type bank
    logging.basicConfig(level=logging.INFO)

    from agent.llm import create_llm, SupportedModel
    from agent.conversation_persistence import ConversationPersistence

    # Load state for testing
    persistence = ConversationPersistence()
    _, state, _ = persistence.load_conversation("baseline")

    if state is None:
        print("❌ Could not load baseline state")
        exit(1)

    # Create bank
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    bank = RelationshipTypeBank(llm, model, state, "test_relationship_bank.json")

    # Test some relationship types
    test_cases = [
        (
            "symbolizes_my_bond_with",
            "Represents a meaningful connection I have with someone",
            "trust choker",
            "David",
        ),
        (
            "reminds_me_of",
            "Brings another concept to mind through association",
            "anime artwork",
            "gaming",
        ),
        (
            "represents_my_connection_to",
            "Shows a meaningful relationship I have",
            "jewelry",
            "David",
        ),  # Should match symbolizes_my_bond_with
        (
            "makes_me_think_of",
            "Causes me to recall or consider something else",
            "music",
            "memories",
        ),  # Should match reminds_me_of
        (
            "located_in",
            "One thing is physically situated within another",
            "gaming setup",
            "apartment",
        ),
    ]

    print("\n🧪 Testing Relationship Type Bank...")

    for proposed, desc, source, target in test_cases:
        print(f"\n  Testing: {source} --[{proposed}]--> {target}")
        rel_type, should_flip = bank.get_or_create_relationship_type(
            proposed, desc, source, target
        )
        print(f"  Result: {rel_type} (flip: {should_flip})")

    # Show final stats
    print(f"\n📊 Final Bank Stats:")
    stats = bank.get_relationship_stats()
    for key, value in stats.items():
        if key == "types":
            print(f"  {key}:")
            for type_info in value:
                print(
                    f"    - {type_info['name']} ({type_info['usage_count']} uses): {type_info['description']}"
                )
        else:
            print(f"  {key}: {value}")

    print(f"\n✅ Relationship type bank test completed!")
