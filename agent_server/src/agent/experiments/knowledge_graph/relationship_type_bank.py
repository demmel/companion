#!/usr/bin/env python3
"""
Relationship Type Bank

Maintains a registry of relationship types with descriptions to avoid
synonymous duplicates and provide consistency in relationship naming.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.state import State

logger = logging.getLogger(__name__)


class RelationshipTypeMatch(BaseModel):
    """Result of relationship type matching"""

    use_existing: bool = Field(
        description="Whether to use an existing relationship type"
    )
    relationship_type: str = Field(description="The relationship type to use")
    reasoning: str = Field(description="Why this choice was made")


class RelationshipTypeEntry(BaseModel):
    """Entry in the relationship type bank"""

    name: str
    description: str
    examples: List[str]
    usage_count: int
    created_at: datetime
    last_used: datetime


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
    ) -> str:
        """
        Get existing relationship type or create new one if appropriate.

        Args:
            proposed_type: The relationship type the LLM wants to use
            description: Description of what this relationship means
            source_entity: Name of source entity
            target_entity: Name of target entity
            context: Additional context about the relationship

        Returns:
            The relationship type to actually use
        """

        if not self.relationship_types:
            # First relationship type - just create it
            return self._create_new_relationship_type(
                proposed_type, description, source_entity, target_entity
            )

        # Check against existing types
        match = self._find_matching_relationship_type(
            proposed_type, description, source_entity, target_entity, context
        )

        if match.use_existing:
            # Use existing type
            existing_entry = self.relationship_types[match.relationship_type]
            existing_entry.usage_count += 1
            existing_entry.last_used = datetime.now()

            # Add new example if it's different
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
            return match.relationship_type
        else:
            # Create new type
            logger.info(
                f"Creating new relationship type: {match.relationship_type} (reason: {match.reasoning})"
            )
            return self._create_new_relationship_type(
                match.relationship_type, description, source_entity, target_entity
            )

    def _find_matching_relationship_type(
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

Choose the best relationship type to use and explain why."""

        try:
            result = direct_structured_llm_call(
                prompt=prompt,
                response_model=RelationshipTypeMatch,
                model=self.model,
                llm=self.llm,
                caller="relationship_type_matching",
                temperature=0.2,
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
    ) -> str:
        """Create a new relationship type entry"""

        example = f"{source_entity} --[{rel_type}]--> {target_entity}"

        entry = RelationshipTypeEntry(
            name=rel_type,
            description=description,
            examples=[example],
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
        )

        self.relationship_types[rel_type] = entry
        self.save_bank()

        return rel_type

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
        result = bank.get_or_create_relationship_type(proposed, desc, source, target)
        print(f"  Result: {result}")

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
