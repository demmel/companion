"""
Character state management for roleplay
"""

import time
import uuid
from typing import Dict, List, Optional, Any


def create_character_state(
    name: str, personality: str, background: str = "", quirks: str = ""
) -> Dict[str, Any]:
    """Create a new character state dictionary"""
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "personality": personality,
        "background": background,
        "quirks": quirks,
        "mood": "neutral",
        "mood_intensity": "moderate",
        "relationship": None,
        "memories": [],
        "recent_actions": [],
        "internal_thoughts": [],
        "created_at": time.time(),
        "last_active": time.time(),
    }


def update_character_mood(
    character: Dict[str, Any], mood: str, intensity: str = "moderate"
) -> None:
    """Update character's mood and intensity"""
    character["mood"] = mood
    character["mood_intensity"] = intensity
    character["last_active"] = time.time()


def add_character_memory(
    character: Dict[str, Any], detail: str, category: str = "general"
) -> None:
    """Add a memory to the character's memory list"""
    memory = {"detail": detail, "category": category, "timestamp": time.time()}
    character["memories"].append(memory)

    # Keep only last 15 memories per character
    if len(character["memories"]) > 15:
        character["memories"] = character["memories"][-15:]

    character["last_active"] = time.time()


def add_character_action(
    character: Dict[str, Any], action: str, reason: str = ""
) -> None:
    """Add an action to the character's recent actions"""
    action_entry = {"action": action, "reason": reason, "timestamp": time.time()}
    character["recent_actions"].append(action_entry)

    # Keep only last 8 actions per character
    if len(character["recent_actions"]) > 8:
        character["recent_actions"] = character["recent_actions"][-8:]

    character["last_active"] = time.time()


def add_internal_thought(character: Dict[str, Any], thought: str) -> None:
    """Add an internal thought to the character"""
    thought_entry = {"thought": thought, "timestamp": time.time()}
    character["internal_thoughts"].append(thought_entry)

    # Keep only last 5 thoughts per character
    if len(character["internal_thoughts"]) > 5:
        character["internal_thoughts"] = character["internal_thoughts"][-5:]

    character["last_active"] = time.time()


def set_character_relationship(
    character: Dict[str, Any], relationship: str, feelings: str = ""
) -> None:
    """Set the character's relationship status with the user"""
    character["relationship"] = {
        "relationship": relationship,
        "feelings": feelings,
        "established_at": time.time(),
    }
    character["last_active"] = time.time()
