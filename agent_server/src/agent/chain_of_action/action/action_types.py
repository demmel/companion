"""
Action types that define what operations the agent can perform.
"""

from enum import Enum


class ActionType(str, Enum):
    """Available action types for the agent"""

    # Cognitive Actions
    THINK = "think"  # Process emotional reactions and analyze situation

    # State Management Actions
    UPDATE_MOOD = "update_mood"  # Change mood
    UPDATE_APPEARANCE = "update_appearance"  # Visual changes
    UPDATE_ENVIRONMENT = "update_environment"  # Setting changes
    # ADD_MEMORY = "add_memory"          # Store important details
    # REMOVE_MEMORY = "remove_memory"    # Forget specific memories
    ADD_PRIORITY = "add_priority"  # Add new priority
    REMOVE_PRIORITY = "remove_priority"  # Remove/complete priority

    # Communication Actions
    SPEAK = "speak"  # Generate conversational response

    # Information Actions
    FETCH_URL = "fetch_url"  # Fetch and analyze content from a web URL
    SEARCH_WEB = "search_web"  # Search the web for information

    # Meta Actions
    WAIT = "wait"  # Wait for response before continuing
    GET_CREATIVE_INSPIRATION = (
        "get_creative_inspiration"  # Get random words for creative inspiration
    )
