"""
Action types that define what operations the agent can perform.
"""

from enum import Enum


class ActionType(str, Enum):
    """Available action types for the agent"""
    
    # Cognitive Actions
    THINK = "think"                    # Process emotional reactions and analyze situation
    
    # State Management Actions (commented out until implemented)
    # UPDATE_MOOD = "update_mood"        # Change mood and intensity
    # ADD_MEMORY = "add_memory"          # Store important details
    # REMOVE_MEMORY = "remove_memory"    # Forget specific memories
    # UPDATE_APPEARANCE = "update_appearance"  # Visual changes
    # UPDATE_ENVIRONMENT = "update_environment"  # Setting changes
    # ADD_GOAL = "add_goal"              # Add new goals
    # REMOVE_GOAL = "remove_goal"        # Complete/abandon goals
    # ADD_DESIRE = "add_desire"          # New immediate wants
    # REMOVE_DESIRE = "remove_desire"    # Satisfy/abandon desires
    
    # Communication Actions (commented out until implemented)
    # SPEAK = "speak"                    # Generate conversational response
    
    # Meta Actions (commented out until implemented)
    # DONE = "done"                      # Complete sequence