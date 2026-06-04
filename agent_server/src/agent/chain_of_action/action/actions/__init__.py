"""
Action implementations.

Importing this package runs every action module, whose ``@register_action`` decorator
populates the action registry and whose ``FooActionData`` class self-collects for the
derived ``ActionData`` union. This is the single import site that makes both registries
authoritative.
"""

from .think_action import ThinkAction
from .wait_action import WaitAction
from .speak_action import SpeakAction
from .update_mood_action import UpdateMoodAction
from .visual_actions import UpdateAppearanceAction, UpdateEnvironmentAction
from .fetch_url_action import FetchUrlAction
from .search_web_action import SearchWebAction
from .priority_actions import AddPriorityAction, RemovePriorityAction
from .evaluate_priorities_action import EvaluatePrioritiesAction
from .creative_inspiration_action import CreativeInspirationAction

__all__ = [
    "ThinkAction",
    "WaitAction",
    "SpeakAction",
    "UpdateMoodAction",
    "UpdateAppearanceAction",
    "UpdateEnvironmentAction",
    "FetchUrlAction",
    "SearchWebAction",
    "AddPriorityAction",
    "RemovePriorityAction",
    "EvaluatePrioritiesAction",
    "CreativeInspirationAction",
]
