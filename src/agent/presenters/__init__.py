"""
Presenter system for agent UIs
"""

from abc import ABC, abstractmethod
from typing import Iterator
from agent.agent_events import AgentEvent


class BasePresenter(ABC):
    """Base class for agent presenters"""

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def process_stream(self, user_input: str) -> None:
        """Process the agent event stream and handle presentation"""
        pass


def get_presenter_for_config(config_name: str, agent) -> BasePresenter:
    """Get the appropriate presenter for an agent configuration"""
    if config_name == "roleplay":
        from .roleplay import RoleplayPresenter

        return RoleplayPresenter(agent)
    else:
        from .generic import GenericPresenter

        return GenericPresenter(agent)
