"""
Base classes and data structures for agent evaluation framework
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from agent.config import AgentConfig


@dataclass
class EvaluationResult:
    """Result of evaluating an agent configuration"""

    config_name: str
    scenario: str
    conversation: List[Dict[str, str]]
    scores: Dict[str, float]
    feedback: str
    suggested_improvements: List[str]
    overall_score: float


@dataclass
class EvaluationConfig:
    """Configuration for evaluating a specific domain"""

    domain_name: str
    test_scenarios: List[str]
    initial_prompt_template: str
    simulation_prompt_template: str
    evaluation_prompt_template: str
    evaluation_criteria: List[str]
    num_conversation_turns: int = 6


class DomainEvaluationConfig(ABC):
    """Abstract base for domain-specific evaluation configurations"""

    @abstractmethod
    def get_evaluation_config(self) -> EvaluationConfig:
        """Return the evaluation configuration for this domain"""
        pass

    @abstractmethod
    def get_agent_config(self) -> AgentConfig:
        """Return the agent configuration for this domain (with any mocks applied)"""
        pass

    @abstractmethod
    def extract_conversation_context(self, agent_state: Dict[str, Any]) -> str:
        """Extract relevant context from agent state for evaluation"""
        pass
