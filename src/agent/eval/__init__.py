"""
Agent Evaluation Framework

Hierarchical evaluation and optimization system:
- Level 1: AgentEvaluator (runs conversations to test prompts)
- Level 2: PromptOptimizer (optimizes prompts using AgentEvaluator)
- Level 3: MutationPromptOptimizer (optimizes the mutation prompts)
"""

from .conversation_generator import ConversationGenerator
from .base import EvaluationResult, EvaluationConfig, DomainEvaluationConfig
from .domains import RoleplayEvaluationConfig

__all__ = [
    "ConversationGenerator",
    "EvaluationResult",
    "EvaluationConfig",
    "DomainEvaluationConfig",
    "RoleplayEvaluationConfig",
]
