"""
Agent Evaluation Framework

Hierarchical evaluation and optimization system:
- Level 1: AgentEvaluator (runs conversations to test prompts)
- Level 2: PromptOptimizer (optimizes prompts using AgentEvaluator)
- Level 3: MutationPromptOptimizer (optimizes the mutation prompts)
"""

from .agent_evaluator import AgentEvaluator
from .base import EvaluationResult, EvaluationConfig, DomainEvaluationConfig
from .domains import RoleplayEvaluationConfig

__all__ = [
    "AgentEvaluator",
    "EvaluationResult",
    "EvaluationConfig",
    "DomainEvaluationConfig",
    "RoleplayEvaluationConfig",
]
