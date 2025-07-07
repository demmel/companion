"""
Level 3: Meta-Prompt Optimizer

Optimizes the optimization process itself using simulated annealing over optimization trends.
Learns from patterns in Level 2 optimization runs to improve mutation prompts, interruption
strategies, and other meta-parameters.
"""

import json
import math
import random
import time
import copy
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field

from agent.eval.optimization_paths import OptimizationPathManager
from agent.eval.structured_llm import structured_llm_call, StructuredLLMError
from agent.eval.prompt_optimizer import (
    IntelligentPromptOptimizer,
    OptimizationRunResult,
)
from agent.eval.base import DomainEvaluationConfig


class OptimizationTrendAnalysis(BaseModel):
    """Analysis of optimization trends across multiple runs"""

    success_patterns: List[str] = Field(
        description="Patterns associated with successful optimizations"
    )
    failure_patterns: List[str] = Field(
        description="Patterns associated with failed or poor optimizations"
    )
    mutation_effectiveness: Dict[str, float] = Field(
        description="Effectiveness of different mutation types"
    )
    interruption_effectiveness: Dict[str, float] = Field(
        description="Effectiveness of different interruption strategies"
    )
    convergence_patterns: List[str] = Field(
        description="Patterns in how optimizations converge or fail to converge"
    )
    meta_insights: List[str] = Field(
        description="Higher-level insights about the optimization process"
    )
    confidence: float = Field(
        description="Confidence in the trend analysis", ge=0.0, le=1.0
    )


class MetaOptimizationStrategy(BaseModel):
    """Strategy for meta-optimization with specific parameters"""

    mutation_prompt_template: str = Field(
        description="Enhanced template for mutation prompts"
    )
    interruption_sensitivity: float = Field(
        description="How sensitive the interruption system should be", ge=0.0, le=1.0
    )
    preference_weight: float = Field(
        description="How much to weight preference alignment vs raw scores",
        ge=0.0,
        le=2.0,
    )
    exploration_temperature: float = Field(
        description="Initial temperature for simulated annealing", ge=0.1, le=2.0
    )
    convergence_patience: int = Field(
        description="Iterations to wait without improvement before stopping",
        ge=3,
        le=20,
    )
    feedback_frequency: str = Field(
        description="Strategy for collecting user feedback: conservative, balanced, aggressive"
    )
    strategy_description: str = Field(
        description="Human-readable description of this strategy"
    )
    expected_benefits: List[str] = Field(
        description="Expected benefits of this strategy"
    )


class MetaOptimizationResult(BaseModel):
    """Result of meta-optimization testing"""

    strategy: MetaOptimizationStrategy = Field(
        description="The strategy that was tested"
    )
    test_runs: List[OptimizationRunResult] = Field(
        description="Results from test optimization runs"
    )
    aggregate_performance: Dict[str, float] = Field(
        description="Aggregated performance metrics"
    )
    strategy_effectiveness: float = Field(
        description="Overall effectiveness score for this strategy", ge=0.0, le=10.0
    )
    user_satisfaction_proxy: float = Field(
        description="Proxy measure for user satisfaction", ge=0.0, le=1.0
    )
    efficiency_score: float = Field(
        description="How efficiently the strategy reaches good results", ge=0.0, le=1.0
    )
    robustness_score: float = Field(
        description="How consistently the strategy performs well", ge=0.0, le=1.0
    )


class StrategyMutationResult(BaseModel):
    """Result of mutating a meta-optimization strategy"""

    improved_strategy: MetaOptimizationStrategy = Field(
        description="The enhanced strategy"
    )
    mutation_type: str = Field(description="Type of mutation performed")
    changes_made: List[str] = Field(description="Specific changes made to the strategy")
    expected_improvements: List[str] = Field(
        description="Expected improvements from these changes"
    )
    mutation_rationale: str = Field(description="Reasoning behind the mutation choices")
    confidence: float = Field(description="Confidence in the mutation", ge=0.0, le=1.0)


class MetaOptimizationHistory(BaseModel):
    """Historical record of meta-optimization runs"""

    schema_version: str = Field(
        default="1.0", description="Schema version for compatibility"
    )
    optimization_runs: List[OptimizationRunResult] = Field(
        default_factory=list, description="All Level 2 optimization runs"
    )
    meta_strategies_tested: List[MetaOptimizationResult] = Field(
        default_factory=list, description="Meta-strategies tested"
    )
    best_strategy: Optional[MetaOptimizationStrategy] = Field(
        default=None, description="Best performing strategy found so far"
    )
    trend_analyses: List[OptimizationTrendAnalysis] = Field(
        default_factory=list, description="Historical trend analyses"
    )
    total_meta_iterations: int = Field(
        default=0, description="Total meta-optimization iterations performed"
    )
    last_updated: float = Field(
        default_factory=time.time, description="Unix timestamp of last update"
    )


class IntelligentMetaOptimizer:
    """Level 3: Meta-optimization system that optimizes the optimization process itself"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        path_manager: OptimizationPathManager,
        model: str = "huihui_ai/mistral-small-abliterated",
    ):

        self.domain_eval_config = domain_eval_config
        self.model = model
        self.path_manager = path_manager

        # Use path manager's meta-history directory and file
        self.history_file = path_manager.paths.meta_optimization_history_file
        self.history = self._load_history()

        # Base Level 2 optimizer for testing strategies
        self.base_optimizer = IntelligentPromptOptimizer(
            domain_eval_config, path_manager
        )

    def _load_history(self) -> MetaOptimizationHistory:
        """Load meta-optimization history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                return MetaOptimizationHistory.model_validate(data)
            except Exception as e:
                print(f"Warning: Could not load meta-optimization history: {e}")

        return MetaOptimizationHistory()

    def _save_history(self):
        """Save history to file"""
        with open(self.history_file, "w") as f:
            json.dump(self.history.model_dump(), f, indent=2)

    def analyze_optimization_trends(self) -> OptimizationTrendAnalysis:
        """Analyze trends across historical optimization runs using LLM"""
        print(
            f"🔍 Analyzing optimization trends from {len(self.history.optimization_runs)} historical runs..."
        )

        if len(self.history.optimization_runs) < 3:
            return OptimizationTrendAnalysis(
                success_patterns=["Insufficient data for pattern analysis"],
                failure_patterns=["Need more optimization runs"],
                mutation_effectiveness={},
                interruption_effectiveness={},
                convergence_patterns=["Limited data available"],
                meta_insights=["Collect more optimization data"],
                confidence=0.0,
            )

        try:
            # Prepare run summaries for analysis
            run_summaries = []
            for run in self.history.optimization_runs[-20:]:  # Analyze last 20 runs
                run_summaries.append(
                    {
                        "prompt_type": run.prompt_type,
                        "success": run.success,
                        "improvement": run.improvement,
                        "iterations": run.iterations,
                        "feedback_sessions_used": run.feedback_sessions_used,
                        "duration": run.duration_seconds,
                        "final_confidence": run.final_confidence,
                        "mutations_attempted": len(run.mutations_attempted),
                        "interruption_decisions": len(run.interruption_log),
                    }
                )

            system_prompt = """Analyze these optimization run patterns to identify what makes optimizations succeed or fail.
            
            Look for:
            - Patterns in successful vs failed optimizations
            - Effective mutation strategies and approaches
            - Interruption timing and effectiveness patterns
            - Convergence behaviors and plateau patterns
            - Meta-level insights about the optimization process
            
            Focus on actionable insights that could improve future optimizations."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze the patterns in these optimization runs to understand what drives success and failure.",
                response_model=OptimizationTrendAnalysis,
                context={
                    "runs": run_summaries,
                    "total_runs": len(self.history.optimization_runs),
                    "domain": self.domain_eval_config.get_evaluation_config().domain_name,
                },
                model=self.model,
            )

            # Store this analysis
            self.history.trend_analyses.append(result)
            self._save_history()

            print(f"   ✅ Analysis complete - confidence: {result.confidence:.2f}")
            print(f"   Success patterns: {len(result.success_patterns)}")
            print(f"   Failure patterns: {len(result.failure_patterns)}")
            print(f"   Meta insights: {len(result.meta_insights)}")

            return result

        except StructuredLLMError as e:
            print(f"   ❌ Trend analysis failed: {e}")
            return OptimizationTrendAnalysis(
                success_patterns=["Analysis failed"],
                failure_patterns=["Could not analyze patterns"],
                mutation_effectiveness={},
                interruption_effectiveness={},
                convergence_patterns=["Error in analysis"],
                meta_insights=[f"Analysis error: {e}"],
                confidence=0.0,
            )

    def generate_baseline_strategy(self) -> MetaOptimizationStrategy:
        """Generate baseline meta-optimization strategy"""
        return MetaOptimizationStrategy(
            mutation_prompt_template="""You are an expert prompt engineer optimizing {prompt_type} prompts.

Create an improved version that addresses the evaluation feedback while aligning with user preferences.

Key requirements:
- Address the specific weakest areas identified
- Incorporate the suggested improvements 
- Align with user preferences and values
- Preserve any template variables like {scenario}
- Make targeted, meaningful improvements

Focus on the most impactful changes that will improve both evaluation scores and user preference alignment.""",
            interruption_sensitivity=0.5,
            preference_weight=1.0,
            exploration_temperature=1.0,
            convergence_patience=5,
            feedback_frequency="balanced",
            strategy_description="Baseline meta-optimization strategy with standard parameters",
            expected_benefits=[
                "Stable optimization behavior",
                "Balanced exploration vs exploitation",
            ],
        )

    def mutate_strategy(
        self,
        strategy: MetaOptimizationStrategy,
        trend_analysis: OptimizationTrendAnalysis,
        iteration: int,
    ) -> StrategyMutationResult:
        """Mutate a meta-optimization strategy based on trend analysis"""

        try:
            system_prompt = """You are an expert at optimizing optimization processes.
            
            Improve this meta-optimization strategy based on the trend analysis of past optimization runs.
            
            Consider:
            - What patterns led to successful optimizations
            - What patterns led to failures
            - How to enhance mutation prompts for better results
            - How to adjust interruption sensitivity and feedback frequency
            - How to balance exploration vs exploitation
            
            Make targeted improvements that address identified weaknesses while preserving strengths."""

            mutation_context = {
                "current_strategy": strategy.model_dump(),
                "trend_analysis": trend_analysis.model_dump(),
                "iteration": iteration,
                "domain": self.domain_eval_config.get_evaluation_config().domain_name,
            }

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Improve this meta-optimization strategy based on the trend analysis.",
                response_model=StrategyMutationResult,
                context=mutation_context,
                model=self.model,
            )

            return result

        except StructuredLLMError as e:
            print(f"Strategy mutation failed: {e}")
            # Fallback mutation
            new_strategy = copy.deepcopy(strategy)

            # Simple random perturbation
            if random.random() < 0.5:
                new_strategy.interruption_sensitivity = max(
                    0.0,
                    min(
                        1.0,
                        strategy.interruption_sensitivity + random.uniform(-0.2, 0.2),
                    ),
                )
            else:
                new_strategy.preference_weight = max(
                    0.0,
                    min(2.0, strategy.preference_weight + random.uniform(-0.3, 0.3)),
                )

            return StrategyMutationResult(
                improved_strategy=new_strategy,
                mutation_type="fallback_random",
                changes_made=["Random parameter adjustment due to LLM error"],
                expected_improvements=["Potential parameter optimization"],
                mutation_rationale=f"Fallback mutation due to error: {e}",
                confidence=0.3,
            )

    def test_strategy(
        self,
        strategy: MetaOptimizationStrategy,
        test_prompt_types: List[str] = ["simulation"],
    ) -> MetaOptimizationResult:
        """Test a meta-optimization strategy with multiple runs"""

        print(f"🧪 Testing strategy: {strategy.strategy_description}")

        # Apply strategy to optimizer configuration
        test_optimizer = IntelligentPromptOptimizer(
            self.domain_eval_config, self.path_manager
        )

        # TODO: Apply strategy parameters to the optimizer
        # This would require extending the Level 2 optimizer to accept strategy parameters
        # For now, we'll run with current configuration and simulate strategy effects

        test_runs = []
        total_improvement = 0.0
        total_efficiency = 0.0
        success_count = 0

        for prompt_type in test_prompt_types:
            try:
                print(f"   Testing with {prompt_type} prompt optimization...")

                run_result = test_optimizer.optimize_prompt(
                    prompt_type=prompt_type, max_iterations=5  # Short test runs
                )

                test_runs.append(run_result)
                total_improvement += run_result.improvement

                # Calculate efficiency as improvement per iteration
                efficiency = (
                    run_result.improvement / run_result.iterations
                    if run_result.iterations > 0
                    else 0
                )
                total_efficiency += efficiency

                if run_result.success:
                    success_count += 1

                # Add this run to our history
                self.history.optimization_runs.append(run_result)

            except Exception as e:
                print(f"   ❌ Test run failed for {prompt_type}: {e}")

        if not test_runs:
            # No successful test runs
            return MetaOptimizationResult(
                strategy=strategy,
                test_runs=[],
                aggregate_performance={"error": 1.0},
                strategy_effectiveness=0.0,
                user_satisfaction_proxy=0.0,
                efficiency_score=0.0,
                robustness_score=0.0,
            )

        # Calculate aggregate metrics
        avg_improvement = total_improvement / len(test_runs)
        avg_efficiency = total_efficiency / len(test_runs)
        success_rate = success_count / len(test_runs)

        # Overall effectiveness combines improvement, efficiency, and success rate
        strategy_effectiveness = (
            avg_improvement * 2 + avg_efficiency * 3 + success_rate * 5
        ) * 2
        strategy_effectiveness = max(0.0, min(10.0, strategy_effectiveness))

        # User satisfaction proxy based on feedback sessions and final confidence
        avg_confidence = sum(run.final_confidence for run in test_runs) / len(test_runs)
        avg_feedback_efficiency = sum(
            run.improvement / max(1, run.feedback_sessions_used) for run in test_runs
        ) / len(test_runs)
        user_satisfaction_proxy = (avg_confidence + avg_feedback_efficiency) / 2

        result = MetaOptimizationResult(
            strategy=strategy,
            test_runs=test_runs,
            aggregate_performance={
                "avg_improvement": avg_improvement,
                "avg_efficiency": avg_efficiency,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
            },
            strategy_effectiveness=strategy_effectiveness,
            user_satisfaction_proxy=user_satisfaction_proxy,
            efficiency_score=avg_efficiency,
            robustness_score=success_rate,
        )

        print(f"   📊 Strategy effectiveness: {strategy_effectiveness:.2f}/10")
        print(f"   Success rate: {success_rate:.2f}")
        print(f"   Avg improvement: {avg_improvement:.2f}")

        return result

    def run_meta_optimization(
        self,
        max_iterations: int = 10,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.9,
    ) -> MetaOptimizationResult:
        """Run complete meta-optimization to find the best optimization strategy"""

        print(f"\n🚀 META-PROMPT OPTIMIZER (LEVEL 3)")
        print(f"Optimizing the optimization process itself...")

        # Analyze current trends
        trend_analysis = self.analyze_optimization_trends()

        # Initialize with baseline or best known strategy
        if self.history.best_strategy:
            current_strategy = self.history.best_strategy
            print(f"Starting with best known strategy")
        else:
            current_strategy = self.generate_baseline_strategy()
            print(f"Starting with baseline strategy")

        # Initial evaluation
        print(f"\n📊 Evaluating current strategy...")
        current_result = self.test_strategy(current_strategy)
        best_strategy = current_strategy
        best_result = current_result

        print(f"Baseline effectiveness: {current_result.strategy_effectiveness:.2f}/10")

        # Meta-optimization loop
        temperature = initial_temperature
        iterations_without_improvement = 0

        for iteration in range(max_iterations):
            print(
                f"\n🔄 Meta-iteration {iteration + 1}/{max_iterations} (T={temperature:.3f})"
            )

            # Generate strategy mutation
            mutation_result = self.mutate_strategy(
                current_strategy, trend_analysis, iteration
            )

            print(f"  Mutation: {mutation_result.mutation_type}")
            print(f"  Changes: {', '.join(mutation_result.changes_made[:2])}")
            print(
                f"  Expected improvements: {', '.join(mutation_result.expected_improvements[:2])}"
            )

            # Test mutated strategy
            mutated_result = self.test_strategy(mutation_result.improved_strategy)

            effectiveness_change = (
                mutated_result.strategy_effectiveness
                - current_result.strategy_effectiveness
            )

            print(
                f"  Effectiveness: {mutated_result.strategy_effectiveness:.2f} (change: {effectiveness_change:+.2f})"
            )

            # Simulated annealing decision
            if effectiveness_change > 0:
                accept = True
                reason = "improvement"
                iterations_without_improvement = 0
            else:
                acceptance_prob = (
                    math.exp(effectiveness_change / temperature)
                    if temperature > 0
                    else 0
                )
                accept = random.random() < acceptance_prob
                reason = f"prob={acceptance_prob:.3f}"
                if not accept:
                    iterations_without_improvement += 1

            if accept:
                current_strategy = mutation_result.improved_strategy
                current_result = mutated_result

                if (
                    current_result.strategy_effectiveness
                    > best_result.strategy_effectiveness
                ):
                    best_strategy = current_strategy
                    best_result = current_result
                    print(f"  ✅ NEW BEST: {best_result.strategy_effectiveness:.2f}")
                else:
                    print(
                        f"  ✅ Accepted: {current_result.strategy_effectiveness:.2f} ({reason})"
                    )
            else:
                print(
                    f"  ❌ Rejected: {mutated_result.strategy_effectiveness:.2f} ({reason})"
                )

            # Store tested strategy
            self.history.meta_strategies_tested.append(mutated_result)

            # Cool down
            temperature *= cooling_rate

            # Early stopping
            if iterations_without_improvement >= 5:
                print(f"  🛑 No improvement for 5 iterations, stopping")
                break

        # Update history with best strategy
        self.history.best_strategy = best_strategy
        self.history.total_meta_iterations += iteration + 1
        self.history.last_updated = time.time()
        self._save_history()

        print(f"\n🎯 META-OPTIMIZATION COMPLETE")
        print(
            f"Best strategy effectiveness: {best_result.strategy_effectiveness:.2f}/10"
        )
        print(f"Strategy description: {best_strategy.strategy_description}")
        print(f"Total meta-iterations: {iteration + 1}")

        # Save the best strategy configuration
        self.path_manager.save_best_strategy(best_strategy.model_dump())
        print(f"💾 Saved best strategy to {self.path_manager.paths.best_strategy_file}")

        return best_result

    def get_meta_summary(self) -> Dict[str, Any]:
        """Get summary of meta-optimization progress"""
        return {
            "total_optimization_runs": len(self.history.optimization_runs),
            "total_meta_iterations": self.history.total_meta_iterations,
            "strategies_tested": len(self.history.meta_strategies_tested),
            "trend_analyses_performed": len(self.history.trend_analyses),
            "has_best_strategy": self.history.best_strategy is not None,
            "best_strategy_effectiveness": (
                self.history.meta_strategies_tested[-1].strategy_effectiveness
                if self.history.meta_strategies_tested
                else 0.0
            ),
            "schema_version": self.history.schema_version,
            "last_updated": self.history.last_updated,
        }


def main():
    """Test the meta-prompt optimizer"""
    print("=== META-PROMPT OPTIMIZER (LEVEL 3) TEST ===")

    # Import domain config
    from agent.eval.domains.roleplay import RoleplayEvaluationConfig
    from agent.eval.optimization_paths import OptimizationPathManager

    domain_config = RoleplayEvaluationConfig()

    # Create path manager
    path_manager = OptimizationPathManager(
        base_dir="test_meta_optimization", domain="roleplay"
    )

    # Create meta-optimizer
    meta_optimizer = IntelligentMetaOptimizer(domain_config, path_manager)

    # Test trend analysis first (simpler)
    print("\n🧪 Testing trend analysis...")
    trend_analysis = meta_optimizer.analyze_optimization_trends()
    print(f"   Confidence: {trend_analysis.confidence:.2f}")
    print(f"   Success patterns: {len(trend_analysis.success_patterns)}")
    print(f"   Meta insights: {len(trend_analysis.meta_insights)}")

    # Test strategy generation
    print("\n🧪 Testing strategy generation...")
    baseline_strategy = meta_optimizer.generate_baseline_strategy()
    print(f"   Strategy: {baseline_strategy.strategy_description}")
    print(f"   Temperature: {baseline_strategy.exploration_temperature}")
    print(f"   Feedback freq: {baseline_strategy.feedback_frequency}")

    # Test strategy mutation
    print("\n🧪 Testing strategy mutation...")
    mutation_result = meta_optimizer.mutate_strategy(
        baseline_strategy, trend_analysis, 1
    )
    print(f"   Mutation type: {mutation_result.mutation_type}")
    print(f"   Changes: {len(mutation_result.changes_made)}")
    print(f"   Confidence: {mutation_result.confidence:.2f}")

    # Show meta-summary
    print(f"\n📈 Meta-Summary:")
    summary = meta_optimizer.get_meta_summary()
    print(f"  Total optimization runs: {summary['total_optimization_runs']}")
    print(f"  Total meta-iterations: {summary['total_meta_iterations']}")
    print(f"  Strategies tested: {summary['strategies_tested']}")
    print(f"  Has best strategy: {summary['has_best_strategy']}")

    print("\n✅ Meta-prompt optimizer test complete!")
    print("\nNote: Full meta-optimization run with Level 2 testing would take longer.")


if __name__ == "__main__":
    main()
