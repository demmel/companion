"""
Level 2: Intelligent Prompt Optimizer V3

Complete rewrite using LLM-first principles with structured calls and semantic preferences.
Uses sophisticated semantic understanding instead of keyword matching.
"""

import json
import math
import random
import time
import copy
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from agent.eval.structured_llm import structured_llm_call, StructuredLLMError
from agent.eval.preferences import (
    SemanticPreferenceManager,
    OptimizationGuidance,
    AlignmentAssessment,
)
from agent.eval.feedback_learner import SmartFeedbackLearner
from agent.eval.interruption import (
    IntelligentInterruptionSystem,
    OptimizationContext,
)
from agent.eval.agent_evaluator import AgentEvaluator
from agent.eval.base import DomainEvaluationConfig, EvaluationResult
from agent.eval.optimization_paths import OptimizationPathManager


class PromptMutationResult(BaseModel):
    """Result of LLM-driven prompt mutation"""

    improved_prompt: str = Field(description="The enhanced prompt text")
    mutation_type: str = Field(description="Type of improvement made")
    description: str = Field(description="Detailed explanation of changes made")
    targeted_issues: List[str] = Field(
        description="Specific issues from evaluation that were addressed"
    )
    alignment_improvements: List[str] = Field(
        description="How the changes better align with user preferences"
    )
    confidence: float = Field(
        description="Confidence in the improvement", ge=0.0, le=1.0
    )


class OptimizationRunResult(BaseModel):
    """Complete result of an optimization run"""

    run_id: str = Field(description="Unique identifier for this optimization run")
    prompt_type: str = Field(
        description="Type of prompt optimized: agent, simulation, or evaluation"
    )
    original_prompt: str = Field(description="The original prompt before optimization")
    optimized_prompt: str = Field(
        description="The best prompt found during optimization"
    )
    original_score: float = Field(description="Initial evaluation score")
    optimized_score: float = Field(description="Final best evaluation score")
    improvement: float = Field(description="Total improvement achieved")
    iterations: int = Field(description="Number of optimization iterations performed")
    feedback_sessions_used: int = Field(
        description="Number of times user feedback was collected"
    )
    duration_seconds: float = Field(description="Total optimization time in seconds")
    mutations_attempted: List[Dict[str, Any]] = Field(
        description="Record of all mutation attempts"
    )
    interruption_log: List[Dict[str, Any]] = Field(
        description="Log of interruption decisions"
    )
    final_confidence: float = Field(
        description="Final confidence in learned preferences"
    )
    success: bool = Field(
        description="Whether optimization achieved meaningful improvement"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Unix timestamp of completion"
    )


class RichEvaluationFeedback(BaseModel):
    """Rich feedback combining evaluation results with preference alignment"""

    overall_score: float = Field(description="Overall evaluation score from 0-10")
    criterion_scores: Dict[str, float] = Field(
        description="Scores for individual evaluation criteria"
    )
    detailed_feedback: str = Field(description="Detailed qualitative feedback text")
    suggested_improvements: List[str] = Field(
        description="Specific suggestions for improvement"
    )
    weakest_areas: List[str] = Field(
        description="Areas with the lowest scores needing attention"
    )
    preference_alignment: AlignmentAssessment = Field(
        description="How well content aligns with user preferences"
    )
    evaluation_result: Optional[EvaluationResult] = Field(
        description="Full evaluation result object"
    )


class PromptOptimizationTarget(ABC):
    """Abstract base for different types of prompt optimization"""

    @abstractmethod
    def evaluate_prompt(self, prompt: str) -> RichEvaluationFeedback:
        """Evaluate a prompt and return rich feedback with preference alignment"""
        pass

    @abstractmethod
    def get_baseline_prompt(self) -> str:
        """Get the current/baseline prompt to optimize"""
        pass

    @abstractmethod
    def apply_optimized_prompt(self, prompt: str):
        """Apply the optimized prompt back to the system"""
        pass


class AgentPromptTarget(PromptOptimizationTarget):
    """Optimization target for agent system prompts with rich evaluation"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        preference_manager: SemanticPreferenceManager,
        path_manager: OptimizationPathManager,
        test_scenarios: Optional[List[str]] = None,
    ):
        self.domain_eval_config = domain_eval_config
        self.prefs = preference_manager
        self.path_manager = path_manager
        self.test_scenarios = (
            test_scenarios
            or domain_eval_config.get_evaluation_config().test_scenarios[:2]
        )

    def evaluate_prompt(self, prompt: str) -> RichEvaluationFeedback:
        """Evaluate agent prompt with both automated evaluation and preference alignment"""

        # Run actual agent evaluation
        evaluator = AgentEvaluator(
            domain_eval_config=self.domain_eval_config,
            agent_model="huihui_ai/mistral-small-abliterated",
        )

        evaluation_results = []
        for scenario in self.test_scenarios:
            try:
                result = evaluator.run_evaluation(scenario)
                evaluation_results.append(result)
            except Exception as e:
                print(f"Warning: Evaluation error for scenario '{scenario}': {e}")
                # Continue with other scenarios

        if not evaluation_results:
            # Fallback if all evaluations failed
            from agent.eval.base import EvaluationResult

            fallback_result = EvaluationResult(
                config_name="fallback",
                scenario="error",
                conversation=[],
                scores={"overall": 5.0},
                feedback="Evaluation failed - using fallback",
                suggested_improvements=["Fix evaluation system"],
                overall_score=5.0,
            )
            evaluation_results = [fallback_result]

        # Aggregate evaluation results
        aggregated = self._aggregate_evaluation_results(evaluation_results)

        # Get preference alignment assessment
        alignment = self.prefs.assess_alignment(
            prompt,
            "conversation",
            {"scenarios": self.test_scenarios, "evaluation_type": "agent_prompt"},
        )

        return RichEvaluationFeedback(
            overall_score=aggregated["overall_score"],
            criterion_scores=aggregated["criterion_scores"],
            detailed_feedback=aggregated["detailed_feedback"],
            suggested_improvements=aggregated["suggested_improvements"],
            weakest_areas=aggregated["weakest_areas"],
            preference_alignment=alignment,
            evaluation_result=evaluation_results[0] if evaluation_results else None,
        )

    def _aggregate_evaluation_results(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Aggregate multiple evaluation results"""
        if not results:
            return {
                "overall_score": 5.0,
                "criterion_scores": {},
                "detailed_feedback": "No evaluation results",
                "suggested_improvements": [],
                "weakest_areas": [],
            }

        # Calculate average scores
        all_criteria = set()
        for result in results:
            all_criteria.update(result.scores.keys())

        criterion_scores = {}
        for criterion in all_criteria:
            scores = [
                r.scores.get(criterion, 0)
                for r in results
                if r.scores.get(criterion) is not None
            ]
            if scores:
                criterion_scores[criterion] = sum(scores) / len(scores)

        overall_scores = [
            r.overall_score for r in results if r.overall_score is not None
        ]
        overall_score = (
            sum(overall_scores) / len(overall_scores) if overall_scores else 5.0
        )

        # Aggregate feedback
        all_feedback = [r.feedback for r in results if r.feedback]
        all_suggestions = []
        for r in results:
            all_suggestions.extend(r.suggested_improvements)

        # Find weakest areas
        weakest_areas = [
            criterion for criterion, score in criterion_scores.items() if score < 6.0
        ]

        return {
            "overall_score": overall_score,
            "criterion_scores": criterion_scores,
            "detailed_feedback": " | ".join(all_feedback),
            "suggested_improvements": list(set(all_suggestions)),
            "weakest_areas": weakest_areas,
        }

    def get_baseline_prompt(self) -> str:
        """Get current agent prompt template"""
        agent_config = self.domain_eval_config.get_agent_config()
        return agent_config.prompt_template

    def apply_optimized_prompt(self, prompt: str):
        """Apply optimized agent prompt by saving and updating template"""
        # Save optimized prompt using path manager
        prompt_file = self.path_manager.save_optimized_prompt("agent", prompt)

        # Update the agent config template for future optimization
        agent_config = self.domain_eval_config.get_agent_config()
        agent_config.prompt_template = prompt

        print(f"📝 Saved optimized agent prompt to {prompt_file}")


class SimulationPromptTarget(PromptOptimizationTarget):
    """Optimization target for user simulation prompts"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        preference_manager: SemanticPreferenceManager,
        path_manager: OptimizationPathManager,
    ):
        self.domain_eval_config = domain_eval_config
        self.prefs = preference_manager
        self.path_manager = path_manager
        self.eval_config = domain_eval_config.get_evaluation_config()

    def evaluate_prompt(self, prompt: str) -> RichEvaluationFeedback:
        """Evaluate simulation prompt with LLM-based analysis"""
        try:
            # Use structured LLM call to evaluate simulation quality
            system_prompt = """Evaluate the quality of this user simulation prompt.
            
            Assess:
            - Realism: How realistic are the simulated user behaviors?
            - Coverage: Does it test a good range of scenarios? 
            - Effectiveness: Will it find issues and drive improvement?
            - Clarity: Are the instructions clear and actionable?
            
            Rate each aspect 1-10 and provide specific feedback."""

            class SimulationEvaluation(BaseModel):
                realism_score: float = Field(
                    description="How realistic the simulated user behavior is",
                    ge=1.0,
                    le=10.0,
                )
                coverage_score: float = Field(
                    description="How well it covers different scenarios",
                    ge=1.0,
                    le=10.0,
                )
                effectiveness_score: float = Field(
                    description="How effective it is at finding issues", ge=1.0, le=10.0
                )
                clarity_score: float = Field(
                    description="How clear the instructions are", ge=1.0, le=10.0
                )
                overall_score: float = Field(
                    description="Overall quality score", ge=1.0, le=10.0
                )
                detailed_feedback: str = Field(
                    description="Detailed analysis of the prompt"
                )
                suggested_improvements: List[str] = Field(
                    description="Specific suggestions for improvement"
                )
                weakest_areas: List[str] = Field(
                    description="Areas needing the most improvement"
                )

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=prompt,
                response_model=SimulationEvaluation,
                context={
                    "domain": "simulation",
                    "eval_config": self.eval_config.domain_name,
                    "scenarios": self.eval_config.test_scenarios[:2],
                },
            )

            # Get preference alignment
            alignment = self.prefs.assess_alignment(
                prompt, "simulation", {"evaluation_type": "simulation_prompt"}
            )

            criterion_scores = {
                "realism": result.realism_score,
                "coverage": result.coverage_score,
                "effectiveness": result.effectiveness_score,
                "clarity": result.clarity_score,
            }

            return RichEvaluationFeedback(
                overall_score=result.overall_score,
                criterion_scores=criterion_scores,
                detailed_feedback=result.detailed_feedback,
                suggested_improvements=result.suggested_improvements,
                weakest_areas=result.weakest_areas,
                preference_alignment=alignment,
                evaluation_result=None,
            )

        except StructuredLLMError as e:
            print(f"Error evaluating simulation prompt: {e}")
            # Fallback evaluation
            alignment = self.prefs.assess_alignment(prompt, "simulation", {})
            return RichEvaluationFeedback(
                overall_score=5.0,
                criterion_scores={"error": 5.0},
                detailed_feedback=f"Evaluation error: {e}",
                suggested_improvements=["Fix evaluation system"],
                weakest_areas=["evaluation"],
                preference_alignment=alignment,
                evaluation_result=None,
            )

    def get_baseline_prompt(self) -> str:
        """Get current simulation prompt"""
        return self.eval_config.simulation_prompt_template

    def apply_optimized_prompt(self, prompt: str):
        """Apply optimized simulation prompt by saving and updating template"""
        # Save optimized prompt using path manager
        prompt_file = self.path_manager.save_optimized_prompt("simulation", prompt)

        # Update the eval config template for future optimization
        self.eval_config.simulation_prompt_template = prompt

        print(f"📝 Saved optimized simulation prompt to {prompt_file}")


class EvaluationPromptTarget(PromptOptimizationTarget):
    """Optimization target for evaluation prompts used to score agent conversations"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        preference_manager: SemanticPreferenceManager,
        path_manager: OptimizationPathManager,
    ):
        self.domain_eval_config = domain_eval_config
        self.prefs = preference_manager
        self.path_manager = path_manager
        self.eval_config = domain_eval_config.get_evaluation_config()

    def evaluate_prompt(self, prompt: str) -> RichEvaluationFeedback:
        """Evaluate evaluation prompt by testing how well it actually evaluates conversations"""
        try:
            # Get test conversations for evaluation testing
            test_conversations = self._get_test_conversations()

            if not test_conversations:
                # Generate test conversations if none exist
                test_conversations = self._generate_test_conversations()

            if not test_conversations:
                return self._fallback_evaluation(prompt)

            # Use the evaluation prompt to evaluate each test conversation
            evaluation_results = []
            for conv_data in test_conversations:
                eval_result = self._evaluate_conversation_with_prompt(prompt, conv_data)
                evaluation_results.append(eval_result)

            # Analyze how well the evaluation prompt performed
            analysis = self._analyze_evaluation_performance(
                prompt, evaluation_results, test_conversations
            )

            # Get preference alignment
            alignment = self.prefs.assess_alignment(
                prompt,
                "evaluation",
                {"evaluation_type": "evaluation_prompt", "test_results": analysis},
            )

            return RichEvaluationFeedback(
                overall_score=analysis["overall_quality"],
                criterion_scores=analysis["criterion_scores"],
                detailed_feedback=analysis["detailed_feedback"],
                suggested_improvements=analysis["suggested_improvements"],
                weakest_areas=analysis["weakest_areas"],
                preference_alignment=alignment,
                evaluation_result=None,
            )

        except StructuredLLMError as e:
            print(f"Error evaluating evaluation prompt: {e}")
            # Fallback evaluation
            alignment = self.prefs.assess_alignment(prompt, "evaluation", {})
            return RichEvaluationFeedback(
                overall_score=5.0,
                criterion_scores={"error": 5.0},
                detailed_feedback=f"Evaluation error: {e}",
                suggested_improvements=["Fix evaluation system"],
                weakest_areas=["evaluation"],
                preference_alignment=alignment,
                evaluation_result=None,
            )

    def _get_test_conversations(self) -> List[Dict[str, Any]]:
        """Get test conversations from path manager"""
        return self.path_manager.load_test_conversations()

    def _generate_test_conversations(self) -> List[Dict[str, Any]]:
        """Generate test conversations by running actual agent evaluations"""
        from agent.eval.agent_evaluator import AgentEvaluator

        test_conversations = []
        evaluator = AgentEvaluator(
            domain_eval_config=self.domain_eval_config,
            agent_model="huihui_ai/mistral-small-abliterated",
        )

        # Get test scenarios
        test_scenarios = self.eval_config.test_scenarios[:3]  # Use first 3 scenarios

        for scenario in test_scenarios:
            try:
                # Run evaluation to get actual conversation
                result = evaluator.run_evaluation(scenario)
                test_conversations.append(
                    {
                        "conversation": result.conversation,
                        "scenario": result.scenario,
                        "known_score": result.overall_score,
                        "known_feedback": result.feedback,
                        "source": "generated",
                    }
                )
            except Exception as e:
                print(
                    f"Failed to generate test conversation for scenario '{scenario}': {e}"
                )
                continue

        # Save these for future use
        if test_conversations:
            self.path_manager.save_test_conversations(test_conversations)

        return test_conversations

    def _evaluate_conversation_with_prompt(
        self, eval_prompt: str, conv_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use the evaluation prompt to evaluate a test conversation"""
        try:
            # Format conversation for evaluation
            conversation_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in conv_data["conversation"]
                ]
            )

            # Use the evaluation prompt to score this conversation
            class ConversationEvaluation(BaseModel):
                overall_score: float = Field(
                    description="Overall quality score from 1-10", ge=1.0, le=10.0
                )
                detailed_feedback: str = Field(
                    description="Detailed evaluation feedback"
                )
                criterion_scores: Dict[str, float] = Field(
                    description="Scores for specific criteria"
                )
                strengths: List[str] = Field(
                    description="What the conversation did well"
                )
                weaknesses: List[str] = Field(description="Areas for improvement")

            result = structured_llm_call(
                system_prompt=eval_prompt,
                user_input=f"Evaluate this conversation for scenario: {conv_data['scenario']}\n\nConversation:\n{conversation_text}",
                response_model=ConversationEvaluation,
                context={
                    "scenario": conv_data["scenario"],
                    "domain": self.eval_config.domain_name,
                },
            )

            return {
                "evaluation_result": result,
                "conversation_data": conv_data,
                "success": True,
            }

        except Exception as e:
            return {
                "evaluation_result": None,
                "conversation_data": conv_data,
                "success": False,
                "error": str(e),
            }

    def _analyze_evaluation_performance(
        self,
        eval_prompt: str,
        evaluation_results: List[Dict[str, Any]],
        test_conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze how well the evaluation prompt performed against known evaluations"""

        try:
            # Compare new evaluations against known scores
            performance_data = []
            score_differences = []

            for eval_result in evaluation_results:
                if eval_result["success"]:
                    conv_data = eval_result["conversation_data"]
                    new_eval = eval_result["evaluation_result"]
                    known_score = conv_data["known_score"]

                    score_diff = abs(new_eval.overall_score - known_score)
                    score_differences.append(score_diff)

                    performance_data.append(
                        {
                            "known_score": known_score,
                            "new_score": new_eval.overall_score,
                            "score_difference": score_diff,
                            "feedback_length": len(new_eval.detailed_feedback),
                            "num_strengths": len(new_eval.strengths),
                            "num_weaknesses": len(new_eval.weaknesses),
                            "scenario": conv_data["scenario"],
                        }
                    )

            if not performance_data:
                return self._fallback_analysis()

            # Calculate metrics
            avg_score_diff = sum(score_differences) / len(score_differences)
            consistency_score = max(
                1.0, 10.0 - (avg_score_diff * 2)
            )  # Lower difference = higher consistency

            # Analyze with LLM
            system_prompt = """Analyze how well this evaluation prompt performed compared to known evaluations.
            
            Consider:
            - Score consistency with previous evaluations
            - Quality and specificity of feedback
            - Appropriateness of strengths/weaknesses identified
            - Overall evaluation effectiveness
            
            Rate the evaluation prompt's performance."""

            class EvaluationPerformanceAnalysis(BaseModel):
                overall_quality: float = Field(
                    description="Overall quality of the evaluation prompt",
                    ge=1.0,
                    le=10.0,
                )
                consistency_score: float = Field(
                    description="How consistent with known evaluations", ge=1.0, le=10.0
                )
                feedback_quality: float = Field(
                    description="Quality of feedback provided", ge=1.0, le=10.0
                )
                coverage_score: float = Field(
                    description="How well it covers important aspects", ge=1.0, le=10.0
                )
                detailed_feedback: str = Field(
                    description="Detailed analysis of performance"
                )
                suggested_improvements: List[str] = Field(
                    description="How to improve the evaluation prompt"
                )
                weakest_areas: List[str] = Field(
                    description="Areas where evaluation performed poorly"
                )

            analysis_result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze the performance of this evaluation prompt.",
                response_model=EvaluationPerformanceAnalysis,
                context={
                    "evaluation_prompt": eval_prompt,
                    "performance_data": performance_data,
                    "avg_score_difference": avg_score_diff,
                    "total_tests": len(performance_data),
                },
            )

            return {
                "overall_quality": analysis_result.overall_quality,
                "criterion_scores": {
                    "consistency": analysis_result.consistency_score,
                    "feedback_quality": analysis_result.feedback_quality,
                    "coverage": analysis_result.coverage_score,
                },
                "detailed_feedback": analysis_result.detailed_feedback,
                "suggested_improvements": analysis_result.suggested_improvements,
                "weakest_areas": analysis_result.weakest_areas,
            }

        except Exception as e:
            print(f"Error analyzing evaluation performance: {e}")
            return self._fallback_analysis()

    def _fallback_evaluation(self, prompt: str) -> RichEvaluationFeedback:
        """Fallback evaluation when no test conversations available"""
        alignment = self.prefs.assess_alignment(prompt, "evaluation", {})
        return RichEvaluationFeedback(
            overall_score=5.0,
            criterion_scores={"fallback": 5.0},
            detailed_feedback="No test conversations available for evaluation",
            suggested_improvements=[
                "Generate test conversations for evaluation testing"
            ],
            weakest_areas=["testing"],
            preference_alignment=alignment,
            evaluation_result=None,
        )

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when evaluation testing fails"""
        return {
            "overall_quality": 5.0,
            "criterion_scores": {"error": 5.0},
            "detailed_feedback": "Failed to analyze evaluation performance",
            "suggested_improvements": ["Fix evaluation testing system"],
            "weakest_areas": ["analysis"],
        }

    def get_baseline_prompt(self) -> str:
        """Get current evaluation prompt"""
        return self.eval_config.evaluation_prompt_template

    def apply_optimized_prompt(self, prompt: str):
        """Apply optimized evaluation prompt by saving and updating template"""
        # Save optimized prompt using path manager
        prompt_file = self.path_manager.save_optimized_prompt("evaluation", prompt)

        # Update the eval config template for future optimization
        self.eval_config.evaluation_prompt_template = prompt

        print(f"📝 Saved optimized evaluation prompt to {prompt_file}")


class IntelligentPromptMutator:
    """LLM-driven prompt mutation with sophisticated preference integration"""

    def __init__(self, model: str = "huihui_ai/mistral-small-abliterated"):
        self.model = model

    def mutate_prompt(
        self,
        prompt: str,
        prompt_type: str,
        evaluation_feedback: RichEvaluationFeedback,
        optimization_guidance: OptimizationGuidance,
        iteration: int,
    ) -> PromptMutationResult:
        """Generate intelligent prompt mutation using rich feedback and preferences"""

        try:
            system_prompt = f"""You are an expert prompt engineer optimizing {prompt_type} prompts.
            
            Create an improved version that addresses the evaluation feedback while aligning with user preferences.
            
            Key requirements:
            - Address the specific weakest areas identified
            - Incorporate the suggested improvements 
            - Align with user preferences and values
            - Preserve any template variables like {{scenario}}
            - Make targeted, meaningful improvements
            
            Focus on the most impactful changes that will improve both evaluation scores and user preference alignment."""

            # Build comprehensive context
            context = {
                "current_prompt": prompt,
                "prompt_type": prompt_type,
                "iteration": iteration,
                "evaluation": {
                    "overall_score": evaluation_feedback.overall_score,
                    "criterion_scores": evaluation_feedback.criterion_scores,
                    "detailed_feedback": evaluation_feedback.detailed_feedback,
                    "suggested_improvements": evaluation_feedback.suggested_improvements,
                    "weakest_areas": evaluation_feedback.weakest_areas,
                },
                "preferences": {
                    "guidance": optimization_guidance.guidance,
                    "positive_directions": optimization_guidance.positive_directions,
                    "negative_directions": optimization_guidance.negative_directions,
                    "priorities": optimization_guidance.improvement_priorities,
                },
                "alignment": {
                    "score": evaluation_feedback.preference_alignment.overall_alignment,
                    "conflicts": evaluation_feedback.preference_alignment.negative_conflicts,
                    "suggestions": evaluation_feedback.preference_alignment.improvement_suggestions,
                },
            }

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Create an improved prompt that addresses the evaluation feedback and aligns with user preferences.",
                response_model=PromptMutationResult,
                context=context,
                model=self.model,
            )

            return result

        except StructuredLLMError as e:
            print(f"LLM mutation failed: {e}")
            # Fallback mutation
            return self._fallback_mutation(prompt, evaluation_feedback)

    def _fallback_mutation(
        self, prompt: str, evaluation_feedback: RichEvaluationFeedback
    ) -> PromptMutationResult:
        """Simple fallback mutation when LLM fails"""
        if evaluation_feedback.suggested_improvements:
            improvement = evaluation_feedback.suggested_improvements[0]
            enhanced_prompt = prompt + f"\n\nFocus on: {improvement}"

            return PromptMutationResult(
                improved_prompt=enhanced_prompt,
                mutation_type="fallback_enhancement",
                description=f"Added focus on: {improvement}",
                targeted_issues=[improvement],
                alignment_improvements=["Addressed evaluation suggestion"],
                confidence=0.4,
            )

        return PromptMutationResult(
            improved_prompt=prompt + "\n\nBe thorough and accurate.",
            mutation_type="minimal_fallback",
            description="Minimal enhancement due to LLM error",
            targeted_issues=[],
            alignment_improvements=[],
            confidence=0.2,
        )


class IntelligentPromptOptimizer:
    """Level 2: Complete intelligent prompt optimizer with LLM-first architecture"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        path_manager: OptimizationPathManager,
    ):

        # Core components with LLM-first design
        self.domain_eval_config = domain_eval_config
        self.path_manager = path_manager
        self.prefs = SemanticPreferenceManager(str(path_manager.paths.preferences_dir))
        self.feedback_learner = SmartFeedbackLearner(self.prefs)
        self.interruption_system = IntelligentInterruptionSystem(self.prefs)
        self.mutator = IntelligentPromptMutator()

        # Optimization state
        self.optimization_history: List[OptimizationRunResult] = []

    def optimize_prompt(
        self,
        prompt_type: str,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 15,
    ) -> OptimizationRunResult:
        """Run complete intelligent prompt optimization"""

        print(f"\n🚀 INTELLIGENT PROMPT OPTIMIZATION V3")
        print(f"Type: {prompt_type}")

        # Get current preference confidence
        prefs_summary = self.prefs.get_summary()
        print(
            f"Preference confidence: {prefs_summary['confidence_levels'].get(prompt_type.replace('agent', 'conversation'), 0.0):.2f}"
        )

        # Create optimization target
        target = self._create_optimization_target(prompt_type)

        # Initialize run
        run_id = f"{prompt_type}_{int(time.time())}"
        start_time = time.time()
        original_prompt = target.get_baseline_prompt()

        # Initial evaluation with rich feedback
        print(f"\n📊 Evaluating baseline prompt...")
        current_prompt = original_prompt
        current_evaluation = target.evaluate_prompt(current_prompt)
        best_prompt = current_prompt
        best_evaluation = current_evaluation

        print(f"Baseline score: {current_evaluation.overall_score:.2f}/10")
        print(
            f"Preference alignment: {current_evaluation.preference_alignment.overall_alignment:.2f}"
        )
        if current_evaluation.weakest_areas:
            print(f"Weakest areas: {', '.join(current_evaluation.weakest_areas)}")

        # Optimization loop with intelligent feedback
        temperature = initial_temperature
        score_history = [current_evaluation.overall_score]
        iterations_without_improvement = 0
        feedback_sessions_used = 0
        mutations_attempted = []
        interruption_log = []

        for iteration in range(max_iterations):
            print(
                f"\n🔄 Iteration {iteration + 1}/{max_iterations} (T={temperature:.3f})"
            )

            # Check for feedback interruption using intelligent analysis
            context = OptimizationContext(
                current_iteration=iteration,
                iterations_without_improvement=iterations_without_improvement,
                best_score=best_evaluation.overall_score,
                current_score=current_evaluation.overall_score,
                score_history=score_history,
                confidence_level=prefs_summary["confidence_levels"].get(
                    prompt_type.replace("agent", "conversation"), 0.0
                ),
                feedback_sessions_this_run=feedback_sessions_used,
                optimization_start_time=start_time,
                total_feedback_sessions=prefs_summary["total_feedback_sessions"],
                preference_summary=prefs_summary,
            )

            interruption_decision = self.interruption_system.should_interrupt(context)
            interruption_log.append(
                {
                    "iteration": iteration,
                    "should_interrupt": interruption_decision.should_interrupt,
                    "reason": (
                        interruption_decision.reason.value
                        if interruption_decision.reason
                        else None
                    ),
                    "reasoning": interruption_decision.reasoning,
                    "urgency": interruption_decision.urgency,
                    "confidence": interruption_decision.confidence,
                }
            )

            if (
                interruption_decision.should_interrupt
                and interruption_decision.urgency in ["high", "medium"]
            ):
                print(
                    f"🤔 Interruption Decision ({interruption_decision.urgency} urgency):"
                )
                print(f"   Reason: {interruption_decision.reasoning}")
                print(f"   Expected benefit: {interruption_decision.expected_benefit}")

                # Actually collect user feedback
                self.feedback_learner.start_feedback_session(prompt_type)
                feedback_result = self.feedback_learner.collect_simple_feedback(
                    content=current_prompt,
                    context={
                        "iteration": iteration,
                        "score": current_evaluation.overall_score,
                    },
                    domain=prompt_type.replace("agent", "conversation"),
                )
                self.feedback_learner.end_feedback_session()
                feedback_sessions_used += 1

            # Get optimization guidance from preferences
            guidance = self.prefs.get_optimization_guidance(
                prompt_type.replace("agent", "conversation")
            )

            # Generate mutation using rich feedback
            mutation_result = self.mutator.mutate_prompt(
                current_prompt, prompt_type, current_evaluation, guidance, iteration
            )

            print(f"  Mutation: {mutation_result.mutation_type}")
            print(f"  Description: {mutation_result.description}")
            if mutation_result.targeted_issues:
                print(f"  Targeting: {', '.join(mutation_result.targeted_issues[:2])}")

            # Evaluate mutated prompt
            mutated_evaluation = target.evaluate_prompt(mutation_result.improved_prompt)
            score_change = (
                mutated_evaluation.overall_score - current_evaluation.overall_score
            )
            alignment_change = (
                mutated_evaluation.preference_alignment.overall_alignment
                - current_evaluation.preference_alignment.overall_alignment
            )

            print(
                f"  Score: {mutated_evaluation.overall_score:.2f} (change: {score_change:+.2f})"
            )
            print(
                f"  Alignment: {mutated_evaluation.preference_alignment.overall_alignment:.2f} (change: {alignment_change:+.2f})"
            )

            # Simulated annealing decision (considering both score and alignment)
            combined_change = score_change + (
                alignment_change * 2.0
            )  # Weight alignment more heavily

            if combined_change > 0:
                accept = True
                reason = "improvement"
                iterations_without_improvement = 0
            else:
                acceptance_prob = (
                    math.exp(combined_change / temperature) if temperature > 0 else 0
                )
                accept = random.random() < acceptance_prob
                reason = f"prob={acceptance_prob:.3f}"
                if not accept:
                    iterations_without_improvement += 1

            # Record mutation attempt
            mutation_record = {
                "iteration": iteration,
                "mutation_result": mutation_result.model_dump(),
                "score_before": current_evaluation.overall_score,
                "score_after": mutated_evaluation.overall_score,
                "score_change": score_change,
                "alignment_before": current_evaluation.preference_alignment.overall_alignment,
                "alignment_after": mutated_evaluation.preference_alignment.overall_alignment,
                "alignment_change": alignment_change,
                "combined_change": combined_change,
                "accepted": accept,
                "acceptance_reason": reason,
                "temperature": temperature,
            }
            mutations_attempted.append(mutation_record)

            if accept:
                current_prompt = mutation_result.improved_prompt
                current_evaluation = mutated_evaluation

                if (
                    current_evaluation.overall_score > best_evaluation.overall_score
                    or (
                        current_evaluation.overall_score
                        >= best_evaluation.overall_score - 0.1
                        and current_evaluation.preference_alignment.overall_alignment
                        > best_evaluation.preference_alignment.overall_alignment
                    )
                ):
                    best_prompt = current_prompt
                    best_evaluation = current_evaluation
                    print(
                        f"  ✅ NEW BEST: Score {best_evaluation.overall_score:.2f}, Alignment {best_evaluation.preference_alignment.overall_alignment:.2f}"
                    )
                else:
                    print(
                        f"  ✅ Accepted: {current_evaluation.overall_score:.2f} ({reason})"
                    )
            else:
                print(
                    f"  ❌ Rejected: {mutated_evaluation.overall_score:.2f} ({reason})"
                )

            score_history.append(current_evaluation.overall_score)

            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                print(f"  🥶 Temperature too low, stopping")
                break

        # Finalize results
        duration = time.time() - start_time
        improvement = (
            best_evaluation.overall_score
            - target.evaluate_prompt(original_prompt).overall_score
        )

        result = OptimizationRunResult(
            run_id=run_id,
            prompt_type=prompt_type,
            original_prompt=original_prompt,
            optimized_prompt=best_prompt,
            original_score=target.evaluate_prompt(original_prompt).overall_score,
            optimized_score=best_evaluation.overall_score,
            improvement=improvement,
            iterations=iteration + 1,
            feedback_sessions_used=feedback_sessions_used,
            duration_seconds=duration,
            mutations_attempted=mutations_attempted,
            interruption_log=interruption_log,
            final_confidence=prefs_summary["confidence_levels"].get(
                prompt_type.replace("agent", "conversation"), 0.0
            ),
            success=improvement > 0.1,
        )

        # Apply optimized prompt if successful
        if result.success:
            target.apply_optimized_prompt(best_prompt)

        self.optimization_history.append(result)

        print(f"\n🎯 OPTIMIZATION COMPLETE")
        print(f"Score improvement: {improvement:+.2f}")
        print(
            f"Final alignment: {best_evaluation.preference_alignment.overall_alignment:.2f}"
        )
        print(f"Duration: {duration:.1f}s")
        print(f"Feedback sessions: {feedback_sessions_used}")

        return result

    def _create_optimization_target(self, prompt_type: str) -> PromptOptimizationTarget:
        """Create appropriate optimization target"""
        if prompt_type == "agent":
            return AgentPromptTarget(
                self.domain_eval_config, self.prefs, self.path_manager
            )
        elif prompt_type == "simulation":
            return SimulationPromptTarget(
                self.domain_eval_config, self.prefs, self.path_manager
            )
        elif prompt_type == "evaluation":
            return EvaluationPromptTarget(
                self.domain_eval_config, self.prefs, self.path_manager
            )
        else:
            raise ValueError(
                f"Unknown prompt type: {prompt_type}. Available: agent, simulation, evaluation"
            )


def main():
    """Test the intelligent prompt optimizer v3"""
    print("=== INTELLIGENT PROMPT OPTIMIZER V3 TEST ===")

    # Import domain config
    from agent.eval.domains.roleplay import RoleplayEvaluationConfig

    domain_config = RoleplayEvaluationConfig()

    # Create path manager
    path_manager = OptimizationPathManager(
        base_dir="test_optimization_data", domain="roleplay"
    )

    # Create optimizer
    optimizer = IntelligentPromptOptimizer(domain_config, path_manager)

    # Test simulation prompt optimization (faster than agent)
    print("\n🧪 Testing simulation prompt optimization...")
    result = optimizer.optimize_prompt(
        prompt_type="simulation", max_iterations=3  # Short test
    )

    print(f"\n📊 Optimization Result:")
    print(f"  Original score: {result.original_score:.2f}")
    print(f"  Final score: {result.optimized_score:.2f}")
    print(f"  Improvement: {result.improvement:+.2f}")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Feedback sessions: {result.feedback_sessions_used}")
    print(f"  Final confidence: {result.final_confidence:.2f}")

    print("\n✅ Intelligent prompt optimizer v3 test complete!")


if __name__ == "__main__":
    main()
