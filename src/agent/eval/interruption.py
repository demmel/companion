"""
Truly Smart Interruption System V2

Uses LLM reasoning to intelligently decide when to interrupt for user feedback
instead of hardcoded thresholds and arbitrary numbers.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.progress import ProgressReporter
from agent.structured_llm import structured_llm_call, StructuredLLMError, ResponseFormat
from agent.eval.preferences import SemanticPreferenceManager


class InterruptionReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    STUCK_OPTIMIZATION = "stuck_optimization"
    CONTRADICTORY_TRENDS = "contradictory_trends"
    STRATEGIC_CHECKPOINT = "strategic_checkpoint"


class InterruptionAnalysis(BaseModel):
    """LLM analysis of whether to interrupt for feedback"""

    should_interrupt: bool = Field(
        description="Whether to interrupt the user for feedback"
    )
    reason: Optional[InterruptionReason] = Field(
        description="Primary reason for interruption recommendation"
    )
    urgency: str = Field(description="Urgency level: low, medium, or high")
    confidence: float = Field(description="Confidence in this decision", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the decision")
    expected_benefit: str = Field(
        description="What benefit user feedback would provide"
    )
    risk_if_no_feedback: str = Field(description="Risk of continuing without feedback")


class OptimizationContext(BaseModel):
    """Current state of optimization for interruption decisions"""

    current_iteration: int = Field(description="Current optimization iteration")
    iterations_without_improvement: int = Field(
        description="Iterations since last improvement"
    )
    best_score: float = Field(description="Best score achieved so far")
    current_score: float = Field(description="Current score")
    score_history: List[float] = Field(
        description="History of scores across iterations"
    )
    confidence_level: float = Field(
        description="Current confidence in learned preferences"
    )
    feedback_sessions_this_run: int = Field(
        description="Feedback sessions used in this run"
    )
    optimization_start_time: float = Field(description="When optimization started")
    total_feedback_sessions: int = Field(
        description="Total feedback sessions ever collected"
    )
    preference_summary: Dict[str, Any] = Field(
        description="Summary of learned preferences"
    )


class IntelligentInterruptionSystem:
    """Truly smart interruption system using LLM reasoning"""

    def __init__(
        self,
        preference_manager: SemanticPreferenceManager,
        llm: LLM,
        model: SupportedModel,
        progress_reporter: ProgressReporter,
    ):
        self.prefs = preference_manager
        self.llm = llm
        self.model = model
        self.interruption_history: List[Dict[str, Any]] = []
        self.progress_reporter = progress_reporter

    def should_interrupt(self, context: OptimizationContext) -> InterruptionAnalysis:
        """Use LLM reasoning to decide whether to interrupt for feedback"""

        try:
            system_prompt = """You are an expert at optimization strategy and user experience.
            
            Decide whether to interrupt the user for feedback during prompt optimization.
            
            Consider:
            - Whether we have enough learned preferences to optimize effectively
            - If optimization seems stuck or going in wrong direction
            - The cost/benefit of interrupting vs continuing autonomously
            - User's time and attention as valuable resources
            - Whether feedback now would significantly improve results
            
            Guidelines:
            - Only interrupt when genuinely uncertain or when feedback would be highly valuable
            - Don't interrupt for minor improvements or when confident in direction
            - Consider the user's overall experience and minimize unnecessary interruptions
            - Balance exploration (feedback) with exploitation (using known preferences)
            
            Be strategic and respectful of user time."""

            # Prepare rich context for LLM decision
            context_data = {
                "optimization_progress": {
                    "iteration": context.current_iteration,
                    "iterations_without_improvement": context.iterations_without_improvement,
                    "best_score": context.best_score,
                    "current_score": context.current_score,
                    "score_trend": self._analyze_score_trend(context.score_history),
                    "total_improvement": (
                        context.best_score - context.score_history[0]
                        if context.score_history
                        else 0
                    ),
                },
                "preference_knowledge": {
                    "confidence_level": context.confidence_level,
                    "total_sessions": context.total_feedback_sessions,
                    "sessions_this_run": context.feedback_sessions_this_run,
                    "preference_summary": context.preference_summary,
                },
                "optimization_context": {
                    "duration_minutes": (
                        context.optimization_start_time
                        - context.optimization_start_time
                    )
                    / 60,
                    "previous_interruptions": len(self.interruption_history),
                },
            }

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Should we interrupt the user for feedback at this point in the optimization?",
                response_model=InterruptionAnalysis,
                context=context_data,
                model=self.model,
                llm=self.llm,
                format=ResponseFormat.CUSTOM,
            )

            # Record this decision for learning
            self.interruption_history.append(
                {
                    "context": context.model_dump(),
                    "decision": result.model_dump(),
                    "timestamp": context.optimization_start_time,
                }
            )

            return result

        except StructuredLLMError as e:
            self.progress_reporter.print(f"Error in interruption analysis: {e}")
            # Conservative fallback - don't interrupt unless we really need initial calibration
            return InterruptionAnalysis(
                should_interrupt=False,
                reason=None,
                urgency="low",
                confidence=0.0,
                reasoning=f"Fallback decision due to analysis error: {e}",
                expected_benefit="None",
                risk_if_no_feedback="Continuing without feedback may lead to suboptimal results.",
            )

    def _analyze_score_trend(self, score_history: List[float]) -> str:
        """Analyze the trend in scores"""
        if len(score_history) < 3:
            return "insufficient_data"

        recent = score_history[-3:]
        early = score_history[:3] if len(score_history) >= 6 else score_history[:-3]

        if not early:
            return "early_stage"

        recent_avg = sum(recent) / len(recent)
        early_avg = sum(early) / len(early)

        diff = recent_avg - early_avg

        if diff > 0.5:
            return "strong_improvement"
        elif diff > 0.1:
            return "gradual_improvement"
        elif diff > -0.1:
            return "stable"
        elif diff > -0.5:
            return "gradual_decline"
        else:
            return "strong_decline"

    def get_interruption_summary(self) -> Dict[str, Any]:
        """Get summary of interruption patterns for meta-learning"""
        if not self.interruption_history:
            return {"total_decisions": 0, "interruption_rate": 0.0}

        total_decisions = len(self.interruption_history)
        interruptions = sum(
            1
            for decision in self.interruption_history
            if decision["decision"]["should_interrupt"]
        )

        return {
            "total_decisions": total_decisions,
            "interruption_rate": interruptions / total_decisions,
            "recent_decisions": (
                self.interruption_history[-5:]
                if len(self.interruption_history) >= 5
                else self.interruption_history
            ),
            "most_common_reasons": self._get_common_reasons(),
            "average_confidence": sum(
                d["decision"]["confidence"] for d in self.interruption_history
            )
            / total_decisions,
        }

    def _get_common_reasons(self) -> List[Dict[str, Any]]:
        """Get most common interruption reasons"""
        reason_counts = {}
        for decision in self.interruption_history:
            if (
                decision["decision"]["should_interrupt"]
                and decision["decision"]["reason"]
            ):
                reason = decision["decision"]["reason"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reason_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]


def main():
    """Test the intelligent interruption system"""
    print("=== INTELLIGENT INTERRUPTION SYSTEM V2 TEST ===")

    # Create preference manager and interruption system
    from agent.eval.preferences import SemanticPreferenceManager

    from agent.llm import create_llm, SupportedModel

    from agent.progress import NullProgressReporter

    prefs = SemanticPreferenceManager(
        llm=create_llm(),
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO,
        progress_reporter=NullProgressReporter(),
        preferences_dir="test_preferences",
    )
    llm = create_llm()
    interruption_system = IntelligentInterruptionSystem(
        prefs, llm, SupportedModel.DOLPHIN_MISTRAL_NEMO, NullProgressReporter()
    )

    # Test different scenarios
    test_scenarios = [
        {
            "name": "Initial optimization - no preferences learned",
            "context": OptimizationContext(
                current_iteration=3,
                iterations_without_improvement=1,
                best_score=6.5,
                current_score=6.3,
                score_history=[6.0, 6.5, 6.3],
                confidence_level=0.1,
                feedback_sessions_this_run=0,
                optimization_start_time=1000.0,
                total_feedback_sessions=0,
                preference_summary={"no_preferences": True},
            ),
        },
        {
            "name": "Good progress with learned preferences",
            "context": OptimizationContext(
                current_iteration=8,
                iterations_without_improvement=1,
                best_score=8.2,
                current_score=8.2,
                score_history=[6.5, 7.0, 7.3, 7.8, 8.0, 8.1, 8.2, 8.2],
                confidence_level=0.8,
                feedback_sessions_this_run=1,
                optimization_start_time=1000.0,
                total_feedback_sessions=5,
                preference_summary={"strong_preferences": True},
            ),
        },
        {
            "name": "Optimization seems stuck",
            "context": OptimizationContext(
                current_iteration=12,
                iterations_without_improvement=5,
                best_score=7.2,
                current_score=6.8,
                score_history=[
                    6.8,
                    7.0,
                    7.2,
                    7.0,
                    6.9,
                    6.8,
                    6.7,
                    6.8,
                    6.9,
                    6.8,
                    6.8,
                    6.8,
                ],
                confidence_level=0.6,
                feedback_sessions_this_run=0,
                optimization_start_time=1000.0,
                total_feedback_sessions=3,
                preference_summary={"moderate_preferences": True},
            ),
        },
    ]

    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")

        try:
            decision = interruption_system.should_interrupt(scenario["context"])

            print(f"  Should interrupt: {decision.should_interrupt}")
            print(f"  Reason: {decision.reason}")
            print(f"  Urgency: {decision.urgency}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reasoning: {decision.reasoning}")
            print(f"  Expected benefit: {decision.expected_benefit}")

        except Exception as e:
            print(f"  ❌ Test failed: {e}")

    # Test interruption summary
    print(f"\n📈 Interruption Summary:")
    summary = interruption_system.get_interruption_summary()
    print(f"  Total decisions: {summary['total_decisions']}")
    print(f"  Interruption rate: {summary['interruption_rate']:.2f}")
    print(f"  Average confidence: {summary.get('average_confidence', 0):.2f}")

    print("\n✅ Intelligent interruption system v2 test complete!")


if __name__ == "__main__":
    main()
