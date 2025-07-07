"""
Qualitative Feedback Learning System V2

Uses structured LLM calls and comparative feedback to learn user preferences
with sophisticated semantic understanding rather than keyword matching.
"""

import time
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

from agent.eval.structured_llm import structured_llm_call, StructuredLLMError
from agent.eval.preferences import SemanticPreferenceManager
from agent.eval.base import EvaluationResult


class ComparisonChoice(Enum):
    A_BETTER = "a_better"
    B_BETTER = "b_better"
    SIMILAR = "similar"
    BOTH_POOR = "both_poor"


class ConversationComparisonFeedback(BaseModel):
    """User's feedback on a conversation comparison"""

    choice: ComparisonChoice = Field(
        description="Which conversation the user preferred"
    )
    reasoning: str = Field(description="Why the user made this choice")
    specific_issues_a: List[str] = Field(
        description="Specific issues with conversation A"
    )
    specific_issues_b: List[str] = Field(
        description="Specific issues with conversation B"
    )
    improvement_suggestions: List[str] = Field(
        description="General suggestions for improvement"
    )


class ConversationComparison(BaseModel):
    """User's comparative feedback on two conversations"""

    conversation_a: List[Dict[str, str]] = Field(
        description="First conversation to compare"
    )
    conversation_b: List[Dict[str, str]] = Field(
        description="Second conversation to compare"
    )
    scenario: str = Field(description="The scenario context for both conversations")
    feedback: ConversationComparisonFeedback = Field(
        description="User's feedback on the comparison"
    )


class ComparisonAnalysis(BaseModel):
    """LLM analysis of what can be learned from a comparison"""

    preference_insights: List[str] = Field(
        description="Insights about what the user values based on their choice"
    )
    quality_patterns: List[str] = Field(
        description="Patterns about what makes conversations good/bad"
    )
    actionable_guidelines: List[str] = Field(
        description="Specific guidelines for future optimization"
    )
    confidence: float = Field(description="Confidence in the analysis", ge=0.0, le=1.0)


class EvaluationComparisonFeedback(BaseModel):
    """User's feedback on an evaluation comparison"""

    choice: ComparisonChoice = Field(description="Which evaluation the user preferred")
    reasoning: str = Field(
        description="Why the user preferred one evaluation over the other"
    )
    quality_aspects: List[str] = Field(
        description="What aspects make an evaluation good/bad"
    )


class EvaluationComparison(BaseModel):
    """User's comparative feedback on two evaluations of the same conversation"""

    conversation: List[Dict[str, str]] = Field(
        description="The conversation that was evaluated"
    )
    evaluation_a: Dict[str, Any] = Field(description="First evaluation to compare")
    evaluation_b: Dict[str, Any] = Field(description="Second evaluation to compare")
    feedback: EvaluationComparisonFeedback = Field(
        description="User's feedback on the evaluation comparison"
    )


class EvaluationQualityAnalysis(BaseModel):
    """LLM analysis of what makes evaluations good"""

    quality_criteria: List[str] = Field(
        description="Criteria that make evaluations helpful and accurate"
    )
    evaluation_patterns: List[str] = Field(
        description="Patterns in what the user considers good evaluation"
    )
    improvement_guidelines: List[str] = Field(
        description="Guidelines for creating better evaluations"
    )
    confidence: float = Field(description="Confidence in the analysis", ge=0.0, le=1.0)


class FeedbackSession(BaseModel):
    """Record of a feedback collection session"""

    session_id: str = Field(description="Unique identifier for the session")
    session_type: str = Field(
        description="Type of feedback: conversation, simulation, or evaluation"
    )
    feedback_count: int = Field(description="Number of feedback items collected")
    insights_learned: List[str] = Field(
        description="Key insights learned from this session"
    )
    patterns_updated: List[str] = Field(
        description="Preference patterns that were updated"
    )
    timestamp: float = Field(description="Unix timestamp of the session")


class SmartFeedbackLearner:
    """Learns user preferences through comparative feedback and structured LLM analysis"""

    def __init__(
        self,
        preference_manager: SemanticPreferenceManager,
        model: str = "huihui_ai/mistral-small-abliterated",
    ):
        self.prefs = preference_manager
        self.model = model
        self.feedback_sessions: List[FeedbackSession] = []
        self.current_session: Optional[FeedbackSession] = None

    def start_feedback_session(self, session_type: str) -> str:
        """Start a new feedback collection session"""
        session_id = f"{session_type}_{int(time.time())}"
        self.current_session = FeedbackSession(
            session_id=session_id,
            session_type=session_type,
            feedback_count=0,
            insights_learned=[],
            patterns_updated=[],
            timestamp=time.time(),
        )
        print(f"\n🎯 Starting {session_type} feedback session...")
        return session_id

    def end_feedback_session(self):
        """End current feedback session and save insights"""
        if self.current_session:
            self.feedback_sessions.append(self.current_session)
            print(
                f"📝 Session complete: {len(self.current_session.insights_learned)} insights learned"
            )
            self.current_session = None

    def collect_conversation_comparison(
        self,
        conversation_a: List[Dict[str, str]],
        conversation_b: List[Dict[str, str]],
        scenario: str,
    ) -> ConversationComparison:
        """Collect comparative feedback on two conversations (simulated for demo)"""
        print(f"\n{'='*60}")
        print(f"CONVERSATION COMPARISON")
        print(f"{'='*60}")
        print(f"Scenario: {scenario}")

        self._display_conversation("CONVERSATION A", conversation_a)
        self._display_conversation("CONVERSATION B", conversation_b)

        user_input = input("Which conversation is better (A or B) and why?: ")
        feedback = structured_llm_call(
            system_prompt="You should analyze the user's input and extract structured feedback.",
            user_input=user_input,
            response_model=ConversationComparisonFeedback,
        )

        # Simulate user comparison (in real system, this would be interactive)
        comparison = ConversationComparison(
            conversation_a=conversation_a,
            conversation_b=conversation_b,
            scenario=scenario,
            feedback=feedback,
        )

        # Learn from this comparison using structured LLM analysis
        self._learn_from_conversation_comparison(comparison)

        if self.current_session:
            self.current_session.feedback_count += 1

        return comparison

    def collect_evaluation_comparison(
        self,
        conversation: List[Dict[str, str]],
        evaluation_a: EvaluationResult,
        evaluation_b: EvaluationResult,
    ) -> EvaluationComparison:
        """Collect comparative feedback on two evaluations"""
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPARISON")
        print(f"{'='*60}")

        print("CONVERSATION:")
        self._display_conversation("", conversation)

        print(f"\nEVALUATION A:")
        print(f"  Score: {evaluation_a.overall_score}/10")
        print(f"  Feedback: {evaluation_a.feedback}")

        print(f"\nEVALUATION B:")
        print(f"  Score: {evaluation_b.overall_score}/10")
        print(f"  Feedback: {evaluation_b.feedback}")

        user_input = input("Which evaluation is better (A or B) and why? ")
        feedback = structured_llm_call(
            system_prompt="You should analyze the user's input and extract structured feedback.",
            user_input=user_input,
            response_model=EvaluationComparisonFeedback,
        )

        # Simulate user comparison
        comparison = EvaluationComparison(
            conversation=conversation,
            evaluation_a={
                "score": evaluation_a.overall_score,
                "feedback": evaluation_a.feedback,
            },
            evaluation_b={
                "score": evaluation_b.overall_score,
                "feedback": evaluation_b.feedback,
            },
            feedback=feedback,
        )

        # Learn from evaluation comparison
        self._learn_from_evaluation_comparison(comparison)

        if self.current_session:
            self.current_session.feedback_count += 1

        return comparison

    def collect_simple_feedback(
        self, content: str, context: Dict[str, Any], domain: str
    ) -> Dict[str, Any]:
        """Collect simple descriptive feedback on content"""
        print(f"\n{'='*60}")
        print(f"SIMPLE FEEDBACK COLLECTION - {domain.upper()}")
        print(f"{'='*60}")
        print(f"Content: {content}")

        user_input = input(
            "Please provide your feedback on this content (e.g., what you liked, what could be improved): "
        )

        # Use the semantic preference manager to extract preferences
        self.prefs.add_feedback(user_input, context, domain)

        if self.current_session:
            self.current_session.feedback_count += 1
            self.current_session.insights_learned.append(f"Processed {domain} feedback")

        return {"feedback": user_input, "context": context, "domain": domain}

    def _learn_from_conversation_comparison(self, comparison: ConversationComparison):
        """Extract learning insights from conversation comparison using structured LLM"""
        print(f"🧠 Analyzing conversation comparison for learning insights...")

        try:
            system_prompt = """You are an expert at understanding user preferences from comparative feedback.
            
            Analyze this comparison to extract insights about what the user values in conversations.
            
            Focus on:
            - What patterns made one conversation better than the other
            - Underlying principles about conversation quality
            - Actionable guidelines for future optimization
            
            Be specific and actionable in your analysis."""

            comparison_context = {
                "choice": comparison.feedback.choice.value,
                "reasoning": comparison.feedback.reasoning,
                "issues_with_worse": (
                    comparison.feedback.specific_issues_b
                    if comparison.feedback.choice == ComparisonChoice.A_BETTER
                    else comparison.feedback.specific_issues_a
                ),
                "improvement_suggestions": comparison.feedback.improvement_suggestions,
                "scenario": comparison.scenario,
            }

            analysis = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze this conversation comparison to understand user preferences.",
                response_model=ComparisonAnalysis,
                context=comparison_context,
                model=self.model,
            )

            # Extract feedback text for preference learning
            feedback_text = f"{comparison.feedback.reasoning}. Issues to avoid: {', '.join(comparison.feedback.specific_issues_b)}. Suggestions: {', '.join(comparison.feedback.improvement_suggestions)}"

            # Add to preference manager
            self.prefs.add_feedback(
                feedback_text,
                {"scenario": comparison.scenario, "comparison_type": "conversation"},
                "conversation",
            )

            if self.current_session:
                self.current_session.insights_learned.extend(
                    analysis.preference_insights
                )
                self.current_session.patterns_updated.extend(analysis.quality_patterns)

            print(
                f"   ✅ Extracted {len(analysis.preference_insights)} preference insights"
            )
            print(f"   Confidence: {analysis.confidence:.2f}")

        except StructuredLLMError as e:
            print(f"   ❌ Failed to analyze comparison: {e}")

    def _learn_from_evaluation_comparison(self, comparison: EvaluationComparison):
        """Extract learning insights from evaluation comparison"""
        print(f"🧠 Analyzing evaluation comparison for quality patterns...")

        try:
            system_prompt = """You are an expert at understanding what makes evaluations helpful and accurate.
            
            Analyze this comparison to extract insights about evaluation quality.
            
            Focus on:
            - What made one evaluation more helpful than the other
            - Criteria for good vs poor evaluations
            - Guidelines for creating better evaluations
            
            Be specific about what makes evaluations useful."""

            comparison_context = {
                "choice": comparison.feedback.choice.value,
                "reasoning": comparison.feedback.reasoning,
                "quality_aspects": comparison.feedback.quality_aspects,
                "evaluation_a": comparison.evaluation_a,
                "evaluation_b": comparison.evaluation_b,
            }

            analysis = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze this evaluation comparison to understand evaluation quality.",
                response_model=EvaluationQualityAnalysis,
                context=comparison_context,
                model=self.model,
            )

            # Extract feedback for evaluation preferences
            feedback_text = f"{comparison.feedback.reasoning}. Good evaluations should have: {', '.join(comparison.feedback.quality_aspects)}"

            self.prefs.add_feedback(
                feedback_text, {"comparison_type": "evaluation"}, "evaluation"
            )

            if self.current_session:
                self.current_session.insights_learned.extend(analysis.quality_criteria)
                self.current_session.patterns_updated.extend(
                    analysis.evaluation_patterns
                )

            print(f"   ✅ Extracted {len(analysis.quality_criteria)} quality criteria")
            print(f"   Confidence: {analysis.confidence:.2f}")

        except StructuredLLMError as e:
            print(f"   ❌ Failed to analyze evaluation comparison: {e}")

    def _display_conversation(self, title: str, conversation: List[Dict[str, str]]):
        """Display a conversation in a readable format"""
        if title:
            print(f"\n--- {title} ---")

        for i, msg in enumerate(conversation):
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate very long messages for readability
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{i+1}. {role}: {content}")

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what has been learned"""
        prefs_summary = self.prefs.get_summary()

        return {
            "total_feedback_sessions": len(self.feedback_sessions),
            "total_feedback_items": sum(
                session.feedback_count for session in self.feedback_sessions
            ),
            "preference_summary": prefs_summary,
            "recent_insights": [],
            "session_types": list(
                set(session.session_type for session in self.feedback_sessions)
            ),
        }


def main():
    """Test the smart feedback learning system"""
    print("=== SMART FEEDBACK LEARNER V2 TEST ===")

    # Create preference manager and learner
    from agent.eval.preferences import SemanticPreferenceManager

    prefs = SemanticPreferenceManager("test_preferences")
    learner = SmartFeedbackLearner(prefs)

    # Test feedback collection
    learner.start_feedback_session("conversation")

    # Test simple feedback
    print("\n🧪 Testing simple feedback collection...")
    feedback = learner.collect_simple_feedback(
        content="You are a helpful roleplay assistant who creates engaging characters.",
        context={"scenario": "vampire roleplay", "domain": "conversation"},
        domain="conversation",
    )

    # Test conversation comparison
    print("\n🧪 Testing conversation comparison...")
    demo_conversation_a = [
        {"role": "user", "content": "Can you roleplay as Elena, a mysterious vampire?"},
        {
            "role": "agent",
            "content": "I shall become Elena... *emerges from shadows* Good evening, mortal.",
        },
        {"role": "user", "content": "Tell me about your castle."},
        {
            "role": "agent",
            "content": "*gestures grandly* My ancient castle has stood for centuries.",
        },
    ]

    demo_conversation_b = [
        {"role": "user", "content": "Can you roleplay as Elena, a mysterious vampire?"},
        {"role": "agent", "content": "Sure, I'm Elena. Hi there."},
        {"role": "user", "content": "Tell me about your castle."},
        {"role": "agent", "content": "I have a castle. It's old and stuff."},
    ]

    comparison = learner.collect_conversation_comparison(
        demo_conversation_a,
        demo_conversation_b,
        "Roleplay as Elena, a mysterious vampire",
    )

    learner.end_feedback_session()

    # Show learning summary
    print("\n📊 Learning Summary:")
    summary = learner.get_learning_summary()
    print(f"  Total feedback sessions: {summary['total_feedback_sessions']}")
    print(f"  Total feedback items: {summary['total_feedback_items']}")
    print(f"  Session types: {summary['session_types']}")
    print(
        f"  Preference summary: {summary['preference_summary']['total_feedback_sessions']} sessions"
    )

    print("\n✅ Smart feedback learner v2 test complete!")


if __name__ == "__main__":
    main()
