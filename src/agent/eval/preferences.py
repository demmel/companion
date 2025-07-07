"""
Semantic User Preference System V2

Uses structured LLM calls for sophisticated semantic understanding
of user preferences with proper Pydantic validation.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from agent.eval.structured_llm import structured_llm_call, StructuredLLMError


class SemanticPreference(BaseModel):
    """A semantically understood user preference"""

    description: str = Field(
        description="Natural language description of what user values or wants to avoid"
    )
    evidence: List[str] = Field(
        description="Specific examples or quotes from feedback that support this preference"
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0", ge=0.0, le=1.0
    )
    domain: str = Field(
        description="Domain this preference applies to: conversation, simulation, or evaluation"
    )
    preference_type: str = Field(
        description="Either 'positive' (what user likes) or 'negative' (what user dislikes)"
    )
    last_updated: float = Field(
        description="Unix timestamp when preference was last updated"
    )
    feedback_count: int = Field(
        description="Number of feedback sessions that contributed to this preference",
        ge=1,
    )


class PreferenceExtractionResult(BaseModel):
    """Result of extracting preferences from user feedback"""

    preferences: List[SemanticPreference] = Field(
        description="List of extracted semantic preferences"
    )
    core_insights: List[str] = Field(
        description="Broader themes about what the user fundamentally values"
    )
    overall_confidence: float = Field(
        description="Overall confidence in the analysis", ge=0.0, le=1.0
    )


class PatternSynthesisResult(BaseModel):
    """Result of analyzing patterns across multiple feedback sessions"""

    core_values: List[str] = Field(
        description="Fundamental values the user consistently cares about"
    )
    consistent_patterns: List[Dict[str, Any]] = Field(
        description="Patterns that appear consistently across feedback sessions"
    )
    preference_evolution: str = Field(
        description="How the user's preferences have changed over time"
    )
    meta_insights: List[str] = Field(
        description="Insights about the user's feedback style and decision making"
    )
    synthesis_confidence: float = Field(
        description="Confidence in the pattern synthesis", ge=0.0, le=1.0
    )


class AlignmentAssessment(BaseModel):
    """Assessment of how well content aligns with user preferences"""

    overall_alignment: float = Field(
        description="Overall alignment score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    positive_alignments: List[str] = Field(
        description="Aspects that match positive user preferences"
    )
    negative_conflicts: List[str] = Field(
        description="Aspects that conflict with user preferences"
    )
    improvement_suggestions: List[str] = Field(
        description="Specific ways to better align with user preferences"
    )
    confidence: float = Field(
        description="Confidence in the alignment assessment", ge=0.0, le=1.0
    )


class OptimizationGuidance(BaseModel):
    """Guidance for optimization based on user preferences"""

    guidance: str = Field(description="Overall strategic guidance for optimization")
    positive_directions: List[str] = Field(
        description="Specific things to emphasize or increase"
    )
    negative_directions: List[str] = Field(
        description="Specific things to avoid or decrease"
    )
    improvement_priorities: List[str] = Field(
        description="Ordered list of what to focus on first"
    )
    confidence: float = Field(description="Confidence in the guidance", ge=0.0, le=1.0)


class SemanticPreferenceProfile(BaseModel):
    """Complete semantic preference profile for a user"""

    schema_version: str = Field(
        default="2.0", description="Schema version for compatibility"
    )
    conversation_preferences: List[SemanticPreference] = Field(
        default_factory=list, description="Preferences for conversation quality"
    )
    simulation_preferences: List[SemanticPreference] = Field(
        default_factory=list, description="Preferences for user simulation realism"
    )
    evaluation_preferences: List[SemanticPreference] = Field(
        default_factory=list, description="Preferences for evaluation accuracy"
    )
    optimization_preferences: List[SemanticPreference] = Field(
        default_factory=list,
        description="Preferences about the optimization process itself",
    )
    core_values: List[str] = Field(
        default_factory=list, description="Fundamental values that span domains"
    )
    total_feedback_sessions: int = Field(
        default=0, description="Total number of feedback sessions", ge=0
    )
    last_updated: float = Field(
        default_factory=time.time, description="Unix timestamp of last update"
    )


class SemanticPreferenceManager:
    """Manages semantic preferences with structured LLM calls"""

    def __init__(
        self,
        preferences_dir: str = "preferences",
        model: str = "huihui_ai/mistral-small-abliterated",
    ):
        self.preferences_dir = Path(preferences_dir)
        self.preferences_dir.mkdir(exist_ok=True)
        self.model = model

        self.profile_file = self.preferences_dir / "semantic_preferences_v2.json"

        # Load existing profile
        self.profile = self._load_profile()

    def _load_profile(self) -> SemanticPreferenceProfile:
        """Load semantic preference profile from file"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, "r") as f:
                    data = json.load(f)
                return SemanticPreferenceProfile.model_validate(data)
            except Exception as e:
                print(f"Warning: Could not load semantic preferences: {e}")

        return SemanticPreferenceProfile()

    def _save_profile(self):
        """Save profile to file"""
        with open(self.profile_file, "w") as f:
            json.dump(self.profile.model_dump(), f, indent=2)

    def add_feedback(self, user_feedback: str, context: Dict[str, Any], domain: str):
        """Add new user feedback and extract preferences using structured LLM"""
        print(f"🧠 Extracting semantic preferences from feedback...")

        try:
            # Extract preferences using structured LLM call
            system_prompt = f"""You are an expert at understanding user preferences from feedback about {domain}.
            
            Extract the underlying preferences and values from the user's feedback.
            
            Focus on:
            - What the user truly values (positive preferences)
            - What they want to avoid (negative preferences)  
            - The deeper principles behind their feedback
            - Specific behaviors, qualities, and characteristics they care about
            
            Be specific and actionable. Don't just restate their words - infer the underlying values."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=user_feedback,
                response_model=PreferenceExtractionResult,
                context={
                    "domain": domain,
                    "context": context,
                    "timestamp": time.time(),
                },
                model=self.model,
            )

            # Add domain and timestamp to extracted preferences
            for pref in result.preferences:
                pref.domain = domain
                pref.last_updated = time.time()
                pref.feedback_count = 1

            # Add to appropriate domain
            if domain == "conversation":
                self.profile.conversation_preferences.extend(result.preferences)
            elif domain == "simulation":
                self.profile.simulation_preferences.extend(result.preferences)
            elif domain == "evaluation":
                self.profile.evaluation_preferences.extend(result.preferences)

            # Update metadata
            self.profile.total_feedback_sessions += 1
            self.profile.last_updated = time.time()

            # Periodically synthesize patterns
            if self.profile.total_feedback_sessions % 5 == 0:
                self._update_core_values()

            self._save_profile()

            print(f"   ✅ Extracted {len(result.preferences)} semantic preferences")
            print(f"   Overall confidence: {result.overall_confidence:.2f}")

            for pref in result.preferences:
                print(f"   • {pref.preference_type}: {pref.description[:80]}...")

            if result.core_insights:
                print(f"   💡 Core insights: {', '.join(result.core_insights)}")

        except StructuredLLMError as e:
            print(f"   ❌ Failed to extract preferences: {e}")

    def _update_core_values(self):
        """Update core values by analyzing preference patterns"""
        print(f"🔍 Analyzing preference patterns across domains...")

        try:
            # Collect all preferences for pattern analysis
            all_preferences = (
                self.profile.conversation_preferences
                + self.profile.simulation_preferences
                + self.profile.evaluation_preferences
            )

            if len(all_preferences) < 3:
                return  # Need more data

            # Prepare preference summary for analysis
            preference_summary = []
            for pref in all_preferences[-20:]:  # Last 20 preferences
                preference_summary.append(
                    {
                        "description": pref.description,
                        "type": pref.preference_type,
                        "domain": pref.domain,
                        "confidence": pref.confidence,
                    }
                )

            system_prompt = """Analyze these user preferences to identify consistent patterns and core values.
            
            Look for:
            - Values that appear consistently across different domains
            - Fundamental principles that drive their preferences
            - Meta-patterns about what they optimize for
            - Evolution in their preference sophistication
            
            Focus on deep, actionable insights about their core values."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze the patterns in these user preferences.",
                response_model=PatternSynthesisResult,
                context={
                    "preferences": preference_summary,
                    "total_sessions": self.profile.total_feedback_sessions,
                    "domains": ["conversation", "simulation", "evaluation"],
                },
                model=self.model,
            )

            # Update core values
            self.profile.core_values = result.core_values
            self._save_profile()

            print(f"   ✅ Updated core values: {', '.join(result.core_values)}")
            print(f"   Pattern confidence: {result.synthesis_confidence:.2f}")

        except StructuredLLMError as e:
            print(f"   ❌ Failed to analyze patterns: {e}")

    def get_preferences(self, domain: str) -> List[SemanticPreference]:
        """Get preferences for a specific domain"""
        if domain == "conversation":
            return self.profile.conversation_preferences
        elif domain == "simulation":
            return self.profile.simulation_preferences
        elif domain == "evaluation":
            return self.profile.evaluation_preferences
        else:
            return []

    def assess_alignment(
        self, content: str, domain: str, context: Dict[str, Any]
    ) -> AlignmentAssessment:
        """Assess how well content aligns with user preferences"""
        preferences = self.get_preferences(domain)

        if not preferences:
            return AlignmentAssessment(
                overall_alignment=0.5,
                positive_alignments=[],
                negative_conflicts=[],
                improvement_suggestions=["No preferences learned yet for this domain"],
                confidence=0.0,
            )

        try:
            # Build preference context
            positive_prefs = [
                p.description
                for p in preferences
                if p.preference_type == "positive" and p.confidence > 0.6
            ]
            negative_prefs = [
                p.description
                for p in preferences
                if p.preference_type == "negative" and p.confidence > 0.6
            ]

            system_prompt = f"""Assess how well the given content aligns with the user's learned preferences for {domain}.
            
            Analyze:
            - How well it matches what the user values (positive preferences)
            - Whether it conflicts with what they want to avoid (negative preferences)
            - Specific improvements to better align with their preferences
            
            Be specific and actionable in your assessment."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=content,
                response_model=AlignmentAssessment,
                context={
                    "domain": domain,
                    "positive_preferences": positive_prefs,
                    "negative_preferences": negative_prefs,
                    "core_values": self.profile.core_values,
                    "additional_context": context,
                },
                model=self.model,
            )

            return result

        except StructuredLLMError as e:
            print(f"Error assessing alignment: {e}")
            return AlignmentAssessment(
                overall_alignment=0.5,
                positive_alignments=[],
                negative_conflicts=[],
                improvement_suggestions=[f"Error in alignment assessment: {e}"],
                confidence=0.0,
            )

    def get_optimization_guidance(self, domain: str) -> OptimizationGuidance:
        """Get structured optimization guidance for a domain"""
        preferences = self.get_preferences(domain)

        if not preferences:
            return OptimizationGuidance(
                guidance="No preferences learned yet - collect initial feedback to provide guidance",
                positive_directions=[],
                negative_directions=[],
                improvement_priorities=["Collect user feedback to learn preferences"],
                confidence=0.0,
            )

        try:
            # Prepare preference context
            positive_prefs = [p for p in preferences if p.preference_type == "positive"]
            negative_prefs = [p for p in preferences if p.preference_type == "negative"]

            system_prompt = f"""Based on the user's learned preferences, provide strategic optimization guidance for {domain}.
            
            Create actionable guidance that:
            - Emphasizes what the user values most
            - Avoids what they dislike
            - Prioritizes the most important improvements
            - Aligns with their core values
            
            Be specific and prioritized in your recommendations."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=f"Provide optimization guidance for {domain}",
                response_model=OptimizationGuidance,
                context={
                    "domain": domain,
                    "positive_preferences": [
                        {"description": p.description, "confidence": p.confidence}
                        for p in positive_prefs
                    ],
                    "negative_preferences": [
                        {"description": p.description, "confidence": p.confidence}
                        for p in negative_prefs
                    ],
                    "core_values": self.profile.core_values,
                    "total_preferences": len(preferences),
                    "total_sessions": self.profile.total_feedback_sessions,
                },
                model=self.model,
            )

            return result

        except StructuredLLMError as e:
            print(f"Error generating guidance: {e}")
            return OptimizationGuidance(
                guidance=f"Error generating guidance: {e}",
                positive_directions=[],
                negative_directions=[],
                improvement_priorities=[],
                confidence=0.0,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of learned preferences"""
        confidence_levels = {}
        for domain in ["conversation", "simulation", "evaluation"]:
            prefs = self.get_preferences(domain)
            if prefs:
                avg_confidence = sum(p.confidence for p in prefs) / len(prefs)
                session_factor = min(1.0, self.profile.total_feedback_sessions / 10)
                confidence_levels[domain] = avg_confidence * session_factor
            else:
                confidence_levels[domain] = 0.0

        return {
            "total_feedback_sessions": self.profile.total_feedback_sessions,
            "conversation_preferences_count": len(
                self.profile.conversation_preferences
            ),
            "simulation_preferences_count": len(self.profile.simulation_preferences),
            "evaluation_preferences_count": len(self.profile.evaluation_preferences),
            "core_values": self.profile.core_values,
            "confidence_levels": confidence_levels,
            "schema_version": self.profile.schema_version,
        }


def main():
    """Test the semantic preference system v2"""
    print("=== SEMANTIC PREFERENCE SYSTEM V2 TEST ===")

    # Create manager
    manager = SemanticPreferenceManager("test_preferences")

    # Test feedback processing
    print("\n🧪 Testing preference extraction...")

    test_feedback = """This character felt really engaging and had great personality depth. 
    I loved how they stayed consistent throughout the conversation and used tools appropriately. 
    However, the dialogue was a bit repetitive and could have been more varied."""

    test_context = {
        "scenario": "Roleplay as Elena, a mysterious vampire",
        "conversation_length": 8,
        "tools_used": ["mood_setting", "character_action"],
    }

    # Add feedback
    manager.add_feedback(test_feedback, test_context, "conversation")

    # Test guidance generation
    print("\n🎯 Testing optimization guidance...")
    guidance = manager.get_optimization_guidance("conversation")

    print(f"  Guidance: {guidance.guidance}")
    print(f"  Positive directions: {guidance.positive_directions}")
    print(f"  Negative directions: {guidance.negative_directions}")
    print(f"  Priorities: {guidance.improvement_priorities}")
    print(f"  Confidence: {guidance.confidence:.2f}")

    # Test alignment assessment
    print("\n🔍 Testing content alignment...")
    test_content = "You are a helpful roleplay assistant who maintains character consistency and creates engaging dialogue with varied responses."

    alignment = manager.assess_alignment(test_content, "conversation", test_context)
    print(f"  Overall alignment: {alignment.overall_alignment:.2f}")
    print(f"  Positive alignments: {alignment.positive_alignments}")
    print(f"  Negative conflicts: {alignment.negative_conflicts}")
    print(f"  Suggestions: {alignment.improvement_suggestions}")
    print(f"  Confidence: {alignment.confidence:.2f}")

    # Show summary
    print("\n📊 Summary:")
    summary = manager.get_summary()
    print(f"  Total sessions: {summary['total_feedback_sessions']}")
    print(f"  Conversation preferences: {summary['conversation_preferences_count']}")
    print(f"  Core values: {summary['core_values']}")
    print(f"  Domain confidence: {summary['confidence_levels']}")

    print("\n✅ Semantic preference system v2 test complete!")


if __name__ == "__main__":
    main()
