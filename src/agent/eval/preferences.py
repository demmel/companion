"""
Semantic User Preference System V3 - Advanced Preference Intelligence

A sophisticated preference management system that:
- Detects and resolves contradictions between preferences
- Dynamically adjusts confidence levels based on evidence
- Supports preference removal and lifecycle management
- Synthesizes holistic philosophy from individual preferences
- Tracks preference evolution and relationship patterns

Uses structured LLM calls for advanced semantic reasoning.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
from rich.console import Console

from agent.llm import LLM, SupportedModel
from agent.structured_llm import structured_llm_call, StructuredLLMError
from agent.progress import ProgressReporter

logger = logging.getLogger(__name__)
console = Console()


class PreferenceStatus(str, Enum):
    """Status of a preference in its lifecycle"""

    ACTIVE = "active"  # Currently active and influencing decisions
    DEPRECATED = "deprecated"  # Still tracked but not actively used
    CONTRADICTED = "contradicted"  # Conflicts with newer preferences
    MERGED = "merged"  # Combined into a broader preference
    REMOVED = "removed"  # Explicitly removed by user or system


class ContradictionType(str, Enum):
    """Types of contradictions between preferences"""

    DIRECT = "direct"  # Directly opposite preferences
    CONTEXTUAL = "contextual"  # Same context, different values
    PRIORITY = "priority"  # Different prioritization of same values
    SCOPE = "scope"  # Different scope (general vs specific)


class LLMPreferenceExtraction(BaseModel):
    """LLM extraction of preference - semantic content and analysis the LLM can determine"""

    # Core semantic content
    description: str = Field(
        description="Natural language description of what user values or wants to avoid"
    )
    evidence: List[str] = Field(
        description="Specific examples or quotes from feedback that support this preference"
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0", ge=0.0, le=1.0
    )
    preference_type: str = Field(
        description="Either 'positive' (what user likes) or 'negative' (what user dislikes)"
    )

    # LLM-analyzable context and categorization
    contexts: List[str] = Field(
        default_factory=list,
        description="Specific contexts where this preference applies",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing this preference",
    )

    # LLM-assessed importance and insights
    priority_weight: float = Field(
        default=1.0,
        description="Priority weight for this preference (higher = more important)",
        ge=0.0,
    )
    user_notes: Optional[str] = Field(
        default=None, description="Optional user notes about this preference"
    )


class SemanticPreference(BaseModel):
    """Full system preference with LLM content + system metadata"""

    # LLM-extracted semantic content
    description: str = Field(
        description="Natural language description of what user values or wants to avoid"
    )
    evidence: List[str] = Field(
        description="Specific examples or quotes from feedback that support this preference"
    )
    confidence: float = Field(
        description="Dynamic confidence level from 0.0 to 1.0", ge=0.0, le=1.0
    )
    preference_type: str = Field(
        description="Either 'positive' (what user likes) or 'negative' (what user dislikes)"
    )
    contexts: List[str] = Field(
        default_factory=list,
        description="Specific contexts where this preference applies",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing and filtering preferences",
    )

    # System-managed metadata (set by PreferenceManager)
    id: str = Field(description="Unique identifier for this preference")
    domain: str = Field(
        description="Domain this preference applies to: conversation, simulation, evaluation, or general"
    )
    status: PreferenceStatus = Field(
        default=PreferenceStatus.ACTIVE,
        description="Current status in preference lifecycle",
    )

    # Lifecycle tracking (system-managed)
    created_at: float = Field(description="Unix timestamp when preference was created")
    last_updated: float = Field(
        description="Unix timestamp when preference was last updated"
    )
    last_reinforced: float = Field(
        description="Unix timestamp when preference was last reinforced"
    )
    feedback_count: int = Field(
        description="Number of feedback sessions that contributed to this preference",
        ge=1,
    )
    reinforcement_count: int = Field(
        default=1, description="Number of times this preference was reinforced", ge=1
    )

    # Contradiction and relationship tracking (system-managed)
    contradicts: List[str] = Field(
        default_factory=list, description="IDs of preferences this one contradicts"
    )
    supersedes: List[str] = Field(
        default_factory=list,
        description="IDs of preferences this one replaces or supersedes",
    )
    merged_from: List[str] = Field(
        default_factory=list,
        description="IDs of preferences that were merged into this one",
    )

    # System metadata
    priority_weight: float = Field(
        default=1.0,
        description="Priority weight for this preference (higher = more important)",
        ge=0.0,
    )
    user_notes: Optional[str] = Field(
        default=None, description="Optional user notes about this preference"
    )


class ContradictionAnalysis(BaseModel):
    """Analysis of contradictions between preferences"""

    contradicting_preference_ids: List[str] = Field(
        description="IDs of preferences that contradict each other"
    )
    contradiction_type: ContradictionType = Field(
        description="Type of contradiction detected"
    )
    contradiction_description: str = Field(
        description="Explanation of how these preferences contradict"
    )
    resolution_strategy: str = Field(
        description="Recommended strategy for resolving this contradiction"
    )
    confidence: float = Field(
        description="Confidence in contradiction detection", ge=0.0, le=1.0
    )


class ContradictionAnalysisResult(BaseModel):
    """Container for contradiction analysis results"""

    contradictions: List[ContradictionAnalysis] = Field(
        description="List of detected contradictions"
    )


class PreferenceEvolution(BaseModel):
    """Analysis of how preferences have evolved over time"""

    preference_id: str = Field(description="ID of the preference being analyzed")
    evolution_type: str = Field(
        description="Type of evolution: strengthened, weakened, specialized, generalized, contradicted"
    )
    confidence_change: float = Field(
        description="Change in confidence over time (-1.0 to 1.0)"
    )
    description_changes: List[str] = Field(
        description="Key changes in how this preference is described"
    )
    evidence_strength: str = Field(
        description="Assessment of evidence strength: weak, moderate, strong"
    )
    recommendation: str = Field(
        description="Recommendation for this preference: keep, merge, deprecate, remove"
    )


class PreferenceExtractionResult(BaseModel):
    """Pure LLM result of extracting preferences from user feedback - no system metadata"""

    preferences: List[LLMPreferenceExtraction] = Field(
        description="List of extracted preferences (semantic content only)"
    )
    reinforced_preferences: List[str] = Field(
        default_factory=list,
        description="IDs of existing preferences that were reinforced",
    )
    contradictions_detected: List[ContradictionAnalysis] = Field(
        default_factory=list,
        description="Contradictions found with existing preferences",
    )
    core_insights: List[str] = Field(
        description="Broader themes about what the user fundamentally values"
    )
    overall_confidence: float = Field(
        description="Overall confidence in the analysis", ge=0.0, le=1.0
    )


class PhilosophySynthesis(BaseModel):
    """Holistic synthesis of user's preference philosophy"""

    core_philosophy: str = Field(
        description="User's overarching philosophy synthesized from all preferences"
    )
    value_hierarchy: List[str] = Field(
        description="Ordered list of user's core values from most to least important"
    )
    decision_principles: List[str] = Field(
        description="Key principles that guide the user's decision making"
    )
    preference_themes: Dict[str, str] = Field(
        description="Major themes and their descriptions across all preferences"
    )
    consistency_assessment: str = Field(
        description="Assessment of how consistent the user's preferences are"
    )
    confidence: float = Field(
        description="Confidence in the philosophy synthesis", ge=0.0, le=1.0
    )


class PatternSynthesisResult(BaseModel):
    """Enhanced result of analyzing patterns across multiple feedback sessions"""

    philosophy_synthesis: PhilosophySynthesis = Field(
        description="Holistic philosophy derived from all preferences"
    )
    preference_evolution: List[PreferenceEvolution] = Field(
        description="Analysis of how individual preferences have evolved"
    )
    contradiction_summary: List[ContradictionAnalysis] = Field(
        description="Summary of all detected contradictions"
    )
    recommendations: Dict[str, List[str]] = Field(
        description="Recommendations for preference management by category"
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
    """Advanced semantic preference profile with sophisticated management"""

    schema_version: str = Field(
        default="3.0", description="Schema version for compatibility"
    )

    # Preference storage by domain
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
    general_preferences: List[SemanticPreference] = Field(
        default_factory=list, description="Cross-domain general preferences"
    )

    # Philosophy and synthesis
    current_philosophy: Optional[PhilosophySynthesis] = Field(
        default=None, description="Current synthesized philosophy"
    )
    philosophy_history: List[PhilosophySynthesis] = Field(
        default_factory=list, description="Historical philosophy snapshots"
    )

    # Contradiction and lifecycle tracking
    active_contradictions: List[ContradictionAnalysis] = Field(
        default_factory=list, description="Currently unresolved contradictions"
    )
    resolved_contradictions: List[ContradictionAnalysis] = Field(
        default_factory=list, description="Previously resolved contradictions"
    )

    # Metadata and statistics
    total_feedback_sessions: int = Field(
        default=0, description="Total number of feedback sessions", ge=0
    )
    total_preferences_created: int = Field(
        default=0, description="Total preferences created over time", ge=0
    )
    last_contradiction_check: float = Field(
        default=0.0, description="Last time contradictions were checked"
    )
    last_philosophy_synthesis: float = Field(
        default=0.0, description="Last time philosophy was synthesized"
    )
    last_updated: float = Field(
        default_factory=time.time, description="Unix timestamp of last update"
    )


class SemanticPreferenceManager:
    """Advanced preference manager with contradiction detection and philosophy synthesis"""

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        progress_reporter: ProgressReporter,
        preferences_dir: str = "preferences",
        auto_contradiction_check: bool = True,
        auto_philosophy_synthesis: bool = True,
    ):
        self.preferences_dir = Path(preferences_dir)
        self.preferences_dir.mkdir(exist_ok=True)
        self.llm = llm
        self.model = model
        self.progress_reporter = progress_reporter
        self.auto_contradiction_check = auto_contradiction_check
        self.auto_philosophy_synthesis = auto_philosophy_synthesis

        self.profile_file = self.preferences_dir / "semantic_preferences_v3.json"

        # Legacy file migration
        legacy_file = self.preferences_dir / "semantic_preferences_v2.json"
        if legacy_file.exists() and not self.profile_file.exists():
            console.print("🔄 Found legacy preferences file, will migrate on load...")

        # Load existing profile
        self.profile = self._load_profile()

        # Auto-run maintenance if enabled
        self._run_maintenance_if_needed()

    def _load_profile(self) -> SemanticPreferenceProfile:
        """Load semantic preference profile from file with migration support"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, "r") as f:
                    data = json.load(f)

                # Handle migration from v2.0 to v3.0
                if data.get("schema_version", "2.0") == "2.0":
                    data = self._migrate_v2_to_v3(data)

                return SemanticPreferenceProfile.model_validate(data)
            except Exception as e:
                logger.warning(f"Could not load semantic preferences: {e}")
                # Try to backup corrupted file
                backup_path = self.profile_file.with_suffix(".backup.json")
                try:
                    self.profile_file.rename(backup_path)
                    console.print(f"Corrupted file backed up to: {backup_path}")
                except:
                    pass

        return SemanticPreferenceProfile()

    def _migrate_v2_to_v3(self, v2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v2.0 preference data to v3.0 format"""
        console.print("🔄 Migrating preferences from v2.0 to v3.0...")

        v3_data = {
            "schema_version": "3.0",
            "conversation_preferences": [],
            "simulation_preferences": [],
            "evaluation_preferences": [],
            "optimization_preferences": [],
            "general_preferences": [],
            "current_philosophy": None,
            "philosophy_history": [],
            "active_contradictions": [],
            "resolved_contradictions": [],
            "total_feedback_sessions": v2_data.get("total_feedback_sessions", 0),
            "total_preferences_created": 0,
            "last_contradiction_check": 0.0,
            "last_philosophy_synthesis": 0.0,
            "last_updated": time.time(),
        }

        # Migrate preferences with new fields
        for domain in [
            "conversation_preferences",
            "simulation_preferences",
            "evaluation_preferences",
            "optimization_preferences",
        ]:
            old_prefs = v2_data.get(domain, [])
            new_prefs = []

            for i, old_pref in enumerate(old_prefs):
                new_pref = {
                    "id": f"migrated_{domain}_{i}_{int(time.time())}",
                    "description": old_pref.get("description", ""),
                    "evidence": old_pref.get("evidence", []),
                    "confidence": old_pref.get("confidence", 0.5),
                    "domain": old_pref.get(
                        "domain", domain.replace("_preferences", "")
                    ),
                    "preference_type": old_pref.get("preference_type", "positive"),
                    "status": "active",
                    "created_at": old_pref.get("last_updated", time.time()),
                    "last_updated": old_pref.get("last_updated", time.time()),
                    "last_reinforced": old_pref.get("last_updated", time.time()),
                    "feedback_count": old_pref.get("feedback_count", 1),
                    "reinforcement_count": 1,
                    "contradicts": [],
                    "supersedes": [],
                    "merged_from": [],
                    "contexts": [],
                    "priority_weight": 1.0,
                    "tags": [],
                    "user_notes": None,
                }
                new_prefs.append(new_pref)
                v3_data["total_preferences_created"] += 1

            v3_data[domain] = new_prefs

        console.print(
            f"✅ Migrated {v3_data['total_preferences_created']} preferences to v3.0"
        )
        return v3_data

    def _save_profile(self):
        """Save profile to file with backup"""
        # Create backup of existing file
        if self.profile_file.exists():
            backup_path = self.profile_file.with_suffix(".bak")
            try:
                self.profile_file.rename(backup_path)
            except:
                pass  # Backup failed, continue anyway

        # Save new profile
        with open(self.profile_file, "w") as f:
            json.dump(self.profile.model_dump(), f, indent=2)

    def _run_maintenance_if_needed(self):
        """Run automated maintenance tasks if sufficient time has passed"""
        current_time = time.time()

        # Check for contradictions every 24 hours or after 5 new feedback sessions
        time_since_contradiction_check = (
            current_time - self.profile.last_contradiction_check
        )
        sessions_since_check = self.profile.total_feedback_sessions

        if time_since_contradiction_check > 86400 or (  # 24 hours
            sessions_since_check % 5 == 0
            and sessions_since_check > 0
            and self.auto_contradiction_check
        ):
            self._detect_and_resolve_contradictions()

        # Synthesize philosophy every 48 hours or after 10 feedback sessions
        time_since_philosophy = current_time - self.profile.last_philosophy_synthesis
        if time_since_philosophy > 172800 or (  # 48 hours
            sessions_since_check % 10 == 0
            and sessions_since_check > 0
            and self.auto_philosophy_synthesis
        ):
            self._synthesize_philosophy()

    def _generate_preference_id(self, description: str, domain: str) -> str:
        """Generate a unique ID for a preference"""
        import hashlib

        content = f"{description}_{domain}_{time.time()}"
        return f"pref_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def _create_preference_from_extraction(
        self, llm_extraction: LLMPreferenceExtraction, domain: str
    ) -> SemanticPreference:
        """Convert LLM extraction to full system preference with metadata"""
        current_time = time.time()

        return SemanticPreference(
            # LLM-extracted semantic content
            description=llm_extraction.description,
            evidence=llm_extraction.evidence,
            confidence=llm_extraction.confidence,
            preference_type=llm_extraction.preference_type,
            contexts=llm_extraction.contexts,
            tags=llm_extraction.tags + ["new"],  # Add "new" tag to user tags
            priority_weight=llm_extraction.priority_weight,
            user_notes=llm_extraction.user_notes,
            # System-managed metadata
            id=self._generate_preference_id(llm_extraction.description, domain),
            domain=domain,
            status=PreferenceStatus.ACTIVE,
            created_at=current_time,
            last_updated=current_time,
            last_reinforced=current_time,
            feedback_count=1,
            reinforcement_count=1,
            # Lists initialize to empty by default
        )

    def _get_all_active_preferences(self) -> List[SemanticPreference]:
        """Get all active preferences across all domains"""
        all_prefs = []
        for domain_prefs in [
            self.profile.conversation_preferences,
            self.profile.simulation_preferences,
            self.profile.evaluation_preferences,
            self.profile.optimization_preferences,
            self.profile.general_preferences,
        ]:
            all_prefs.extend(
                [p for p in domain_prefs if p.status == PreferenceStatus.ACTIVE]
            )
        return all_prefs

    def _find_preference_by_id(self, pref_id: str) -> Optional[SemanticPreference]:
        """Find a preference by its ID across all domains"""
        for pref in self._get_all_active_preferences():
            if pref.id == pref_id:
                return pref
        return None

    def _detect_and_resolve_contradictions(self):
        """Detect contradictions between preferences and attempt resolution"""
        self.progress_reporter.print("🔍 Analyzing preferences for contradictions...")

        active_prefs = self._get_all_active_preferences()
        if len(active_prefs) < 2:
            self.profile.last_contradiction_check = time.time()
            return

        try:
            # Prepare preference summaries for analysis
            pref_summaries = []
            for pref in active_prefs:
                pref_summaries.append(
                    {
                        "id": pref.id,
                        "description": pref.description,
                        "type": pref.preference_type,
                        "domain": pref.domain,
                        "confidence": pref.confidence,
                        "evidence_count": len(pref.evidence),
                        "contexts": pref.contexts,
                    }
                )

            system_prompt = """You are an expert at analyzing user preferences for contradictions and inconsistencies.
            
            Analyze the provided preferences to identify:
            1. Direct contradictions (opposite preferences)
            2. Contextual conflicts (same situation, different values)
            3. Priority conflicts (incompatible prioritization)
            4. Scope conflicts (general vs specific contradictions)
            
            For each contradiction found:
            - Identify the specific preferences involved
            - Classify the type of contradiction
            - Explain how they contradict
            - Suggest a resolution strategy
            
            Focus on meaningful contradictions that would cause decision conflicts."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Analyze these preferences for contradictions.",
                response_model=ContradictionAnalysisResult,
                context={
                    "preferences": pref_summaries,
                    "total_preferences": len(active_prefs),
                    "domains": list(set(p.domain for p in active_prefs)),
                },
                model=self.model,
                llm=self.llm,
            )

            # Process detected contradictions
            new_contradictions = []
            for contradiction in result.contradictions:
                if self._is_new_contradiction(contradiction):
                    new_contradictions.append(contradiction)
                    self._handle_contradiction(contradiction)

            if new_contradictions:
                self.profile.active_contradictions.extend(new_contradictions)
                self.progress_reporter.print(
                    f"⚠️ Found {len(new_contradictions)} new contradictions"
                )
                for cont in new_contradictions:
                    self.progress_reporter.print(
                        f"   {cont.contradiction_type.value}: {cont.contradiction_description[:80]}..."
                    )
            else:
                self.progress_reporter.print("✅ No new contradictions detected")

            self.profile.last_contradiction_check = time.time()
            self._save_profile()

        except StructuredLLMError as e:
            logger.error(f"Failed to analyze contradictions: {e}")

    def _is_new_contradiction(self, contradiction: ContradictionAnalysis) -> bool:
        """Check if this is a new contradiction we haven't seen before"""
        pref_ids = set(contradiction.contradicting_preference_ids)

        for existing in (
            self.profile.active_contradictions + self.profile.resolved_contradictions
        ):
            existing_ids = set(existing.contradicting_preference_ids)
            if (
                pref_ids == existing_ids
                and existing.contradiction_type == contradiction.contradiction_type
            ):
                return False
        return True

    def _handle_contradiction(self, contradiction: ContradictionAnalysis):
        """Handle a detected contradiction based on the resolution strategy"""
        if "merge" in contradiction.resolution_strategy.lower():
            self._merge_contradicting_preferences(contradiction)
        elif "deprecate" in contradiction.resolution_strategy.lower():
            self._deprecate_weaker_preference(contradiction)
        elif "contextualize" in contradiction.resolution_strategy.lower():
            self._contextualize_preferences(contradiction)
        # Otherwise, leave for manual resolution

    def _merge_contradicting_preferences(self, contradiction: ContradictionAnalysis):
        """Merge contradicting preferences into a single, more nuanced preference"""
        # This is a complex operation that would create a new preference
        # that captures the nuance of both contradicting preferences
        # For now, mark them for manual review
        for pref_id in contradiction.contradicting_preference_ids:
            pref = self._find_preference_by_id(pref_id)
            if pref:
                pref.tags.append("needs_merge")

    def _deprecate_weaker_preference(self, contradiction: ContradictionAnalysis):
        """Deprecate the preference with lower confidence/evidence"""
        prefs = [
            self._find_preference_by_id(pid)
            for pid in contradiction.contradicting_preference_ids
        ]
        prefs = [p for p in prefs if p is not None]

        if len(prefs) >= 2:
            # Sort by confidence and evidence strength
            prefs.sort(
                key=lambda p: (p.confidence, len(p.evidence), p.reinforcement_count)
            )
            weakest = prefs[0]
            weakest.status = PreferenceStatus.DEPRECATED
            weakest.tags.append("auto_deprecated")

    def _contextualize_preferences(self, contradiction: ContradictionAnalysis):
        """Add context tags to help distinguish when each preference applies"""
        for pref_id in contradiction.contradicting_preference_ids:
            pref = self._find_preference_by_id(pref_id)
            if pref:
                pref.tags.append("needs_context")

    def add_feedback(self, user_feedback: str, context: Dict[str, Any], domain: str):
        """Add new user feedback and extract preferences with advanced analysis"""
        self.progress_reporter.print(
            f"🧠 Extracting semantic preferences from feedback..."
        )

        try:
            # Get existing preferences for contradiction detection
            existing_prefs = self.get_preferences(domain)
            existing_summaries = [
                {
                    "id": p.id,
                    "description": p.description,
                    "type": p.preference_type,
                    "confidence": p.confidence,
                }
                for p in existing_prefs
                if p.status == PreferenceStatus.ACTIVE
            ]

            # Extract preferences using enhanced structured LLM call
            system_prompt = f"""You are an expert at understanding user preferences from feedback about {domain}.
            
            Analyze the user's feedback and:
            1. Extract new preferences and values
            2. Identify existing preferences that are reinforced
            3. Detect any contradictions with existing preferences
            4. Assess confidence levels based on evidence strength
            
            For new preferences:
            - Infer deeper principles behind their feedback
            - Assign appropriate confidence based on evidence strength
            - Identify specific contexts where they apply
            - Assess relative priority/importance
            
            For contradictions:
            - Clearly identify which existing preferences conflict
            - Explain the nature of the contradiction
            - Suggest resolution strategies
            
            Be specific and actionable. Don't just restate their words - infer the underlying values."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=user_feedback,
                response_model=PreferenceExtractionResult,
                context={
                    "domain": domain,
                    "context": context,
                    "existing_preferences": existing_summaries,
                    "timestamp": time.time(),
                },
                model=self.model,
                llm=self.llm,
            )

            # Process new preferences - convert LLM extractions to full system preferences
            new_prefs_added = 0
            for llm_extraction in result.preferences:
                # Convert LLM extraction to full system preference with metadata
                pref = self._create_preference_from_extraction(llm_extraction, domain)

                # Add to appropriate domain
                if domain == "conversation":
                    self.profile.conversation_preferences.append(pref)
                elif domain == "simulation":
                    self.profile.simulation_preferences.append(pref)
                elif domain == "evaluation":
                    self.profile.evaluation_preferences.append(pref)
                elif domain == "general":
                    self.profile.general_preferences.append(pref)

                new_prefs_added += 1
                self.profile.total_preferences_created += 1

            # Process reinforced preferences
            reinforced_count = 0
            for pref_id in result.reinforced_preferences:
                pref = self._find_preference_by_id(pref_id)
                if pref:
                    pref.reinforcement_count += 1
                    pref.last_reinforced = time.time()
                    pref.confidence = min(
                        1.0, pref.confidence + 0.1
                    )  # Small confidence boost
                    reinforced_count += 1

            # Handle detected contradictions
            if result.contradictions_detected:
                self.profile.active_contradictions.extend(
                    result.contradictions_detected
                )
                self.progress_reporter.print(
                    f"   ⚠️ Detected {len(result.contradictions_detected)} contradictions"
                )

            # Update metadata
            self.profile.total_feedback_sessions += 1
            self.profile.last_updated = time.time()

            # Run maintenance if needed
            self._run_maintenance_if_needed()

            self._save_profile()

            self.progress_reporter.print(
                f"   ✅ Added {new_prefs_added} new preferences"
            )
            if reinforced_count > 0:
                self.progress_reporter.print(
                    f"   💪 Reinforced {reinforced_count} existing preferences"
                )
            self.progress_reporter.print(
                f"   Overall confidence: {result.overall_confidence:.2f}"
            )

            if result.core_insights:
                self.progress_reporter.print(
                    f"   💡 Core insights: {', '.join(result.core_insights)}"
                )

        except StructuredLLMError as e:
            self.progress_reporter.print(f"   ❌ Failed to extract preferences: {e}")

    def _synthesize_philosophy(self):
        """Synthesize holistic philosophy from all preferences"""
        self.progress_reporter.print(
            f"🧸 Synthesizing holistic preference philosophy..."
        )

        try:
            active_prefs = self._get_all_active_preferences()
            if len(active_prefs) < 3:
                return  # Need more data

            # Prepare comprehensive preference analysis
            preference_analysis = {
                "active_preferences": [],
                "domains_covered": set(),
                "preference_types": {"positive": 0, "negative": 0},
                "confidence_distribution": [],
                "evolution_trends": [],
            }

            for pref in active_prefs:
                preference_analysis["active_preferences"].append(
                    {
                        "id": pref.id,
                        "description": pref.description,
                        "type": pref.preference_type,
                        "domain": pref.domain,
                        "confidence": pref.confidence,
                        "reinforcement_count": pref.reinforcement_count,
                        "contexts": pref.contexts,
                        "tags": pref.tags,
                        "created_days_ago": (time.time() - pref.created_at) / 86400,
                    }
                )

                preference_analysis["domains_covered"].add(pref.domain)
                preference_analysis["preference_types"][pref.preference_type] += 1
                preference_analysis["confidence_distribution"].append(pref.confidence)

            preference_analysis["domains_covered"] = list(
                preference_analysis["domains_covered"]
            )

            system_prompt = """You are an expert philosopher and psychologist analyzing user preferences to derive their core philosophy.
            
            Create a comprehensive synthesis that:
            1. Identifies the user's core philosophy and worldview
            2. Establishes a value hierarchy (most to least important)
            3. Extracts decision-making principles
            4. Identifies major themes across all preferences
            5. Assesses consistency and coherence
            
            Focus on:
            - Deep, fundamental values that drive all preferences
            - How different domains connect to overarching principles
            - The user's implicit decision-making framework
            - Areas of strength vs inconsistency in their preferences
            
            Provide actionable insights that could guide future optimization."""

            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input="Synthesize a holistic philosophy from these user preferences.",
                response_model=PatternSynthesisResult,
                context={
                    "preference_analysis": preference_analysis,
                    "total_sessions": self.profile.total_feedback_sessions,
                    "total_preferences": len(active_prefs),
                    "active_contradictions": len(self.profile.active_contradictions),
                    "profile_age_days": (
                        time.time()
                        - getattr(self.profile, "created_at", time.time() - 86400)
                    )
                    / 86400,
                },
                model=self.model,
                llm=self.llm,
            )

            # Store philosophy synthesis
            if self.profile.current_philosophy:
                self.profile.philosophy_history.append(self.profile.current_philosophy)

            self.profile.current_philosophy = result.philosophy_synthesis
            self.profile.last_philosophy_synthesis = time.time()

            # Process evolution recommendations
            for evolution in result.preference_evolution:
                pref = self._find_preference_by_id(evolution.preference_id)
                if pref:
                    if evolution.recommendation == "remove":
                        pref.status = PreferenceStatus.REMOVED
                    elif evolution.recommendation == "deprecate":
                        pref.status = PreferenceStatus.DEPRECATED
                    elif evolution.recommendation == "merge":
                        pref.tags.append("candidate_for_merge")

                    # Adjust confidence based on evolution analysis
                    if evolution.confidence_change != 0:
                        new_confidence = max(
                            0.0, min(1.0, pref.confidence + evolution.confidence_change)
                        )
                        pref.confidence = new_confidence

            self._save_profile()

            self.progress_reporter.print(
                f"   ✅ Philosophy synthesized: {result.philosophy_synthesis.core_philosophy[:80]}..."
            )
            self.progress_reporter.print(
                f"   🏆 Top values: {', '.join(result.philosophy_synthesis.value_hierarchy[:3])}"
            )
            self.progress_reporter.print(
                f"   📊 Synthesis confidence: {result.synthesis_confidence:.2f}"
            )

            if result.contradiction_summary:
                self.progress_reporter.print(
                    f"   ⚠️ Found {len(result.contradiction_summary)} contradictions to resolve"
                )

        except StructuredLLMError as e:
            self.progress_reporter.print(f"   ❌ Failed to synthesize philosophy: {e}")

    def get_preferences(
        self, domain: str, include_inactive: bool = False
    ) -> List[SemanticPreference]:
        """Get preferences for a specific domain with status filtering"""
        if domain == "conversation":
            prefs = self.profile.conversation_preferences
        elif domain == "simulation":
            prefs = self.profile.simulation_preferences
        elif domain == "evaluation":
            prefs = self.profile.evaluation_preferences
        elif domain == "general":
            prefs = self.profile.general_preferences
        elif domain == "all":
            prefs = self._get_all_active_preferences()
        else:
            return []

        if include_inactive:
            return prefs
        else:
            return [p for p in prefs if p.status == PreferenceStatus.ACTIVE]

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
                    "core_values": (
                        getattr(self.profile.current_philosophy, "value_hierarchy", [])
                        if self.profile.current_philosophy
                        else []
                    ),
                    "additional_context": context,
                },
                model=self.model,
                llm=self.llm,
            )

            return result

        except StructuredLLMError as e:
            logger.error(f"Error assessing alignment: {e}")
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
                    "core_values": (
                        getattr(self.profile.current_philosophy, "value_hierarchy", [])
                        if self.profile.current_philosophy
                        else []
                    ),
                    "total_preferences": len(preferences),
                    "total_sessions": self.profile.total_feedback_sessions,
                },
                model=self.model,
                llm=self.llm,
            )

            return result

        except StructuredLLMError as e:
            logger.error(f"Error generating guidance: {e}")
            return OptimizationGuidance(
                guidance=f"Error generating guidance: {e}",
                positive_directions=[],
                negative_directions=[],
                improvement_priorities=[],
                confidence=0.0,
            )

    def remove_preference(
        self, preference_id: str, reason: str = "user_request"
    ) -> bool:
        """Remove a preference by ID"""
        pref = self._find_preference_by_id(preference_id)
        if pref:
            pref.status = PreferenceStatus.REMOVED
            pref.user_notes = f"Removed: {reason}"
            pref.last_updated = time.time()
            self._save_profile()
            self.progress_reporter.print(
                f"✅ Removed preference: {pref.description[:50]}..."
            )
            return True
        return False

    def adjust_preference_confidence(
        self, preference_id: str, new_confidence: float, reason: str = ""
    ) -> bool:
        """Manually adjust confidence of a preference"""
        pref = self._find_preference_by_id(preference_id)
        if pref and 0.0 <= new_confidence <= 1.0:
            old_confidence = pref.confidence
            pref.confidence = new_confidence
            pref.last_updated = time.time()
            if reason:
                pref.user_notes = (
                    f"{pref.user_notes or ''} | Confidence adjusted: {reason}"
                )
            self._save_profile()
            self.progress_reporter.print(
                f"✅ Updated confidence: {old_confidence:.2f} → {new_confidence:.2f}"
            )
            return True
        return False

    def get_philosophy_summary(self) -> Dict[str, Any]:
        """Get current philosophy synthesis"""
        if not self.profile.current_philosophy:
            return {
                "status": "not_synthesized",
                "recommendation": "Need more feedback to synthesize philosophy",
            }

        phil = self.profile.current_philosophy
        return {
            "status": "available",
            "core_philosophy": phil.core_philosophy,
            "top_values": phil.value_hierarchy[:5],
            "key_principles": phil.decision_principles[:3],
            "consistency_level": phil.consistency_assessment,
            "confidence": phil.confidence,
            "last_updated": self.profile.last_philosophy_synthesis,
        }

    def get_contradictions_summary(self) -> Dict[str, Any]:
        """Get summary of detected contradictions"""
        active = self.profile.active_contradictions
        resolved = self.profile.resolved_contradictions

        return {
            "active_count": len(active),
            "resolved_count": len(resolved),
            "active_contradictions": [
                {
                    "type": c.contradiction_type.value,
                    "description": c.contradiction_description,
                    "affected_preferences": len(c.contradicting_preference_ids),
                    "confidence": c.confidence,
                }
                for c in active
            ],
            "last_check": self.profile.last_contradiction_check,
        }

    def get_average_confidence(self) -> float:
        """Get average confidence across all active preferences"""
        active_prefs = self._get_all_active_preferences()
        if not active_prefs:
            return 0.0
        return sum(p.confidence for p in active_prefs) / len(active_prefs)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of learned preferences"""
        confidence_levels = {}
        status_counts = {}

        for domain in ["conversation", "simulation", "evaluation", "general"]:
            prefs = self.get_preferences(domain, include_inactive=True)
            active_prefs = [p for p in prefs if p.status == PreferenceStatus.ACTIVE]

            if active_prefs:
                avg_confidence = sum(p.confidence for p in active_prefs) / len(
                    active_prefs
                )
                session_factor = min(1.0, self.profile.total_feedback_sessions / 10)
                confidence_levels[domain] = avg_confidence * session_factor
            else:
                confidence_levels[domain] = 0.0

            # Count by status
            status_counts[domain] = {}
            for status in PreferenceStatus:
                status_counts[domain][status.value] = len(
                    [p for p in prefs if p.status == status]
                )

        return {
            "schema_version": self.profile.schema_version,
            "total_feedback_sessions": self.profile.total_feedback_sessions,
            "total_preferences_created": self.profile.total_preferences_created,
            "confidence_levels": confidence_levels,
            "status_counts": status_counts,
            "philosophy": self.get_philosophy_summary(),
            "contradictions": self.get_contradictions_summary(),
            "last_updated": self.profile.last_updated,
            "maintenance": {
                "last_contradiction_check": self.profile.last_contradiction_check,
                "last_philosophy_synthesis": self.profile.last_philosophy_synthesis,
                "auto_contradiction_check": self.auto_contradiction_check,
                "auto_philosophy_synthesis": self.auto_philosophy_synthesis,
            },
        }


def main():
    """Test the advanced semantic preference system v3"""
    print("=== SEMANTIC PREFERENCE SYSTEM V3 TEST ===")

    # Create manager with advanced features
    from agent.llm import create_llm, SupportedModel

    llm = create_llm()
    from agent.progress import NullProgressReporter

    manager = SemanticPreferenceManager(
        llm=llm,
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO,
        progress_reporter=NullProgressReporter(),
        auto_contradiction_check=True,
        auto_philosophy_synthesis=True,
    )

    # Test feedback processing with contradiction detection
    print("\n🧠 Testing preference extraction...")

    test_feedback_1 = """This character felt really engaging and had great personality depth. 
    I loved how they stayed consistent throughout the conversation and used tools appropriately. 
    However, the dialogue was a bit repetitive and could have been more varied."""

    test_context = {
        "scenario": "Roleplay as Elena, a mysterious vampire",
        "conversation_length": 8,
        "tools_used": ["mood_setting", "character_action"],
    }

    # Add first feedback
    manager.add_feedback(test_feedback_1, test_context, "conversation")

    # Add contradictory feedback to test contradiction detection
    test_feedback_2 = """I actually prefer when characters are more unpredictable and change their personality 
    throughout the conversation. Consistency can be boring. Also, I love repetitive dialogue patterns 
    because they feel more natural."""

    manager.add_feedback(test_feedback_2, test_context, "conversation")

    # Test philosophy synthesis
    print("\n🧸 Testing philosophy synthesis...")
    manager._synthesize_philosophy()

    philosophy = manager.get_philosophy_summary()
    if philosophy["status"] == "available":
        print(f"  Core philosophy: {philosophy['core_philosophy'][:100]}...")
        print(f"  Top values: {philosophy['top_values']}")
        print(f"  Consistency: {philosophy['consistency_level']}")

    # Test contradiction detection
    print("\n⚠️ Testing contradiction detection...")
    contradictions = manager.get_contradictions_summary()
    print(f"  Active contradictions: {contradictions['active_count']}")
    print(f"  Resolved contradictions: {contradictions['resolved_count']}")

    for contradiction in contradictions["active_contradictions"]:
        print(f"    {contradiction['type']}: {contradiction['description'][:80]}...")

    # Test guidance generation
    print("\n🎯 Testing optimization guidance...")
    guidance = manager.get_optimization_guidance("conversation")

    print(f"  Guidance: {guidance.guidance}")
    print(f"  Positive directions: {guidance.positive_directions}")
    print(f"  Negative directions: {guidance.negative_directions}")
    print(f"  Priorities: {guidance.improvement_priorities}")
    print(f"  Confidence: {guidance.confidence:.2f}")

    # Test preference management
    print("\n🛠️ Testing preference management...")
    prefs = manager.get_preferences("conversation")
    if prefs:
        test_pref = prefs[0]
        print(f"  Original confidence: {test_pref.confidence:.2f}")

        # Test confidence adjustment
        manager.adjust_preference_confidence(test_pref.id, 0.8, "Testing adjustment")
        print(f"  Adjusted confidence: {test_pref.confidence:.2f}")

    # Show comprehensive summary
    print("\n📊 Comprehensive Summary:")
    summary = manager.get_summary()
    print(f"  Schema version: {summary['schema_version']}")
    print(f"  Total sessions: {summary['total_feedback_sessions']}")
    print(f"  Total preferences created: {summary['total_preferences_created']}")
    print(f"  Philosophy status: {summary['philosophy']['status']}")
    print(f"  Active contradictions: {summary['contradictions']['active_count']}")

    for domain, counts in summary["status_counts"].items():
        active = counts.get("active", 0)
        total = sum(counts.values())
        if total > 0:
            print(f"  {domain}: {active}/{total} preferences active")

    print("\n✅ Advanced semantic preference system v3 test complete!")


if __name__ == "__main__":
    main()
