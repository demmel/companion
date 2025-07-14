"""
Sequential Multi-Prompt Optimization System

Implements the sequential workflow:
1. Generate conversation A with agent prompt A + sim prompt A
2. Evaluate with agent eval prompt A + sim eval prompt A
3. Optimize agent prompt A→B, sim prompt A→B using evaluations
4. Generate conversation B with agent prompt B + sim prompt B
5. Evaluate conversation B with same eval prompts A
6. Compare conversations via user feedback (if interruption) or learned preferences
7. Check if evaluators ranked correctly vs user preferences
8. Optimize evaluator prompts if they were wrong
9. Test new evaluators on existing conversations A+B
10. Cycle with optimized prompts as new "A" prompts
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from agent.eval.optimization_paths import OptimizationPathManager
from agent.llm import LLM, SupportedModel
from agent.structured_llm import structured_llm_call, StructuredLLMError, ResponseFormat
from agent.eval.feedback_learner import SmartFeedbackLearner
from agent.eval.preferences import SemanticPreferenceManager
from agent.eval.conversation_generator import ConversationGenerator
from agent.eval.base import DomainEvaluationConfig
from agent.eval.interruption import IntelligentInterruptionSystem, OptimizationContext
from agent.eval.prompt_versioning import PromptVersionManager
from agent.progress import ProgressReporter, NullProgressReporter

AGENT_MUTATION_PROMPT = """You are an expert prompt engineer specializing in AI agent behavior.

You are given:
1. Current agent prompt
2. Evaluation of agent performance 

Your task is to improve the agent prompt to address the issues identified in the evaluation.

Focus on:
- Fixing specific behavioral issues mentioned in the evaluation
- Improving agent response quality and consistency
- Enhancing tool usage and conversation flow
- Maintaining the agent's core personality and goals

Provide the improved prompt along with a clear rationale for your changes."""

SIM_MUTATION_PROMPT = """You are an expert prompt engineer specializing in user simulation.

You are given:
1. Current user simulation prompt
2. Evaluation of simulation performance

Your task is to improve the simulation prompt to address the issues identified in the evaluation.

Focus on:
- Making simulated user behavior more realistic and varied
- Improving conversation engagement and natural flow
- Creating better test scenarios for the agent
- Addressing any artificial or repetitive behaviors

Provide the improved prompt along with a clear rationale for your changes."""

AGENT_EVAL_MUTATION_PROMPT = """You are an expert prompt engineer specializing in evaluation systems.

You are given:
1. Current agent evaluation prompt
2. User feedback about what makes good agent behavior

Your task is to improve the evaluation prompt to better align with user preferences.

Focus on:
- Incorporating user values and preferences into evaluation criteria
- Creating more accurate assessment of agent behavior quality
- Identifying the aspects of conversation that users actually care about
- Ensuring evaluations predict user satisfaction

Provide the improved evaluation prompt along with rationale for changes."""

SIM_EVAL_MUTATION_PROMPT = """You are an expert prompt engineer specializing in evaluation systems.

You are given:
1. Current simulation evaluation prompt
2. User feedback about what makes good user simulation

Your task is to improve the evaluation prompt to better align with user preferences.

Focus on:
- Incorporating user values about simulation quality
- Creating more accurate assessment of simulation behavior quality
- Identifying the aspects of conversation that users actually care about
- Ensuring evaluations predict user satisfaction

Provide the improved evaluation prompt along with rationale for changes."""


class PromptEvaluation(BaseModel):
    """Base class for prompt evaluations"""

    overall_score: float = Field(
        description="Overall score for the behavior (0-10)", ge=0.0, le=10.0
    )
    detailed_feedback: str = Field(description="Detailed feedback on the behavior")
    strengths: List[str] = Field(
        description="List of strengths observed in the behavior"
    )
    weaknesses: List[str] = Field(
        description="List of weaknesses or issues in the behavior"
    )
    key_observations: List[str] = Field(
        description="Key observations about the behavior"
    )


class AgentPromptEvaluation(PromptEvaluation):
    """Evaluation of an agent prompt based on user feedback"""

    overall_score: float = Field(
        description="Overall score for the agent behavior (0-10)", ge=0.0, le=10.0
    )
    detailed_feedback: str = Field(
        description="Detailed feedback on the agent's behavior"
    )
    strengths: List[str] = Field(
        description="List of strengths observed in the agent's behavior"
    )
    weaknesses: List[str] = Field(
        description="List of weaknesses or issues in the agent's behavior"
    )
    key_observations: List[str] = Field(
        description="Key observations about the agent's performance"
    )


class SimulationPromptEvaluation(PromptEvaluation):
    """Evaluation of a simulation prompt based on user feedback"""

    overall_score: float = Field(
        description="Overall score for the simulation behavior (0-10)", ge=0.0, le=10.0
    )
    detailed_feedback: str = Field(
        description="Detailed feedback on the simulation's behavior"
    )
    strengths: List[str] = Field(
        description="List of strengths observed in the simulation's behavior"
    )
    weaknesses: List[str] = Field(
        description="List of weaknesses or issues in the simulation's behavior"
    )
    key_observations: List[str] = Field(
        description="Key observations about the simulation's performance"
    )


@dataclass
class Conversation:
    conversation: List[Dict[str, str]]

    agent_prompt: str
    sim_prompt: str

    agent_eval: Optional[AgentPromptEvaluation] = None
    sim_eval: Optional[SimulationPromptEvaluation] = None


@dataclass
class ConversationPair:
    """A pair of conversations with their evaluations for comparison"""

    conversation_a: List[Dict[str, str]]
    conversation_b: List[Dict[str, str]]
    scenario: str

    # Prompts used
    agent_prompt_a: str
    agent_prompt_b: str
    sim_prompt_a: str
    sim_prompt_b: str
    agent_eval_prompt: str
    sim_eval_prompt: str

    # Evaluations using current evaluator prompts
    agent_eval_a: Optional[AgentPromptEvaluation] = None
    agent_eval_b: Optional[AgentPromptEvaluation] = None
    sim_eval_a: Optional[SimulationPromptEvaluation] = None
    sim_eval_b: Optional[SimulationPromptEvaluation] = None

    # User feedback (when collected)
    user_prefers_agent: Optional[str] = None  # "A", "B", or "similar"
    user_prefers_sim: Optional[str] = None  # "A", "B", or "similar"

    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PromptMutation(BaseModel):
    """Result of prompt mutation using hardcoded mutation prompts"""

    improved_prompt: str = Field(description="The enhanced prompt text")
    rationale: str = Field(description="Explanation of changes made")
    targeted_issues: List[str] = Field(
        description="Issues from evaluation that were addressed"
    )
    confidence: float = Field(
        description="Confidence in the improvement", ge=0.0, le=1.0
    )


class EvaluatorComparison(BaseModel):
    """Comparison of old vs new evaluator performance"""

    old_ranking: str = Field(description="Old evaluator preference: A, B, or similar")
    new_ranking: str = Field(description="New evaluator preference: A, B, or similar")
    user_preference: str = Field(description="User preference: A, B, or similar")
    old_matches_user: bool = Field(
        description="Whether old evaluator matches user preference"
    )
    new_matches_user: bool = Field(
        description="Whether new evaluator matches user preference"
    )
    recommendation: str = Field(description="Which evaluator to use: old or new")


class SequentialOptimizationRun(BaseModel):
    """Complete record of a sequential optimization run"""

    run_id: str = Field(description="Unique identifier for this run")
    domain: str = Field(description="Domain being optimized")
    scenario: str = Field(description="Test scenario used")

    initial_prompts: Dict[str, str] = Field(
        description="Initial prompts before optimization"
    )
    final_prompts: Dict[str, str] = Field(description="Final optimized prompts")

    conversation_pairs: int = Field(
        description="Number of conversation pairs generated"
    )
    feedback_sessions: int = Field(description="Number of user feedback sessions")
    evaluator_optimizations: Dict[str, bool] = Field(
        description="Which evaluators were optimized"
    )

    success: bool = Field(description="Whether optimization achieved improvement")
    duration_seconds: float = Field(description="Total optimization time")
    timestamp: float = Field(default_factory=time.time)


class SequentialOptimizer:
    """
    Sequential optimization system following the specified workflow.
    Uses hardcoded mutation prompts for now, optimizes evaluators based on user feedback alignment.
    """

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        preference_manager: SemanticPreferenceManager,
        llm: LLM,
        model: SupportedModel,
        progress: ProgressReporter,
        path_manager,  # Required path manager for saving optimized prompts
        agent_mutation_prompt: str = AGENT_MUTATION_PROMPT,
        sim_mutation_prompt: str = SIM_MUTATION_PROMPT,
        agent_eval_mutation_prompt: str = AGENT_EVAL_MUTATION_PROMPT,
        sim_eval_mutation_prompt: str = SIM_EVAL_MUTATION_PROMPT,
    ):
        self.domain_config = domain_eval_config
        self.prefs = preference_manager
        self.llm = llm
        self.model = model
        self.progress = progress
        self.path_manager = path_manager

        # Initialize components
        self.feedback_learner = SmartFeedbackLearner(
            preference_manager, llm, model, progress
        )
        self.interruption_system = IntelligentInterruptionSystem(
            preference_manager, llm, model, progress
        )
        self.version_manager = PromptVersionManager(path_manager, progress)

        # Conversation pairs for analysis
        self.conversation_pairs: List[ConversationPair] = []

        # Optimization history
        self.optimization_history: List[SequentialOptimizationRun] = []

        # Mutation prompts
        self.agent_mutation_prompt = agent_mutation_prompt
        self.sim_mutation_prompt = sim_mutation_prompt
        self.agent_eval_mutation_prompt = agent_eval_mutation_prompt
        self.sim_eval_mutation_prompt = sim_eval_mutation_prompt

    def mutate_from_performance(
        self, current_prompt: str, evaluation: PromptEvaluation, mutation_prompt: str
    ) -> PromptMutation:
        """Apply a mutation prompt to improve a current prompt based on evaluation"""

        evaluation_text = f"""
Evaluation Results:
- Overall Score: {evaluation.overall_score}/10
- Detailed Feedback: {evaluation.detailed_feedback}
- Strengths: {evaluation.strengths}
- Weaknesses: {evaluation.weaknesses}
- Key Observations: {evaluation.key_observations}
"""

        system_prompt = mutation_prompt
        user_input = f"""
Current Prompt:
{current_prompt}

{evaluation_text}

Please provide an improved version of the prompt that addresses the evaluation issues.
"""

        try:
            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=user_input,
                response_model=PromptMutation,
                model=self.model,
                llm=self.llm,
                format=ResponseFormat.CUSTOM,
            )
            return result

        except StructuredLLMError as e:
            self.progress.print(f"Prompt mutation failed: {e}")
            return PromptMutation(
                improved_prompt=current_prompt,
                rationale=f"Mutation failed: {e}",
                targeted_issues=[],
                confidence=0.0,
            )

    def mutate_from_alignment(
        self,
        current_prompt: str,
        user_pref: str,
        eval_ranking: str,
        reasoning: str,
        mutation_prompt: str,
    ) -> PromptMutation:
        """Apply a mutation prompt to improve evaluator alignment with user preferences"""

        alignment_context = f"""
User Preference: {user_pref}
Evaluator Ranking: {eval_ranking}
User Reasoning: {reasoning}

The evaluator ranked differently than the user preferred, indicating misalignment.
"""

        system_prompt = mutation_prompt
        user_input = f"""
Current Evaluator Prompt:
{current_prompt}

{alignment_context}

Please provide an improved version of the evaluator prompt that better aligns with user preferences.
"""

        try:
            result = structured_llm_call(
                system_prompt=system_prompt,
                user_input=user_input,
                response_model=PromptMutation,
                model=self.model,
                llm=self.llm,
                format=ResponseFormat.CUSTOM,
            )
            return result

        except StructuredLLMError as e:
            self.progress.print(f"Evaluator mutation failed: {e}")
            return PromptMutation(
                improved_prompt=current_prompt,
                rationale=f"Mutation failed: {e}",
                targeted_issues=[],
                confidence=0.0,
            )

    def generate_conversation(
        self, scenario: str, agent_prompt: str, sim_prompt: str
    ) -> List[Dict[str, str]]:
        """Generate a conversation using specified prompts"""

        # Create temporary evaluator with custom prompts
        eval_config = self.domain_config.get_evaluation_config()
        agent_config = self.domain_config.get_agent_config()

        # Temporarily update prompts
        original_agent_prompt = agent_config.prompt_template
        original_sim_prompt = eval_config.simulation_prompt_template

        agent_config.prompt_template = agent_prompt
        eval_config.simulation_prompt_template = sim_prompt

        try:
            evaluator = ConversationGenerator(
                domain_eval_config=self.domain_config,
                model=self.model,
                llm=self.llm,
                progress=self.progress,
            )

            conversation = evaluator.generate_conversation(scenario)
            return conversation

        finally:
            # Restore original prompts
            agent_config.prompt_template = original_agent_prompt
            eval_config.simulation_prompt_template = original_sim_prompt

    def evaluate_conversation(
        self,
        conversation: List[Dict[str, str]],
        scenario: str,
        agent_eval_prompt: str,
        sim_eval_prompt: str,
    ) -> Tuple[AgentPromptEvaluation, SimulationPromptEvaluation]:
        """Evaluate a conversation with both agent and simulation evaluators"""

        conversation_text = "\n\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation]
        )

        # Agent evaluation
        agent_eval_context = {"scenario": scenario, "conversation": conversation_text}
        agent_eval = structured_llm_call(
            system_prompt=agent_eval_prompt,
            user_input="Evaluate the agent's behavior in this conversation.",
            response_model=AgentPromptEvaluation,
            context=agent_eval_context,
            model=self.model,
            llm=self.llm,
            format=ResponseFormat.CUSTOM,
        )

        # Simulation evaluation
        sim_eval_context = {"scenario": scenario, "conversation": conversation_text}
        sim_eval = structured_llm_call(
            system_prompt=sim_eval_prompt,
            user_input="Evaluate the simulation quality in this conversation.",
            response_model=SimulationPromptEvaluation,
            context=sim_eval_context,
            model=self.model,
            llm=self.llm,
            format=ResponseFormat.CUSTOM,
        )

        return agent_eval, sim_eval

    def get_user_preferences(
        self, conversation_pair: ConversationPair
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get user preferences for agent and simulation behavior via feedback or learned preferences"""

        # Check if interruption system recommends collecting feedback
        context = OptimizationContext(
            current_iteration=len(self.conversation_pairs),
            iterations_without_improvement=0,  # TODO: track this properly
            best_score=0.0,  # TODO: track this properly
            current_score=0.0,  # TODO: track this properly
            score_history=[],
            confidence_level=self.prefs.get_average_confidence(),
            feedback_sessions_this_run=0,  # TODO: track this properly
            optimization_start_time=0.0,  # TODO: track this properly
            total_feedback_sessions=0,  # TODO: track this properly
            preference_summary={},  # TODO: get from preference manager
        )

        # Get interruption decision and show reasoning transparently
        interruption_decision = self.interruption_system.should_interrupt(context)
        self._display_interruption_decision(interruption_decision, context)

        if interruption_decision.should_interrupt:
            return self._collect_user_feedback(conversation_pair)
        else:
            return self._use_learned_preferences(conversation_pair)

    def _collect_user_feedback(
        self, conversation_pair: ConversationPair
    ) -> Tuple[Optional[str], Optional[str]]:
        """Collect user feedback comparing agent and simulation behavior"""

        self.progress.print(
            f"\n🔍 Collecting user feedback for conversation comparison..."
        )

        self.feedback_learner.start_feedback_session("sequential_optimization")

        try:
            # Agent behavior comparison
            self.progress.print("\n--- AGENT BEHAVIOR COMPARISON ---")
            agent_comparison = self.feedback_learner.collect_conversation_comparison(
                conversation_pair.conversation_a,
                conversation_pair.conversation_b,
                conversation_pair.scenario,
            )

            agent_preference = (
                "A"
                if agent_comparison.feedback.choice.value == "a_better"
                else (
                    "B"
                    if agent_comparison.feedback.choice.value == "b_better"
                    else "similar"
                )
            )

            # Simulation behavior comparison
            self.progress.print("\n--- SIMULATION BEHAVIOR COMPARISON ---")
            sim_comparison = self.feedback_learner.collect_conversation_comparison(
                conversation_pair.conversation_a,
                conversation_pair.conversation_b,
                conversation_pair.scenario,
            )

            sim_preference = (
                "A"
                if sim_comparison.feedback.choice.value == "a_better"
                else (
                    "B"
                    if sim_comparison.feedback.choice.value == "b_better"
                    else "similar"
                )
            )

            return agent_preference, sim_preference

        finally:
            self.feedback_learner.end_feedback_session()

    def _use_learned_preferences(
        self, conversation_pair: ConversationPair
    ) -> Tuple[Optional[str], Optional[str]]:
        """Use learned preferences to predict user choice (placeholder for now)"""

        self.progress.print("📚 Using learned preferences instead of user feedback...")

        # For now, return None to indicate we should use evaluator rankings
        # TODO: Implement preference-based prediction
        return None, None

    def compare_evaluators(
        self,
        conversation_pair: ConversationPair,
        new_agent_eval_prompt: Optional[str] = None,
        new_sim_eval_prompt: Optional[str] = None,
    ) -> Tuple[Optional[EvaluatorComparison], Optional[EvaluatorComparison]]:
        """Compare old vs new evaluator performance on existing conversations"""

        results = []

        # Test agent evaluator if we have a new one
        if new_agent_eval_prompt:
            # Run new agent evaluator on both conversations
            new_agent_eval_a, _ = self.evaluate_conversation(
                conversation_pair.conversation_a,
                conversation_pair.scenario,
                new_agent_eval_prompt,
                conversation_pair.sim_eval_prompt,  # Use existing sim eval
            )

            new_agent_eval_b, _ = self.evaluate_conversation(
                conversation_pair.conversation_b,
                conversation_pair.scenario,
                new_agent_eval_prompt,
                conversation_pair.sim_eval_prompt,
            )

            # Compare rankings
            old_score_a = (
                conversation_pair.agent_eval_a.overall_score
                if conversation_pair.agent_eval_a
                else 5.0
            )
            old_score_b = (
                conversation_pair.agent_eval_b.overall_score
                if conversation_pair.agent_eval_b
                else 5.0
            )
            new_score_a = new_agent_eval_a.overall_score
            new_score_b = new_agent_eval_b.overall_score

            old_ranking = (
                "A"
                if old_score_a > old_score_b
                else "B" if old_score_b > old_score_a else "similar"
            )
            new_ranking = (
                "A"
                if new_score_a > new_score_b
                else "B" if new_score_b > new_score_a else "similar"
            )

            user_pref = conversation_pair.user_prefers_agent or "similar"

            agent_comparison = EvaluatorComparison(
                old_ranking=old_ranking,
                new_ranking=new_ranking,
                user_preference=user_pref,
                old_matches_user=(old_ranking == user_pref),
                new_matches_user=(new_ranking == user_pref),
                recommendation=(
                    "new"
                    if (new_ranking == user_pref and old_ranking != user_pref)
                    else "old"
                ),
            )
            results.append(agent_comparison)
        else:
            results.append(None)

        # Test simulation evaluator if we have a new one
        if new_sim_eval_prompt:
            # Similar logic for simulation evaluator
            _, new_sim_eval_a = self.evaluate_conversation(
                conversation_pair.conversation_a,
                conversation_pair.scenario,
                conversation_pair.agent_eval_prompt,  # Use existing agent eval
                new_sim_eval_prompt,
            )

            _, new_sim_eval_b = self.evaluate_conversation(
                conversation_pair.conversation_b,
                conversation_pair.scenario,
                conversation_pair.agent_eval_prompt,
                new_sim_eval_prompt,
            )

            old_score_a = (
                conversation_pair.sim_eval_a.overall_score
                if conversation_pair.sim_eval_a
                else 5.0
            )
            old_score_b = (
                conversation_pair.sim_eval_b.overall_score
                if conversation_pair.sim_eval_b
                else 5.0
            )
            new_score_a = new_sim_eval_a.overall_score
            new_score_b = new_sim_eval_b.overall_score

            old_ranking = (
                "A"
                if old_score_a > old_score_b
                else "B" if old_score_b > old_score_a else "similar"
            )
            new_ranking = (
                "A"
                if new_score_a > new_score_b
                else "B" if new_score_b > new_score_a else "similar"
            )

            user_pref = conversation_pair.user_prefers_sim or "similar"

            sim_comparison = EvaluatorComparison(
                old_ranking=old_ranking,
                new_ranking=new_ranking,
                user_preference=user_pref,
                old_matches_user=(old_ranking == user_pref),
                new_matches_user=(new_ranking == user_pref),
                recommendation=(
                    "new"
                    if (new_ranking == user_pref and old_ranking != user_pref)
                    else "old"
                ),
            )
            results.append(sim_comparison)
        else:
            results.append(None)

        return tuple(results)

    def run_optimization_cycle(
        self,
        scenario: str,
        agent_prompt: str,
        sim_prompt: str,
        agent_eval_prompt: str,
        sim_eval_prompt: str,
    ) -> Tuple[Dict[str, str], int]:
        """
        Run one complete optimization cycle following the workflow.
        Returns (updated_prompts, feedback_sessions_count)
        """

        self.progress.print(
            f"\n🔄 Starting optimization cycle for scenario: {scenario[:60]}..."
        )

        # Step 1: Generate conversation A
        self.progress.print("📝 Generating conversation A...")
        conversation_a = self.generate_conversation(scenario, agent_prompt, sim_prompt)

        # Show conversation A
        self._display_conversation("Generated Conversation A", conversation_a)

        # Step 2: Evaluate conversation A
        self.progress.print("📊 Evaluating conversation A...")
        agent_eval_a, sim_eval_a = self.evaluate_conversation(
            conversation_a, scenario, agent_eval_prompt, sim_eval_prompt
        )

        # Show evaluation results clearly
        self._display_evaluation_results("Agent Evaluation A", agent_eval_a)
        self._display_evaluation_results("Simulation Evaluation A", sim_eval_a)

        # Step 3: Optimize agent prompt A→B
        self.progress.print("🎯 Optimizing agent prompt...")
        agent_mutation = self.mutate_from_performance(
            agent_prompt, agent_eval_a, self.agent_mutation_prompt
        )
        agent_prompt_b = agent_mutation.improved_prompt

        # Save agent prompt version with diff
        run_id = getattr(self, "_current_run_id", "unknown")
        cycle_num = getattr(self, "_current_cycle", 0)
        self.version_manager.save_prompt_with_diff(
            prompt_type="agent",
            new_prompt=agent_prompt_b,
            old_prompt=agent_prompt,
            run_id=run_id,
            cycle=cycle_num,
            step="optimized",
            show_diff=True,
        )

        # Step 4: Optimize sim prompt A→B
        self.progress.print("🎯 Optimizing simulation prompt...")
        sim_mutation = self.mutate_from_performance(
            sim_prompt, sim_eval_a, self.sim_mutation_prompt
        )
        sim_prompt_b = sim_mutation.improved_prompt

        # Save simulation prompt version with diff
        self.version_manager.save_prompt_with_diff(
            prompt_type="simulation",
            new_prompt=sim_prompt_b,
            old_prompt=sim_prompt,
            run_id=run_id,
            cycle=cycle_num,
            step="optimized",
            show_diff=True,
        )

        # Step 5: Generate conversation B
        self.progress.print("📝 Generating conversation B...")
        conversation_b = self.generate_conversation(
            scenario, agent_prompt_b, sim_prompt_b
        )

        # Show conversation B
        self._display_conversation("Generated Conversation B", conversation_b)

        # Step 6: Evaluate conversation B
        self.progress.print("📊 Evaluating conversation B...")
        agent_eval_b, sim_eval_b = self.evaluate_conversation(
            conversation_b, scenario, agent_eval_prompt, sim_eval_prompt
        )

        # Show evaluation results clearly
        self._display_evaluation_results("Agent Evaluation B", agent_eval_b)
        self._display_evaluation_results("Simulation Evaluation B", sim_eval_b)

        # Create conversation pair
        pair = ConversationPair(
            conversation_a=conversation_a,
            conversation_b=conversation_b,
            scenario=scenario,
            agent_prompt_a=agent_prompt,
            agent_prompt_b=agent_prompt_b,
            sim_prompt_a=sim_prompt,
            sim_prompt_b=sim_prompt_b,
            agent_eval_prompt=agent_eval_prompt,
            sim_eval_prompt=sim_eval_prompt,
            agent_eval_a=agent_eval_a,
            agent_eval_b=agent_eval_b,
            sim_eval_a=sim_eval_a,
            sim_eval_b=sim_eval_b,
        )

        # Step 7: Get user preferences (feedback or learned)
        self.progress.print("\n👤 Determining whether to collect user feedback...")
        feedback_sessions = 0
        agent_preference, sim_preference = self.get_user_preferences(pair)

        if agent_preference or sim_preference:
            feedback_sessions = 1  # We collected feedback
            pair.user_prefers_agent = agent_preference
            pair.user_prefers_sim = sim_preference

        # Steps 8-9: Check evaluator alignment
        agent_eval_score_a = agent_eval_a.overall_score
        agent_eval_score_b = agent_eval_b.overall_score
        agent_eval_ranking = (
            "A"
            if agent_eval_score_a > agent_eval_score_b
            else "B" if agent_eval_score_b > agent_eval_score_a else "similar"
        )

        sim_eval_score_a = sim_eval_a.overall_score
        sim_eval_score_b = sim_eval_b.overall_score
        sim_eval_ranking = (
            "A"
            if sim_eval_score_a > sim_eval_score_b
            else "B" if sim_eval_score_b > sim_eval_score_a else "similar"
        )

        agent_eval_correct = (
            (pair.user_prefers_agent == agent_eval_ranking)
            if pair.user_prefers_agent
            else True
        )
        sim_eval_correct = (
            (pair.user_prefers_sim == sim_eval_ranking)
            if pair.user_prefers_sim
            else True
        )

        self.progress.print(
            f"🎯 Agent evaluator {'✅ correct' if agent_eval_correct else '❌ incorrect'}"
        )
        self.progress.print(
            f"🎯 Simulation evaluator {'✅ correct' if sim_eval_correct else '❌ incorrect'}"
        )

        # Steps 10-11: Optimize evaluators if wrong
        new_agent_eval_prompt = None
        new_sim_eval_prompt = None

        if not agent_eval_correct and pair.user_prefers_agent:
            self.progress.print("🔧 Optimizing agent evaluator...")
            agent_eval_mutation = self.mutate_from_alignment(
                agent_eval_prompt,
                pair.user_prefers_agent,
                agent_eval_ranking,
                "Based on user feedback",  # TODO: Get actual reasoning
                self.agent_eval_mutation_prompt,
            )
            new_agent_eval_prompt = agent_eval_mutation.improved_prompt

            # Save evaluator prompt version with diff
            self.version_manager.save_prompt_with_diff(
                prompt_type="agent_eval",
                new_prompt=new_agent_eval_prompt,
                old_prompt=agent_eval_prompt,
                run_id=run_id,
                cycle=cycle_num,
                step="evaluator_optimized",
                show_diff=True,
            )

        if not sim_eval_correct and pair.user_prefers_sim:
            self.progress.print("🔧 Optimizing simulation evaluator...")
            sim_eval_mutation = self.mutate_from_alignment(
                sim_eval_prompt,
                pair.user_prefers_sim,
                sim_eval_ranking,
                "Based on user feedback",
                self.sim_eval_mutation_prompt,
            )
            new_sim_eval_prompt = sim_eval_mutation.improved_prompt

            # Save simulation evaluator prompt version with diff
            self.version_manager.save_prompt_with_diff(
                prompt_type="sim_eval",
                new_prompt=new_sim_eval_prompt,
                old_prompt=sim_eval_prompt,
                run_id=run_id,
                cycle=cycle_num,
                step="evaluator_optimized",
                show_diff=True,
            )

        # Step 12: Test new evaluators
        updated_prompts = {
            "agent": agent_prompt_b,
            "simulation": sim_prompt_b,
            "agent_eval": agent_eval_prompt,
            "sim_eval": sim_eval_prompt,
        }

        if new_agent_eval_prompt or new_sim_eval_prompt:
            self.progress.print("🧪 Testing new evaluators...")
            agent_comparison, sim_comparison = self.compare_evaluators(
                pair, new_agent_eval_prompt, new_sim_eval_prompt
            )

            # Accept better evaluators
            if (
                agent_comparison
                and agent_comparison.recommendation == "new"
                and new_agent_eval_prompt
            ):
                self.progress.print("✅ Accepting new agent evaluator")
                updated_prompts["agent_eval"] = new_agent_eval_prompt

            if (
                sim_comparison
                and sim_comparison.recommendation == "new"
                and new_sim_eval_prompt
            ):
                self.progress.print("✅ Accepting new simulation evaluator")
                updated_prompts["sim_eval"] = new_sim_eval_prompt

        # Save conversation pair
        self.conversation_pairs.append(pair)

        self.progress.print(f"✅ Optimization cycle complete")

        return updated_prompts, feedback_sessions

    def run_sequential_optimization(
        self, scenario: str, max_cycles: int = 3
    ) -> SequentialOptimizationRun:
        """Run sequential optimization for multiple cycles"""

        run_id = f"sequential_{int(time.time())}"
        start_time = time.time()

        # Set current run context for version tracking
        self._current_run_id = run_id
        self._current_cycle = 0

        self.progress.print(f"\n🚀 Starting Sequential Optimization (Run: {run_id})")
        self.progress.print(f"Scenario: {scenario}")
        self.progress.print(f"Max cycles: {max_cycles}")

        # Step 0: Collect initial optimization goals from user
        optimization_goals = self._collect_optimization_goals(scenario)

        # Use goals to seed preferences and customize prompts
        self._apply_optimization_goals(optimization_goals)

        # Get initial prompts
        eval_config = self.domain_config.get_evaluation_config()
        agent_config = self.domain_config.get_agent_config()

        initial_prompts = {
            "agent": agent_config.prompt_template,
            "simulation": eval_config.simulation_prompt_template,
            "agent_eval": eval_config.evaluation_prompt_template,
            "sim_eval": eval_config.simulation_evaluation_prompt_template,
        }

        # Save initial prompt versions
        self.progress.print(f"\n💾 Saving initial prompt versions...")
        for prompt_type, prompt_content in initial_prompts.items():
            self.version_manager.save_prompt_version(
                prompt_type=prompt_type,
                prompt=prompt_content,
                run_id=run_id,
                cycle=0,
                step="initial",
                metadata={"optimization_goals": optimization_goals},
            )

        current_prompts = initial_prompts.copy()
        total_feedback_sessions = 0
        evaluator_optimizations = {"agent_eval": False, "sim_eval": False}

        # Run optimization cycles
        for cycle in range(max_cycles):
            # Update current cycle for version tracking
            self._current_cycle = cycle + 1

            self.progress.print(f"\n{'='*60}")
            self.progress.print(f"CYCLE {cycle + 1}/{max_cycles}")
            self.progress.print(f"{'='*60}")

            updated_prompts, feedback_sessions = self.run_optimization_cycle(
                scenario,
                current_prompts["agent"],
                current_prompts["simulation"],
                current_prompts["agent_eval"],
                current_prompts["sim_eval"],
            )

            # Track changes
            if updated_prompts["agent_eval"] != current_prompts["agent_eval"]:
                evaluator_optimizations["agent_eval"] = True
            if updated_prompts["sim_eval"] != current_prompts["sim_eval"]:
                evaluator_optimizations["sim_eval"] = True

            current_prompts = updated_prompts
            total_feedback_sessions += feedback_sessions

        # Create run record
        duration = time.time() - start_time

        run_record = SequentialOptimizationRun(
            run_id=run_id,
            domain=self.domain_config.get_evaluation_config().domain_name,
            scenario=scenario,
            initial_prompts=initial_prompts,
            final_prompts=current_prompts,
            conversation_pairs=len(self.conversation_pairs),
            feedback_sessions=total_feedback_sessions,
            evaluator_optimizations=evaluator_optimizations,
            success=True,  # TODO: Define success criteria
            duration_seconds=duration,
        )

        self.optimization_history.append(run_record)

        # Save final optimized prompts to files (backward compatibility)
        self.progress.print(f"\n💾 Saving final optimized prompts...")
        for prompt_type, prompt_content in current_prompts.items():
            if prompt_content != initial_prompts.get(prompt_type):
                # Save versioned final prompt
                self.version_manager.save_prompt_version(
                    prompt_type=prompt_type,
                    prompt=prompt_content,
                    run_id=run_id,
                    cycle=max_cycles,
                    step="final",
                    metadata={
                        "total_cycles": max_cycles,
                        "feedback_sessions": total_feedback_sessions,
                    },
                )

                # Also save to standard location for backward compatibility
                saved_file = self.path_manager.save_optimized_prompt(
                    prompt_type, prompt_content
                )
                self.progress.print(f"   ✅ Saved {prompt_type}: {saved_file}")
            else:
                self.progress.print(f"   ➖ {prompt_type}: No changes")

        # Generate evolution reports for each prompt type
        self.progress.print(f"\n📊 Generating prompt evolution reports...")
        for prompt_type in ["agent", "simulation", "agent_eval", "sim_eval"]:
            versions = self.version_manager.get_prompt_versions(prompt_type, run_id)
            if len(versions) > 1:
                report_file = self.version_manager.save_evolution_report(
                    prompt_type, run_id
                )
                self.progress.print(
                    f"   📋 {prompt_type} evolution: {report_file.name}"
                )

        # Generate and display final summary
        final_summary = self.version_manager.generate_final_summary(run_id)
        self.progress.print(f"\n{final_summary}")

        self.progress.print(f"\n🎉 Sequential Optimization Complete!")
        self.progress.print(f"Duration: {duration:.1f}s")
        self.progress.print(f"Conversation pairs: {len(self.conversation_pairs)}")
        self.progress.print(f"Feedback sessions: {total_feedback_sessions}")
        self.progress.print(f"Evaluator optimizations: {evaluator_optimizations}")

        self.progress.print(
            f"\n📁 All prompt versions and evolution reports saved to: {self.path_manager.paths.optimized_prompts_dir}"
        )

        return run_record

    def _display_conversation(self, title: str, conversation: List[Dict[str, str]]):
        """Display a conversation in a readable format"""
        self.progress.print(f"\n{'='*60}")
        self.progress.print(f"📖 {title}")
        self.progress.print(f"{'='*60}")

        for i, msg in enumerate(conversation, 1):
            role = msg["role"].upper()
            content = msg["content"]

            # Truncate very long content for readability
            if len(content) > 300:
                content = content[:300] + "..."

            self.progress.print(f"\n{i}. {role}:")
            self.progress.print(f"   {content}")

        self.progress.print(f"\n📊 Conversation Summary: {len(conversation)} turns")

    def _display_evaluation_results(self, title: str, evaluation):
        """Display evaluation results clearly"""
        self.progress.print(f"\n{'='*60}")
        self.progress.print(f"📋 {title}")
        self.progress.print(f"{'='*60}")

        self.progress.print(f"🎯 Overall Score: {evaluation.overall_score}/10")
        self.progress.print(f"\n📝 Detailed Feedback:")
        self.progress.print(f"   {evaluation.detailed_feedback}")

        self.progress.print(f"\n✅ Strengths:")
        for strength in evaluation.strengths:
            self.progress.print(f"   • {strength}")

        self.progress.print(f"\n⚠️  Weaknesses:")
        for weakness in evaluation.weaknesses:
            self.progress.print(f"   • {weakness}")

        self.progress.print(f"\n🔍 Key Observations:")
        for obs in evaluation.key_observations:
            self.progress.print(f"   • {obs}")

    def _display_interruption_decision(self, decision, context):
        """Display interruption system reasoning transparently"""
        self.progress.print(f"\n{'='*60}")
        self.progress.print(f"🤖 Interruption System Decision")
        self.progress.print(f"{'='*60}")

        self.progress.print(
            f"🎯 Decision: {'INTERRUPT for user feedback' if decision.should_interrupt else 'CONTINUE autonomously'}"
        )

        if decision.reason:
            self.progress.print(f"📍 Primary Reason: {decision.reason.value}")

        self.progress.print(f"⚡ Urgency Level: {decision.urgency}")
        self.progress.print(f"🎲 Confidence: {decision.confidence:.2f}")

        self.progress.print(f"\n💭 Reasoning:")
        self.progress.print(f"   {decision.reasoning}")

        if decision.should_interrupt:
            self.progress.print(f"\n✨ Expected Benefit:")
            self.progress.print(f"   {decision.expected_benefit}")

            self.progress.print(f"\n⚠️  Risk if No Feedback:")
            self.progress.print(f"   {decision.risk_if_no_feedback}")

        self.progress.print(f"\n📊 Context:")
        self.progress.print(f"   • Current iteration: {context.current_iteration}")
        self.progress.print(
            f"   • Iterations without improvement: {context.iterations_without_improvement}"
        )
        self.progress.print(
            f"   • Total feedback sessions: {context.total_feedback_sessions}"
        )
        self.progress.print(
            f"   • Preference confidence: {context.confidence_level:.2f}"
        )

    def _collect_optimization_goals(self, scenario: str) -> Dict[str, Any]:
        """Collect initial optimization goals from user through simple feedback"""
        self.progress.print(f"\n{'='*80}")
        self.progress.print(f"🎯 OPTIMIZATION GOALS COLLECTION")
        self.progress.print(f"{'='*80}")
        self.progress.print(f"Scenario: {scenario}")
        self.progress.print()
        self.progress.print(
            "Before we begin optimizing, please share your goals for improvement."
        )
        self.progress.print(
            "What aspects of the current agent behavior would you like to optimize?"
        )
        self.progress.print()

        # Collect current strengths
        self.progress.print("1. What aspects of the current agent behavior work well?")
        self.progress.print(
            "   (e.g., 'Good character consistency', 'Natural dialogue flow', etc.)"
        )
        current_strengths = self.progress.input("> ").strip()

        self.progress.print()
        self.progress.print("2. What aspects need improvement?")
        self.progress.print(
            "   (e.g., 'More engaging responses', 'Better tool usage', 'More personality', etc.)"
        )
        improvement_areas = self.progress.input("> ").strip()

        self.progress.print()
        self.progress.print("3. Any specific behavioral issues you've noticed?")
        self.progress.print(
            "   (e.g., 'Too repetitive', 'Breaks character', 'Responses too short', etc.)"
        )
        specific_issues = self.progress.input("> ").strip()

        self.progress.print()
        self.progress.print("4. What would ideal agent behavior look like for you?")
        self.progress.print(
            "   (e.g., 'Immersive roleplay', 'Helpful and engaging', 'Creative responses', etc.)"
        )
        ideal_behavior = self.progress.input("> ").strip()

        goals = {
            "scenario": scenario,
            "current_strengths": current_strengths or "No specific strengths mentioned",
            "improvement_areas": improvement_areas or "General improvement desired",
            "specific_issues": specific_issues or "No specific issues mentioned",
            "ideal_behavior": ideal_behavior or "Better overall performance",
            "timestamp": time.time(),
        }

        self.progress.print(f"\n✅ Optimization goals collected!")
        self.progress.print(f"   Current strengths: {goals['current_strengths']}")
        self.progress.print(f"   Areas for improvement: {goals['improvement_areas']}")
        self.progress.print(f"   Specific issues: {goals['specific_issues']}")
        self.progress.print(f"   Ideal behavior: {goals['ideal_behavior']}")

        return goals

    def _apply_optimization_goals(self, goals: Dict[str, Any]):
        """Apply optimization goals to seed preferences"""
        self.progress.print(
            f"\n🔧 Applying optimization goals to seed the preference system..."
        )

        # Create structured feedback from goals to seed preferences
        feedback_text = f"""
        User optimization goals:
        - Current strengths to preserve: {goals['current_strengths']}
        - Areas needing improvement: {goals['improvement_areas']} 
        - Specific issues to address: {goals['specific_issues']}
        - Ideal behavior target: {goals['ideal_behavior']}
        """

        # Add to preference manager as initial seed
        context = {
            "type": "optimization_goals",
            "scenario": goals["scenario"],
            "timestamp": goals["timestamp"],
        }

        self.prefs.add_feedback(feedback_text.strip(), context, "conversation")

        # Store goals for reference during optimization
        if not hasattr(self, "optimization_goals"):
            self.optimization_goals = []
        self.optimization_goals.append(goals)

        self.progress.print(f"   ✅ Goals integrated into preference system")
        self.progress.print(f"   ✅ These goals will guide the optimization process")


def main():
    """Test the sequential optimization system"""
    print("=== SEQUENTIAL OPTIMIZATION TEST ===")

    from agent.eval.domains.roleplay import RoleplayEvaluationConfig
    from agent.eval.preferences import SemanticPreferenceManager
    from agent.llm import create_llm, SupportedModel

    domain_config = RoleplayEvaluationConfig()
    prefs = SemanticPreferenceManager(
        llm=create_llm(),
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO,
        progress_reporter=NullProgressReporter(),
        preferences_dir="test_sequential_prefs",
    )
    llm = create_llm()

    optimizer = SequentialOptimizer(
        domain_config,
        prefs,
        llm,
        SupportedModel.DOLPHIN_MISTRAL_NEMO,
        NullProgressReporter(),
        path_manager=OptimizationPathManager(
            base_dir="test_sequential_optimization", domain="roleplay"
        ),
    )

    # Test with a single scenario
    scenario = "Roleplay as Elena, a mysterious vampire who owns an ancient castle"

    result = optimizer.run_sequential_optimization(scenario, max_cycles=2)

    print(f"\n📊 Results:")
    print(f"Run ID: {result.run_id}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Feedback sessions: {result.feedback_sessions}")

    print("\nSequential optimization test complete!")


if __name__ == "__main__":
    main()
