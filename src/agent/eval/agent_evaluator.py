"""
AgentEvaluator - Level 1 of the optimization hierarchy

The core evaluation engine that runs conversations to test prompts.
All higher-level optimizers use this to evaluate prompt quality.
"""

import json
import time
import re
from typing import List, Dict, Any, Optional
import ollama

from agent.core import Agent, message_to_llm_messages
from agent.config import AgentConfig
from .base import DomainEvaluationConfig, EvaluationResult


class AgentEvaluator:
    """General framework for evaluating any agent configuration"""

    def __init__(
        self,
        domain_eval_config: DomainEvaluationConfig,
        user_simulator_model: str = "huihui_ai/mistral-small-abliterated",
        evaluator_model: str = "huihui_ai/mistral-small-abliterated",
        agent_model: str = "huihui_ai/mistral-small-abliterated",
    ):

        self.domain_eval_config = domain_eval_config
        self.eval_config = domain_eval_config.get_evaluation_config()
        self.agent_config = domain_eval_config.get_agent_config()

        self.user_simulator_model = user_simulator_model
        self.evaluator_model = evaluator_model
        self.agent_model = agent_model

        # Create agent instance
        self.agent = Agent(config=self.agent_config, model=agent_model)

    def get_initial_user_input(self, scenario: str) -> str:
        """Get the initial user input to start the scenario using domain-specific template"""
        prompt = self.eval_config.initial_prompt_template.format(scenario=scenario)

        response = ollama.chat(
            model=self.user_simulator_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response["message"]["content"].strip()

    def get_user_response(
        self, scenario: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Get user simulator's response to the current conversation"""

        # Build conversation for the simulator using the configured prompt template
        simulation_prompt = self.eval_config.simulation_prompt_template.format(
            scenario=scenario
        )

        messages = [{"role": "system", "content": simulation_prompt}]

        # Add conversation history (flip roles from user simulator's perspective)
        for msg in conversation_history:
            role = "assistant" if msg["role"] == "user" else "user"  # Flip the roles
            messages.append({"role": role, "content": msg["content"]})

        response = ollama.chat(model=self.user_simulator_model, messages=messages)
        return response["message"]["content"].strip()

    def run_conversation(self, scenario: str) -> List[Dict[str, str]]:
        """Run interactive conversation between simulated user and agent"""
        num_turns = self.eval_config.num_conversation_turns
        print(f"  Running {num_turns}-turn interactive conversation...")

        conversation = []

        # Get initial user input
        print(f"    Getting initial user input...")
        initial_input = self.get_initial_user_input(scenario)
        print(f"    Initial input: {initial_input[:60]}...")

        conversation.append({"role": "user", "content": initial_input})

        for turn in range(num_turns):
            # Get agent response to latest user input
            latest_user_input = conversation[-1]["content"]
            print(f"    Turn {turn+1}: Getting agent response...")

            start_time = time.time()
            # Get conversation length before agent response
            history_before = len(self.agent.get_conversation_history())

            # Trigger agent processing and wait for completion
            list(self.agent.chat_stream(latest_user_input))  # Consume the stream
            response_time = time.time() - start_time

            # Get new messages added to conversation history
            conversation_history = self.agent.get_conversation_history()
            new_messages = conversation_history[history_before:]

            # Get only agent messages and convert to text format
            full_agent_turn = ""
            for msg in new_messages:
                if msg.role == "assistant":  # Only include agent messages
                    for llm_msg in message_to_llm_messages(msg):
                        full_agent_turn += llm_msg.content + "\n"
            full_agent_turn = full_agent_turn.strip()

            print(
                f"    Turn {turn+1}: Agent turn ({len(full_agent_turn)} chars, {response_time:.1f}s)"
            )
            conversation.append({"role": "agent", "content": full_agent_turn})

            # Get user response (except on last turn)
            if turn < num_turns - 1:
                print(f"    Turn {turn+1}: Getting user response...")
                user_response = self.get_user_response(scenario, conversation)
                print(f"    Turn {turn+1}: User response: {user_response[:60]}...")
                conversation.append({"role": "user", "content": user_response})

            # Brief pause between turns
            time.sleep(1)

        return conversation

    def evaluate_conversation(
        self, scenario: str, conversation: List[Dict[str, str]]
    ) -> EvaluationResult:
        """Evaluate the conversation using the configured evaluation prompt"""
        print(f"  Evaluating conversation...")

        # Build conversation text
        conv_text = "\n\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation]
        )

        # Get agent context
        agent_state = self.agent.get_state()
        assert isinstance(agent_state, dict), "Agent state should be a dictionary"
        agent_context = self.domain_eval_config.extract_conversation_context(
            agent_state
        )

        # Build evaluation prompt using the configured template
        evaluation_prompt = self.eval_config.evaluation_prompt_template.format(
            scenario=scenario, conversation=conv_text, agent_context=agent_context
        )

        response = ollama.chat(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            options={"num_predict": 4096},
        )

        response_text = response["message"]["content"]

        try:
            # Extract and parse JSON
            json_text = self._extract_json_from_response(response_text)

            if not json_text:
                raise ValueError("No valid JSON found in response")

            # Fix common JSON formatting issues
            json_text = self._fix_json_formatting(json_text)

            result_data = json.loads(json_text)

            return EvaluationResult(
                config_name=self.agent_config.name,
                scenario=scenario,
                conversation=conversation,
                scores=result_data.get("scores", {}),
                feedback=result_data.get("feedback", "No feedback provided"),
                suggested_improvements=result_data.get("suggested_improvements", []),
                overall_score=result_data.get("overall_score", 0.0),
            )

        except Exception as e:
            print(f"  Error parsing evaluator response: {e}")
            print(f"  Raw response: {response_text[:500]}...")
            return EvaluationResult(
                config_name=self.agent_config.name,
                scenario=scenario,
                conversation=conversation,
                scores={},
                feedback=f"Evaluation error: {str(e)}",
                suggested_improvements=[],
                overall_score=0.0,
            )

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON from LLM response, handling various formats"""
        json_text = None

        # Method 1: Look for JSON after </think> tag (for reasoning models)
        if "</think>" in response_text:
            post_think = response_text.split("</think>")[-1]
            if "{" in post_think:
                json_start = post_think.index("{")
                json_end = post_think.rindex("}") + 1
                json_text = post_think[json_start:json_end]

        # Method 2: Look for the last complete JSON object in the response
        if not json_text and "{" in response_text:
            brace_count = 0
            for i, char in enumerate(response_text):
                if char == "{":
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = response_text[start_pos : i + 1]
                        try:
                            json.loads(potential_json)
                            json_text = potential_json
                        except:
                            continue

        return json_text

    def _fix_json_formatting(self, json_text: str) -> str:
        """Fix common JSON formatting issues from LLMs"""
        json_text = json_text.strip()

        # Fix missing spaces after colons
        json_text = re.sub(r'":([0-9\[\{"])', r'": \1', json_text)

        # Remove trailing commas before closing braces
        json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)

        return json_text

    def run_evaluation(self, scenario: str) -> EvaluationResult:
        """Run complete evaluation for a scenario"""
        print(
            f"\nEvaluating {self.eval_config.domain_name} scenario: {scenario[:60]}..."
        )

        # Reset agent state for new conversation
        self.agent = Agent(config=self.agent_config, model=self.agent_model)

        # Run interactive conversation
        conversation = self.run_conversation(scenario)

        # Evaluate conversation
        result = self.evaluate_conversation(scenario, conversation)

        print(f"  Overall score: {result.overall_score:.1f}/10")
        return result

    def run_evaluation_suite(self) -> List[EvaluationResult]:
        """Run evaluation suite across all configured scenarios"""
        results = []
        scenarios = self.eval_config.test_scenarios

        for i, scenario in enumerate(scenarios):
            print(f"\n=== Evaluation {i+1}/{len(scenarios)} ===")
            result = self.run_evaluation(scenario)
            results.append(result)

            # Save individual result
            with open(
                f"eval_{self.eval_config.domain_name}_{i+1}_result.json", "w"
            ) as f:
                json.dump(
                    {
                        "config_name": result.config_name,
                        "scenario": result.scenario,
                        "conversation": result.conversation,
                        "scores": result.scores,
                        "feedback": result.feedback,
                        "suggested_improvements": result.suggested_improvements,
                        "overall_score": result.overall_score,
                    },
                    f,
                    indent=2,
                )

            # Pause between evaluations
            time.sleep(3)

        return results

    def analyze_results(self, results: List[EvaluationResult]):
        """Analyze and present evaluation results"""
        print(f"\n{'='*60}")
        print(f"{self.eval_config.domain_name.upper()} AGENT EVALUATION RESULTS")
        print(f"{'='*60}")

        if not results:
            print("No results to analyze.")
            return

        # Calculate average scores
        criteria = self.eval_config.evaluation_criteria

        print(f"\nAVERAGE SCORES:")
        print(f"{'Criteria':<20} {'Score':<10} {'Status':<10}")
        print("-" * 40)

        avg_scores = {}
        for criterion in criteria:
            scores = [r.scores.get(criterion, 0) for r in results if r.scores]
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_scores[criterion] = avg_score
                status = (
                    "GOOD"
                    if avg_score >= 7
                    else "NEEDS WORK" if avg_score >= 5 else "POOR"
                )
                print(f"{criterion:<20} {avg_score:<10.1f} {status:<10}")

        # Overall performance
        overall_scores = [r.overall_score for r in results if r.overall_score]
        if overall_scores:
            avg_overall = sum(overall_scores) / len(overall_scores)
            print(f"\nOVERALL AVERAGE: {avg_overall:.1f}/10")

        # Common improvement suggestions
        print(f"\nCOMMON IMPROVEMENT SUGGESTIONS:")
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.suggested_improvements)

        # Count suggestion frequency
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        # Show most common suggestions
        sorted_suggestions = sorted(
            suggestion_counts.items(), key=lambda x: x[1], reverse=True
        )
        for suggestion, count in sorted_suggestions[:5]:
            print(f"  • {suggestion} (mentioned {count} times)")

        # Show sample feedback
        print(f"\nSAMPLE DETAILED FEEDBACK:")
        for i, result in enumerate(results[:2]):
            if result.feedback and "error" not in result.feedback.lower():
                print(f"\nScenario {i+1}: {result.scenario[:50]}...")
                print(f"Feedback: {result.feedback}")
