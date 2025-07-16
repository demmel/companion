"""
AgentEvaluator - Level 1 of the optimization hierarchy

The core evaluation engine that runs conversations to test prompts.
All higher-level optimizers use this to evaluate prompt quality.
"""

import logging
import time
from typing import List, Dict
from rich.console import Console
from agent.config import AgentConfig
from agent.llm import LLM, SupportedModel, Message
from agent.core import Agent, message_to_llm_messages
from agent.progress import ProgressReporter

logger = logging.getLogger(__name__)
console = Console()


class ConversationGenerator:
    """General framework for evaluating any agent configuration"""

    def __init__(
        self,
        simulation_prompt_template: str,
        num_conversation_turns: int,
        agent_config: AgentConfig,
        model: SupportedModel,
        llm: LLM,
        progress: ProgressReporter,
    ):
        self.model = model
        self.llm = llm
        self.progress = progress
        self.simulation_prompt_template = simulation_prompt_template
        self.num_conversation_turns = num_conversation_turns

        self.agent_config = agent_config

        # Create agent instance
        self.agent = Agent(config=self.agent_config, model=model, llm=llm)

    def get_initial_user_input(self, scenario: str) -> str:
        """Get the initial user input to start the scenario using domain-specific template"""
        prompt = self.simulation_prompt_template.format(scenario=scenario)

        response = self.llm.chat_complete(
            model=self.model,
            messages=[Message(role="user", content=prompt)],
        )

        return response.strip() if response else ""

    def get_user_response(
        self, scenario: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Get user simulator's response to the current conversation"""

        # Build conversation for the simulator using the configured prompt template
        simulation_prompt = self.simulation_prompt_template.format(scenario=scenario)

        messages = [Message(role="system", content=simulation_prompt)]

        # Add conversation history (flip roles from user simulator's perspective)
        for msg in conversation_history:
            role = "assistant" if msg["role"] == "user" else "user"  # Flip the roles
            messages.append(Message(role=role, content=msg["content"]))

        response = self.llm.chat_complete(
            model=self.model,
            messages=messages,
        )
        return response.strip() if response else ""

    def generate_conversation(self, scenario: str) -> List[Dict[str, str]]:
        """Run interactive conversation between simulated user and agent"""
        num_turns = self.num_conversation_turns

        with self.progress.task(
            f"Running {num_turns}-turn conversation", total=num_turns
        ) as task:

            conversation = []

            # Get initial user input
            logger.debug(f"Getting initial user input...")
            initial_input = self.get_initial_user_input(scenario)
            logger.debug(f"Initial input: {initial_input[:60]}...")
            conversation.append({"role": "user", "content": initial_input})

            for turn in range(num_turns):
                # Update progress
                task.update(
                    turn / num_turns, f"Turn {turn+1}/{num_turns}: Agent responding..."
                )

                # Get agent response to latest user input
                latest_user_input = conversation[-1]["content"]
                logger.debug(f"Turn {turn+1}: Getting agent response...")

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
                        for llm_msg in message_to_llm_messages(
                            msg, include_thoughts=False
                        ):
                            full_agent_turn += llm_msg.content + "\n"
                full_agent_turn = full_agent_turn.strip()

                logger.debug(
                    f"Turn {turn+1}: Agent turn ({len(full_agent_turn)} chars, {response_time:.1f}s)"
                )
                conversation.append({"role": "agent", "content": full_agent_turn})

                # Get user response (except on last turn)
                if turn < num_turns - 1:
                    task.update(
                        (turn + 0.5) / num_turns,
                        f"Turn {turn+1}/{num_turns}: User responding...",
                    )

                    logger.debug(f"Turn {turn+1}: Getting user response...")
                    user_response = self.get_user_response(scenario, conversation)
                    logger.debug(
                        f"Turn {turn+1}: User response: {user_response[:60]}..."
                    )
                    conversation.append({"role": "user", "content": user_response})

                # Brief pause between turns
                time.sleep(1)

            return conversation
