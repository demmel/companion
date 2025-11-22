"""Main agent loop for code-only agent."""

import logging

from agent.chain_of_action.prompts import format_section
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.experiments.code_only_agent.state import State, AgentTurn, Iteration
from agent.experiments.code_only_agent.execution import parse_code_blocks, execute_code
from agent.experiments.code_only_agent.functions import initialize_base_functions

logger = logging.getLogger(__name__)


def process_user_input(
    user_input: str, state: State, llm: LLM, model: SupportedModel
) -> AgentTurn:
    """
    Process user input by running agent in iteration loop.

    Returns an AgentTurn containing all iterations.
    """
    turn = AgentTurn(user_input=user_input, iterations=[])
    max_iterations = 10

    # Get available functions from cache
    available_functions = state.function_cache.get_all_funcs()

    for _ in range(max_iterations):
        # Build prompt
        sections = []

        # Add state (includes history)
        state_section = state.format_for_llm()
        if state_section:
            sections.append(state_section)

        # Add current user input
        sections.append(format_section("User Input", user_input))

        # Add current turn iterations so far
        if turn.iterations:
            iter_lines = []
            for i, iteration in enumerate(turn.iterations, 1):
                iter_lines.append(f"Iteration {i}:")
                iter_lines.append(f"  Reasoning: {iteration.reasoning}")
                if iteration.code:
                    iter_lines.append(f"  Code: {iteration.code}")
                if iteration.output:
                    iter_lines.append(f"  Output: {iteration.output}")
            sections.append(format_section("This Turn So Far", "\n".join(iter_lines)))

        # Add available functions from cache
        sections.append(
            format_section(
                "Available Functions",
                state.function_cache.format_for_prompt(),
            )
        )

        # Add instructions
        sections.append(
            format_section(
                "Instructions",
                """Look at "This Turn So Far" to see what you've already done this turn.

Output your INTERNAL reasoning about what still needs to be done. This is for you, not the user.

If you still need to execute code:
- Include it in a ```python code block

If the task is complete (you've already used speak() or accomplished the goal):
- Output ONLY reasoning explaining the task is done
- DO NOT include ANY code blocks
- No code block = you are signaling completion

IMPORTANT: Your reasoning is internal thoughts, NOT a response to the user.""",
            )
        )

        prompt = f"""You are an agent that can ONLY interact with the world through Python code execution.

You cannot directly respond to the user - you can only write and execute code. All actions, including communicating with the user, must be done through code.

{"\n\n".join(sections)}
"""

        # Get LLM response
        logger.debug(f"Calling LLM for iteration {len(turn.iterations) + 1}")
        llm_response = llm.generate_complete(
            model=model, prompt=prompt, caller="code_agent"
        )
        logger.debug(f"Got LLM response: {llm_response[:100]}...")

        # Parse response
        reasoning, code_blocks = parse_code_blocks(llm_response)
        logger.debug(
            f"Parsed - reasoning: {len(reasoning)} chars, code blocks: {len(code_blocks)}"
        )

        # If no code, agent is done
        if not code_blocks:
            turn.iterations.append(
                Iteration(reasoning=reasoning, code=None, output=None)
            )
            break

        # Execute first code block (ignore others for now)
        code = code_blocks[0]
        output = execute_code(code, available_functions=available_functions)

        # Record iteration
        turn.iterations.append(Iteration(reasoning=reasoning, code=code, output=output))

    return turn


def run_agent(
    user_input: str, state: State, llm: LLM, model: SupportedModel
) -> AgentTurn:
    """
    Run the agent programmatically for a single turn.

    Args:
        user_input: The user's input
        state: Current state (will be modified to include this turn in history)
        llm: LLM instance
        model: Model to use

    Returns:
        The AgentTurn containing all iterations
    """
    # Initialize base functions if not already done
    if not state.function_cache.cache:
        initialize_base_functions(state)

    turn = process_user_input(user_input, state, llm, model)
    state.history.append(turn)
    return turn
