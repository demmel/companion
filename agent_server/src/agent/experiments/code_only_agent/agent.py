"""Main agent loop for code-only agent."""

import logging

from agent.chain_of_action.prompts import format_section
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.state_initialization import derive_initial_state_from_message
from agent.experiments.code_only_agent.state import (
    State,
    AgentTurn,
    Iteration,
)
from agent.experiments.code_only_agent.execution import (
    parse_code_blocks,
    execute_code,
    ExecutionState,
    format_output_message,
)
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

    # Create execution state for this turn (persists across iterations)
    exec_state = ExecutionState()

    # Initialize safe builtins
    from agent.experiments.code_only_agent.execution import get_safe_builtins

    exec_state.exec_globals["__builtins__"] = get_safe_builtins()

    # Initialize functions (including speak) with execution state
    initialize_base_functions(state, llm, model, exec_state)

    # Add other cached functions to exec_globals
    available_functions = state.function_cache.get_all_funcs()
    exec_state.exec_globals.update(available_functions)

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
                if iteration.outputs:
                    for output in iteration.outputs:
                        iter_lines.append(f"  {format_output_message(output)}")
            sections.append(format_section("This Turn So Far", "\n".join(iter_lines)))

        # Add available functions from cache
        func_tree = state.function_cache.format_for_prompt(state)
        sections.append(
            format_section(
                "Available Functions",
                f"""The tree below shows currently loaded functions organized by category.
Call functions by their name shown in the signature (e.g., speak(...), get_time(), list_files(...)).
DO NOT use paths or dots - just call the function name directly.

{func_tree}""",
            )
        )

        # Add instructions
        sections.append(
            format_section(
                "Instructions",
                """Output your reasoning, then write code in a ```python block.

You have full Python capabilities:
- Loops: for item in items: ...
- List comprehensions: [x for x in items if condition]
- String methods: .split(), .strip(), .endswith(), etc.
- All builtins: len(), sum(), sorted(), filter(), map(), etc.

Try to solve tasks in ONE comprehensive code block when possible.
Variables persist between iterations if you need multiple steps.

Available functions: speak(message), read_file(path), list_files(dir), get_time()
DO NOT use import statements - functions are pre-loaded.

When task is complete: output reasoning with NO code block to signal done.

Your reasoning is internal thoughts, NOT user-facing communication.""",
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
            turn.iterations.append(Iteration(reasoning=reasoning, code=None))
            break

        # Execute first code block (ignore others for now)
        code = code_blocks[0]

        # Reset iteration messages
        exec_state.reset_iteration_messages()

        # Execute code
        execute_code(code, exec_state)

        # Record iteration
        turn.iterations.append(
            Iteration(
                reasoning=reasoning,
                code=code,
                outputs=exec_state.current_iteration_messages,
            )
        )

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
    # Initialize personality on first turn
    if state.is_first_turn():
        main_state, _ = derive_initial_state_from_message(
            first_message=user_input, llm=llm, model=model
        )
        # Copy personality fields from main agent state
        state.name = main_state.name
        state.role = main_state.role
        state.core_values = main_state.core_values
        state.current_priorities = main_state.current_priorities
        state.next_priority_id = main_state.next_priority_id

    turn = process_user_input(user_input, state, llm, model)
    state.history.append(turn)
    return turn
