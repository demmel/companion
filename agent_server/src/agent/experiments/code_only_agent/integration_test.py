"""Integration test for code_only_agent - actually runs the agent with LLM."""

from agent.experiments.code_only_agent.agent import run_agent
from agent.experiments.code_only_agent.state import State
from agent.llm.models import SupportedModel
from agent.llm.router import create_llm
from agent.ui_output import ui_print


def test_agent_executes_code():
    """Test that the agent actually executes code."""
    ui_print("Creating state...")
    state = State()
    ui_print("Creating LLM...")
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    ui_print(f"Using model: {model}")

    user_input = "What time is it? Then list the files in the current directory and tell me how many there are."

    ui_print(f"Running agent with input: {user_input}")
    turn = run_agent(user_input, state, llm, model)

    ui_print(f"\nAgent completed {len(turn.iterations)} iteration(s)")

    code_executed = False
    for i, iteration in enumerate(turn.iterations, 1):
        ui_print(f"\n--- Iteration {i} ---")
        ui_print(
            f"Reasoning: {iteration.reasoning[:200]}..."
        )  # Truncate long reasoning
        if iteration.code:
            ui_print(f"Code: {iteration.code}")
            code_executed = True
        if iteration.outputs:
            ui_print(f"Outputs:")
            for output in iteration.outputs:
                from agent.experiments.code_only_agent.execution import (
                    format_output_message,
                )

                ui_print(f"  {format_output_message(output)}")

    # Assertions
    assert len(turn.iterations) > 0, "Agent should have at least one iteration"
    assert turn.user_input == user_input
    assert code_executed, "Agent should have executed at least one code block"

    speaks = turn.get_speaks()
    assert len(speaks) > 0, "Agent should have spoken to the user"

    ui_print(f"\n✓ Test completed - code executed: {code_executed}")
    ui_print(f"✓ Agent spoke {len(speaks)} time(s)")
    ui_print(f"Messages: {speaks}")


if __name__ == "__main__":
    test_agent_executes_code()
