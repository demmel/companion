"""Interactive CLI for code-only agent."""

from agent.experiments.code_only_agent.agent import run_agent
from agent.experiments.code_only_agent.state import State
from agent.llm.models import SupportedModel
from agent.llm.router import create_llm
from agent.ui_output import ui_print


def main():
    """Interactive CLI interface."""
    state = State()
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    while True:
        user_input = input("> ")
        if user_input.lower() == "/exit":
            ui_print("Exiting the program. Goodbye!")
            break

        turn = run_agent(user_input, state, llm, model)

        # Display results
        for i, iteration in enumerate(turn.iterations, 1):
            ui_print(f"\n--- Iteration {i} ---")
            ui_print(f"Reasoning: {iteration.reasoning}")
            if iteration.code:
                ui_print(f"\nCode:\n{iteration.code}")
            if iteration.output:
                ui_print(f"\nOutput: {iteration.output}")


if __name__ == "__main__":
    main()
