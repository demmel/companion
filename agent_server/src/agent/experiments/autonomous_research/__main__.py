"""
Entry point for running autonomous research experiments as a module.

Usage:
    python -m agent.experiments.autonomous_research research "Byzantine Empire"
"""

from .cli import main

if __name__ == "__main__":
    main()
