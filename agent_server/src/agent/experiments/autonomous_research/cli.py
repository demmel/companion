#!/usr/bin/env python3
"""
CLI for autonomous research experiments.

Usage:
    python -m agent.experiments.autonomous_research.cli research "Byzantine Empire" "Coffee"
    python -m agent.experiments.autonomous_research.cli compare-integration "Quantum Computing"
    python -m agent.experiments.autonomous_research.cli compare-retrieval "Machine Learning"
"""

import argparse
import logging
import sys
from pathlib import Path

from .experiment_runner import (
    run_simple_experiment,
    compare_integration_strategies,
    compare_retrieval_strategies,
    run_matrix_test,
    ExperimentConfig,
    ExperimentRunner,
    INTEGRATION_STRATEGIES,
    RETRIEVAL_STRATEGIES,
)


def setup_logging(verbose: bool):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO

    # Simple format without module names
    log_format = (
        "%(message)s"
        if not verbose
        else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt="%H:%M:%S",
        force=True,  # Override any existing config
    )


def cmd_research(args):
    """Run research on topics"""
    print(f"\n🔬 Researching: {', '.join(args.topics)}")
    print(f"   Depth: {args.depth} cycles")
    print(f"   Integration: {args.integration}")
    print(f"   Retrieval: {args.retrieval}")
    print()

    from .experiment_runner import STANDARD_RESEARCH_CONFIG, STANDARD_EXTRACTION_CONFIG

    config = ExperimentConfig(
        topics=args.topics,
        research_depth=args.depth,
        integration_strategy=args.integration,
        retrieval_strategy=args.retrieval,
        research_config=STANDARD_RESEARCH_CONFIG,
        extraction_config=STANDARD_EXTRACTION_CONFIG,
        output_dir=Path(args.output) if args.output else None,
    )

    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()

    print("\n✅ Experiment complete!")
    print(f"   Results saved to: {config.output_dir}")
    return results


def cmd_compare_integration(args):
    """Compare integration strategies"""
    print(f"\n🔬 Comparing integration strategies for: {', '.join(args.topics)}")
    print(f"   Depth: {args.depth} cycles")
    print(f"   Strategies: {', '.join(INTEGRATION_STRATEGIES.keys())}")
    print()

    results = compare_integration_strategies(args.topics, args.depth)

    print("\n✅ Comparison complete!")
    return results


def cmd_compare_retrieval(args):
    """Compare retrieval strategies"""
    print(f"\n🔬 Comparing retrieval strategies for: {', '.join(args.topics)}")
    print(f"   Depth: {args.depth} cycles")
    print(f"   Strategies: {', '.join(RETRIEVAL_STRATEGIES.keys())}")
    print()

    results = compare_retrieval_strategies(args.topics, args.depth)

    print("\n✅ Comparison complete!")
    return results


def cmd_matrix_test(args):
    """Run matrix test of all strategy combinations"""
    print(f"\n🔬 Matrix test: {', '.join(args.topics)}")
    print(f"   Depth: {args.depth} cycles")
    print(
        f"   Testing: {len(INTEGRATION_STRATEGIES)} integration × {len(RETRIEVAL_STRATEGIES)} retrieval"
    )
    print(
        f"   Total: {len(INTEGRATION_STRATEGIES) * len(RETRIEVAL_STRATEGIES)} combinations"
    )
    print()

    results = run_matrix_test(args.topics, args.depth)

    print("\n✅ Matrix test complete!")
    print(f"   Tested {len(results)} combinations")
    return results


def cmd_list_strategies(args):
    """List available strategies"""
    print("\n📋 Available Integration Strategies:")
    for name in INTEGRATION_STRATEGIES.keys():
        print(f"   • {name}")

    print("\n📋 Available Retrieval Strategies:")
    for name in RETRIEVAL_STRATEGIES.keys():
        print(f"   • {name}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Research Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research a single topic
  %(prog)s research "Byzantine Empire" --depth 3

  # Research multiple topics
  %(prog)s research "Quantum Computing" "Machine Learning" --depth 2

  # Compare integration strategies
  %(prog)s compare-integration "Coffee brewing" --depth 2

  # Compare retrieval strategies
  %(prog)s compare-retrieval "Byzantine Empire" --depth 2

  # Test all integration × retrieval combinations
  %(prog)s matrix "Byzantine Empire" --depth 2

  # List available strategies
  %(prog)s list-strategies
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Research command
    research_parser = subparsers.add_parser(
        "research", help="Research topics and build knowledge graph"
    )
    research_parser.add_argument("topics", nargs="+", help="Topics to research")
    research_parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=2,
        help="Research depth (number of cycles per topic, default: 2)",
    )
    research_parser.add_argument(
        "-i",
        "--integration",
        choices=list(INTEGRATION_STRATEGIES.keys()),
        default="bridged",
        help="Integration strategy (default: bridged)",
    )
    research_parser.add_argument(
        "-r",
        "--retrieval",
        choices=list(RETRIEVAL_STRATEGIES.keys()),
        default="hybrid",
        help="Retrieval strategy (default: hybrid)",
    )
    research_parser.add_argument(
        "-o", "--output", help="Output directory for results (default: ./results)"
    )

    # Compare integration command
    compare_int_parser = subparsers.add_parser(
        "compare-integration", help="Compare all integration strategies"
    )
    compare_int_parser.add_argument("topics", nargs="+", help="Topics to research")
    compare_int_parser.add_argument(
        "-d", "--depth", type=int, default=2, help="Research depth (default: 2)"
    )

    # Compare retrieval command
    compare_ret_parser = subparsers.add_parser(
        "compare-retrieval", help="Compare all retrieval strategies"
    )
    compare_ret_parser.add_argument("topics", nargs="+", help="Topics to research")
    compare_ret_parser.add_argument(
        "-d", "--depth", type=int, default=2, help="Research depth (default: 2)"
    )

    # Matrix test command
    matrix_parser = subparsers.add_parser(
        "matrix", help="Test all integration × retrieval combinations"
    )
    matrix_parser.add_argument("topics", nargs="+", help="Topics to research")
    matrix_parser.add_argument(
        "-d", "--depth", type=int, default=2, help="Research depth (default: 2)"
    )

    # List strategies command
    list_parser = subparsers.add_parser(
        "list-strategies", help="List available strategies"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    try:
        if args.command == "research":
            cmd_research(args)
        elif args.command == "compare-integration":
            cmd_compare_integration(args)
        elif args.command == "compare-retrieval":
            cmd_compare_retrieval(args)
        elif args.command == "matrix":
            cmd_matrix_test(args)
        elif args.command == "list-strategies":
            cmd_list_strategies(args)
        else:
            parser.print_help()
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
