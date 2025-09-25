#!/usr/bin/env python3
"""
Script to run the DAG Memory Evolution Visualizer.

Usage:
    python run_visualizer.py [--host HOST] [--port PORT] [--debug]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from agent.experiments.temporal_context_dag.visualizer.web_app import run_visualizer


def main():
    parser = argparse.ArgumentParser(description='Run DAG Memory Evolution Visualizer')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print("DAG Memory Evolution Visualizer")
    print("=" * 40)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print("\nOnce the server is running, open your web browser and navigate to:")
    print(f"http://{args.host}:{args.port}")
    print("\nTo stop the server, press Ctrl+C")
    print("=" * 40)

    try:
        run_visualizer(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()