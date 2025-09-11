#!/usr/bin/env python3
"""
Simple UI output utilities for tqdm-compatible console output.

Provides clean user interface output that works well with progress bars
and doesn't interfere with detailed logging to files.
"""

from tqdm import tqdm


def ui_print(message: str) -> None:
    """Print a message to console in a tqdm-compatible way.

    This function provides clean user interface output that won't interfere
    with tqdm progress bars. If tqdm is available, uses tqdm.write().
    Otherwise falls back to regular print().

    Args:
        message: The message to display to the user
    """
    tqdm.write(message)
