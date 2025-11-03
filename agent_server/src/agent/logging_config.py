"""
Logging configuration for the agent system.

Sets up centralized logging with configurable log level from environment variables.
"""

import logging
import sys
from agent.config import config


def setup_logging():
    """
    Configure logging for the entire application.

    This should be called once at application startup, before any other
    modules that use logging are imported.
    """
    log_level = config.log_level()

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)-60s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,  # Override any existing configuration
    )

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")
