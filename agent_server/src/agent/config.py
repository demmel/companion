"""
Configuration management for the agent system.

Loads environment variables from .env file and provides a centralized
configuration interface.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Find and load .env file from the project root
# This works whether we're in src/agent or running from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"

# Load .env file if it exists
if env_file.exists():
    load_dotenv(env_file)


class Config:
    """Centralized configuration for the agent system"""

    # ===== LLM Provider Configuration =====

    @staticmethod
    def ollama_host() -> str:
        """Ollama server host"""
        return os.getenv("OLLAMA_HOST", "localhost:11434")

    @staticmethod
    def anthropic_api_key() -> Optional[str]:
        """Anthropic API key"""
        return os.getenv("ANTHROPIC_API_KEY")

    # ===== Logging =====

    @staticmethod
    def log_level() -> str:
        """Logging level"""
        return os.getenv("LOG_LEVEL", "INFO")

    # ===== Validation =====

    @staticmethod
    def validate_anthropic_config() -> bool:
        """Check if Anthropic is properly configured"""
        api_key = Config.anthropic_api_key()
        if not api_key or api_key == "your_anthropic_api_key_here":
            return False
        return True

    @staticmethod
    def get_missing_config() -> list[str]:
        """Get list of missing required configuration"""
        missing = []

        # Anthropic is optional, but if you want to use it:
        if not Config.validate_anthropic_config():
            missing.append(
                "ANTHROPIC_API_KEY (optional - only needed for Claude models)"
            )

        return missing


# Singleton instance
config = Config()
