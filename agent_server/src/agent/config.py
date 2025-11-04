"""
Configuration management for the agent system.

Loads environment variables from .env file and provides a centralized
configuration interface.
"""

import os
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from agent.llm.models import ModelConfig

# Find and load .env file from the project root
# This works whether we're in src/agent or running from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"

# Load .env file if it exists
if env_file.exists():
    load_dotenv(env_file)


class Config:
    """Centralized configuration for the agent system"""

    # Model configuration cache
    _model_config: Optional[ModelConfig] = None
    _model_config_path: Optional[Path] = None

    # ===== Model Configuration =====

    @staticmethod
    def get_model_config_path() -> Path:
        """Get the path to the model config file"""
        if Config._model_config_path is None:
            # Store in project root for now (could be moved to user data dir)
            Config._model_config_path = project_root / "model_config.json"
        return Config._model_config_path

    @staticmethod
    def get_model_config() -> ModelConfig:
        """Get the current model configuration"""
        if Config._model_config is None:
            config_path = Config.get_model_config_path()
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        data = json.load(f)
                    # Convert string values back to SupportedModel enums
                    from agent.llm.models import SupportedModel

                    model_kwargs = {
                        key: SupportedModel(value) for key, value in data.items()
                    }
                    Config._model_config = ModelConfig(**model_kwargs)
                except Exception as e:
                    import logging

                    logging.warning(
                        f"Failed to load model config from {config_path}: {e}. Using defaults."
                    )
                    Config._model_config = ModelConfig()
            else:
                # Use default config
                Config._model_config = ModelConfig()
        return Config._model_config

    @staticmethod
    def set_model_config(model_config: ModelConfig) -> None:
        """Set and persist the model configuration"""
        Config._model_config = model_config
        config_path = Config.get_model_config_path()

        # Convert to dict with string values for JSON serialization
        data = {key: value.value for key, value in model_config.__dict__.items()}

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

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
