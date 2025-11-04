"""
Model definitions, configurations, and provider detection
"""

from enum import Enum
from dataclasses import dataclass


class SupportedModel(str, Enum):
    """Supported models across all providers"""

    # Ollama models
    MISTRAL_SMALL = "huihui_ai/mistral-small-abliterated"
    MISTRAL_SMALL_3_2 = "mistral-small3.2:latest"
    MISTRAL_SMALL_3_2_Q4 = (
        "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL"
    )
    MISTRAL_SMALL_3_2_Q8 = (
        "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q8_K_XL"
    )
    MISTRAL_NEMO = "mistral-nemo:latest"
    DOLPHIN_MISTRAL_NEMO = "CognitiveComputations/dolphin-mistral-nemo:latest"
    LLAMA_8B = "llama3.1:8b"
    GEMMA_27B = "aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m"
    DEEPSEEK_R1_14B = "huihui_ai/deepseek-r1-abliterated:14b"
    RP_MAX = "technobyte/arliai-rpmax-12b-v1.1:q4_k_m"

    # Anthropic models (latest)
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250805"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"


@dataclass
class OllamaModelConfig:
    """Configuration for an Ollama model"""

    model: SupportedModel
    keep_alive: str = "30m"
    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_top_k: int = 50
    default_repeat_penalty: float = 1.1
    default_num_predict: int = 4096
    context_window: int = 32768
    estimated_token_size: float = 3.4


@dataclass
class AnthropicModelConfig:
    """Configuration for an Anthropic model"""

    model: SupportedModel
    max_tokens: int = 4096
    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_top_k: int = 50
    context_window: int = 200000  # Anthropic models have large context windows
    estimated_token_size: float = 3.4  # Similar to other models


@dataclass
class ModelConfig:
    """Configuration for which model to use for each action type"""

    # Planning and initialization
    state_initialization_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    action_planning_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    situational_analysis_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5

    # Memory operations
    memory_retrieval_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    memory_formation_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    trigger_compression_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5

    # Generation actions
    think_action_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    speak_action_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5

    # Utility actions
    visual_action_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    fetch_url_action_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5
    evaluate_priorities_action_model: SupportedModel = SupportedModel.CLAUDE_SONNET_4_5


# Provider detection
_OLLAMA_MODELS = {
    SupportedModel.MISTRAL_SMALL,
    SupportedModel.MISTRAL_SMALL_3_2,
    SupportedModel.MISTRAL_SMALL_3_2_Q4,
    SupportedModel.MISTRAL_SMALL_3_2_Q8,
    SupportedModel.MISTRAL_NEMO,
    SupportedModel.DOLPHIN_MISTRAL_NEMO,
    SupportedModel.LLAMA_8B,
    SupportedModel.GEMMA_27B,
    SupportedModel.DEEPSEEK_R1_14B,
    SupportedModel.RP_MAX,
}

_ANTHROPIC_MODELS = {
    SupportedModel.CLAUDE_SONNET_4_5,
    SupportedModel.CLAUDE_OPUS_4_1,
    SupportedModel.CLAUDE_HAIKU_4_5,
}


def is_ollama_model(model: SupportedModel) -> bool:
    """Check if a model is an Ollama model"""
    return model in _OLLAMA_MODELS


def is_anthropic_model(model: SupportedModel) -> bool:
    """Check if a model is an Anthropic model"""
    return model in _ANTHROPIC_MODELS
