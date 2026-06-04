"""
Model definitions, configurations, and provider detection.
"""

from dataclasses import dataclass
from typing import ClassVar
from pydantic_core import core_schema


class SupportedModel(str):
    """String-backed model identifier with enum-like compatibility helpers."""

    # Class-variable type annotations so type checkers recognise SupportedModel.XXX
    # Ollama model constants
    MISTRAL_SMALL: ClassVar["SupportedModel"]
    MISTRAL_SMALL_3_2: ClassVar["SupportedModel"]
    MISTRAL_SMALL_3_2_Q4: ClassVar["SupportedModel"]
    MISTRAL_SMALL_3_2_Q8: ClassVar["SupportedModel"]
    MISTRAL_NEMO: ClassVar["SupportedModel"]
    DOLPHIN_MISTRAL_NEMO: ClassVar["SupportedModel"]
    LLAMA_8B: ClassVar["SupportedModel"]
    LLAMA_3B: ClassVar["SupportedModel"]
    GEMMA_27B: ClassVar["SupportedModel"]
    DEEPSEEK_R1_14B: ClassVar["SupportedModel"]
    RP_MAX: ClassVar["SupportedModel"]
    QWEN3_VL_30B: ClassVar["SupportedModel"]
    QWEN3_VL_30B_THINKING: ClassVar["SupportedModel"]
    CYDONIA_24B_VISION: ClassVar["SupportedModel"]
    # Anthropic model constants
    CLAUDE_SONNET_4_5: ClassVar["SupportedModel"]
    CLAUDE_OPUS_4_1: ClassVar["SupportedModel"]
    CLAUDE_HAIKU_4_5: ClassVar["SupportedModel"]

    @property
    def value(self) -> str:
        return str(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: object,
        _handler: object,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


# Ollama model suggestions
SupportedModel.MISTRAL_SMALL = SupportedModel("huihui_ai/mistral-small-abliterated")
SupportedModel.MISTRAL_SMALL_3_2 = SupportedModel("mistral-small3.2:latest")
SupportedModel.MISTRAL_SMALL_3_2_Q4 = SupportedModel(
    "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL"
)
SupportedModel.MISTRAL_SMALL_3_2_Q8 = SupportedModel(
    "hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q8_K_XL"
)
SupportedModel.MISTRAL_NEMO = SupportedModel("mistral-nemo:latest")
SupportedModel.DOLPHIN_MISTRAL_NEMO = SupportedModel(
    "CognitiveComputations/dolphin-mistral-nemo:latest"
)
SupportedModel.LLAMA_8B = SupportedModel("llama3.1:8b")
SupportedModel.LLAMA_3B = SupportedModel("llama3.2:3b")
SupportedModel.GEMMA_27B = SupportedModel("aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m")
SupportedModel.DEEPSEEK_R1_14B = SupportedModel("huihui_ai/deepseek-r1-abliterated:14b")
SupportedModel.RP_MAX = SupportedModel("technobyte/arliai-rpmax-12b-v1.1:q4_k_m")
# Bake-off candidates (vision-capable, uncensored, fit a 24GB GPU at Q4)
SupportedModel.QWEN3_VL_30B = SupportedModel(
    "huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct"
)
SupportedModel.QWEN3_VL_30B_THINKING = SupportedModel(
    "huihui_ai/qwen3-vl-abliterated:30b-a3b-Thinking"
)
SupportedModel.CYDONIA_24B_VISION = SupportedModel(
    "Fermi/Cydonia-24B-v4.3-heretic-vision:Q4_K_M"
)

# Anthropic models
SupportedModel.CLAUDE_SONNET_4_5 = SupportedModel("claude-sonnet-4-5-20250929")
SupportedModel.CLAUDE_OPUS_4_1 = SupportedModel("claude-opus-4-1-20250805")
SupportedModel.CLAUDE_HAIKU_4_5 = SupportedModel("claude-haiku-4-5-20251001")

KNOWN_OLLAMA_MODELS: tuple[SupportedModel, ...] = (
    SupportedModel.MISTRAL_SMALL,
    SupportedModel.MISTRAL_SMALL_3_2,
    SupportedModel.MISTRAL_SMALL_3_2_Q4,
    SupportedModel.MISTRAL_SMALL_3_2_Q8,
    SupportedModel.MISTRAL_NEMO,
    SupportedModel.DOLPHIN_MISTRAL_NEMO,
    SupportedModel.LLAMA_8B,
    SupportedModel.LLAMA_3B,
    SupportedModel.GEMMA_27B,
    SupportedModel.DEEPSEEK_R1_14B,
    SupportedModel.RP_MAX,
    SupportedModel.QWEN3_VL_30B,
    SupportedModel.QWEN3_VL_30B_THINKING,
    SupportedModel.CYDONIA_24B_VISION,
)

KNOWN_ANTHROPIC_MODELS: tuple[SupportedModel, ...] = (
    SupportedModel.CLAUDE_SONNET_4_5,
    SupportedModel.CLAUDE_OPUS_4_1,
    SupportedModel.CLAUDE_HAIKU_4_5,
)

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

    # GPU residency. ollama's num_gpu is the number of layers to offload to the
    # GPU; -1 means "auto", which lets ollama silently offload layers to CPU when
    # its (conservative) estimator thinks the model + KV cache won't fit, tanking
    # throughput. A large value forces every layer onto the GPU. Lower this only
    # for a model that genuinely does not fit in VRAM so the offload is a
    # deliberate choice rather than a silent surprise. Pair with a quantized KV
    # cache (OLLAMA_KV_CACHE_TYPE=q8_0) + OLLAMA_FLASH_ATTENTION=1 on the server
    # to keep the full context window resident on a 24GB GPU.
    num_gpu: int = 999
    num_thread: int = 32

    # Anti-repetition samplers. Defaults preserve prior behavior: penalties off,
    # repeat_last_n at ollama's default, mirostat disabled. ollama does not expose
    # DRY/XTC, so frequency/presence penalties + mirostat are the available levers
    # for the companion's repetitive-output problem.
    default_repeat_last_n: int = 64
    default_frequency_penalty: float = 0.0
    default_presence_penalty: float = 0.0
    default_mirostat: int = 0
    default_mirostat_tau: float = 5.0
    default_mirostat_eta: float = 0.1


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

    # TTS
    tts_rewrite_model: SupportedModel = SupportedModel.MISTRAL_SMALL_3_2_Q4


def is_ollama_model(model: SupportedModel) -> bool:
    """Treat every non-Anthropic model as an Ollama model."""
    return not is_anthropic_model(model)


def is_anthropic_model(model: SupportedModel) -> bool:
    """Check if a model is a known Anthropic model."""
    return model in KNOWN_ANTHROPIC_MODELS
