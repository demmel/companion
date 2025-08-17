"""
LLM client for interfacing with Ollama
"""

import ollama
import time
import logging
from collections import defaultdict
from typing import Iterator, List, Dict, Optional
from pydantic import BaseModel
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class CallStats:
    """Statistics for LLM calls by caller"""

    count: int = 0
    total_time: float = 0.0
    times: List[float] = field(default_factory=list)


class Message(BaseModel):
    role: str
    content: str


class SupportedModel(str, Enum):
    """Supported models with backend-specific identifiers"""

    MISTRAL_SMALL = "huihui_ai/mistral-small-abliterated"
    MISTRAL_SMALL_3_2 = "mistral-small3.2:latest"
    MISTRAL_NEMO = "mistral-nemo:latest"
    DOLPHIN_MISTRAL_NEMO = "CognitiveComputations/dolphin-mistral-nemo:latest"
    LLAMA_8B = "llama3.1:8b"
    GEMMA_27B = "aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m"
    DEEPSEEK_R1_14B = "huihui_ai/deepseek-r1-abliterated:14b"
    RP_MAX = "technobyte/arliai-rpmax-12b-v1.1:q4_k_m"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    model: SupportedModel
    keep_alive: str = "30m"
    default_temperature: float = 0.3
    default_top_p: float = 0.9
    default_top_k: int = 50
    default_repeat_penalty: float = 1.1
    default_num_predict: int = 4096
    context_window: int = 32768
    estimated_token_size: float = 3.4


class LLM:
    """Centralized LLM management with model-specific configurations"""

    def __init__(
        self, client: ollama.Client, models: Dict[SupportedModel, ModelConfig]
    ):
        self.client = client
        self.models = models
        self.call_stats: Dict[str, CallStats] = defaultdict(CallStats)

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        caller: str,
        stream: bool = False,
        keep_alive: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_predict: Optional[int] = None,
        **kwargs,
    ):
        """Send chat request with model-specific defaults and overrides"""

        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available models: {list(self.models.keys())}"
            )

        config = self.models[model]

        # Apply model defaults with overrides
        options = {
            "num_gpu": -1,
            "num_thread": 16,
            "num_ctx": config.context_window,
            "temperature": temperature or config.default_temperature,
            "top_p": top_p or config.default_top_p,
            "top_k": top_k or config.default_top_k,
            "repeat_penalty": repeat_penalty or config.default_repeat_penalty,
            "num_predict": num_predict or config.default_num_predict,
            **kwargs,
        }

        # Convert Message objects to dict format for ollama
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        return self.client.chat(
            model=model.value,  # Use the enum value for backend
            messages=message_dicts,
            stream=stream,
            options=options,
            keep_alive=keep_alive or config.keep_alive,
        )

    def chat_streaming(
        self, model: SupportedModel, messages: List[Message], caller: str, **kwargs
    ) -> Iterator[ollama.ChatResponse]:
        """Convenience method for streaming chat"""
        return self.chat(model, messages, stream=True, caller=caller, **kwargs)  # type: ignore

    def chat_complete(
        self, model: SupportedModel, messages: List[Message], caller: str, **kwargs
    ) -> Optional[str]:
        """Convenience method for non-streaming chat"""
        start_time = time.time()
        response: ollama.ChatResponse = self.chat(
            model, messages, stream=False, caller=caller, **kwargs
        )  # type: ignore

        # Track the call
        duration = time.time() - start_time
        stats = self.call_stats[caller]
        stats.count += 1
        stats.total_time += duration
        stats.times.append(duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")

        return response["message"]["content"]

    def is_model_available(self, model: SupportedModel) -> bool:
        """Check if a model is available"""
        try:
            models = self.client.list()
            model_names = [m["name"] for m in models["models"]]
            return model.value in model_names
        except:
            return False

    def pull_model(self, model: SupportedModel) -> bool:
        """Pull a model if not available"""
        try:
            self.client.pull(model.value)
            return True
        except Exception as e:
            print(f"Error pulling model {model}: {e}")
            return False

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        stream: bool = False,
        keep_alive: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        num_predict: Optional[int] = None,
        **kwargs,
    ):
        """Send direct generation request (no chat template) to LLM"""

        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available models: {list(self.models.keys())}"
            )

        config = self.models[model]

        # Apply model defaults with overrides
        options = {
            "num_gpu": -1,
            "num_thread": 16,
            "num_ctx": config.context_window,
            "temperature": temperature or config.default_temperature,
            "top_p": top_p or config.default_top_p,
            "top_k": top_k or config.default_top_k,
            "repeat_penalty": repeat_penalty or config.default_repeat_penalty,
            "num_predict": num_predict or config.default_num_predict,
            **kwargs,
        }

        return self.client.generate(
            model=model.value,  # Use the enum value for backend
            prompt=prompt,
            stream=stream,
            options=options,
            keep_alive=keep_alive or config.keep_alive,
        )

    def generate_streaming(
        self, model: SupportedModel, prompt: str, caller: str, **kwargs
    ) -> Iterator[ollama.GenerateResponse]:
        """Convenience method for streaming direct generation"""
        return self.generate(model, prompt, caller, stream=True, **kwargs)  # type: ignore

    def generate_complete(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        num_predict: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Convenience method for non-streaming direct generation"""
        start_time = time.time()
        response = self.generate(
            model, prompt, caller, stream=False, num_predict=num_predict, **kwargs
        )

        # Track the call
        duration = time.time() - start_time
        stats = self.call_stats[caller]
        stats.count += 1
        stats.total_time += duration
        stats.times.append(duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")

        return response["response"]  # type: ignore

    def reset_call_stats(self) -> None:
        """Reset all LLM call statistics"""
        self.call_stats.clear()

    def log_stats_summary(self) -> None:
        """Log a summary of LLM call statistics"""
        if not self.call_stats:
            return

        total_calls = sum(stats.count for stats in self.call_stats.values())
        total_time = sum(stats.total_time for stats in self.call_stats.values())

        logger.info(
            f"LLM call statistics: {total_calls} calls, {total_time:.2f}s total"
        )

        # Sort by total time (most expensive first)
        sorted_stats = sorted(
            self.call_stats.items(), key=lambda x: x[1].total_time, reverse=True
        )

        for caller, stats in sorted_stats:
            avg_time = stats.total_time / stats.count if stats.count > 0 else 0
            logger.info(
                f"  {caller}: {stats.count} calls, {stats.total_time:.2f}s, {avg_time:.2f}s avg"
            )


# Default model configurations
DEFAULT_MODELS = {
    SupportedModel.LLAMA_8B: ModelConfig(
        model=SupportedModel.LLAMA_8B,
    ),
    SupportedModel.GEMMA_27B: ModelConfig(
        model=SupportedModel.GEMMA_27B,
    ),
    SupportedModel.MISTRAL_SMALL: ModelConfig(
        model=SupportedModel.MISTRAL_SMALL,
        estimated_token_size=3.4,
    ),
    SupportedModel.MISTRAL_SMALL_3_2: ModelConfig(
        model=SupportedModel.MISTRAL_SMALL_3_2,
        estimated_token_size=3.4,
    ),
    SupportedModel.DOLPHIN_MISTRAL_NEMO: ModelConfig(
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO,
    ),
    SupportedModel.MISTRAL_NEMO: ModelConfig(
        model=SupportedModel.MISTRAL_NEMO,
    ),
    SupportedModel.DEEPSEEK_R1_14B: ModelConfig(
        model=SupportedModel.DEEPSEEK_R1_14B,
        default_temperature=0.6,
        default_repeat_penalty=1.2,
        default_top_p=0.95,
    ),
    SupportedModel.RP_MAX: ModelConfig(
        model=SupportedModel.RP_MAX,
    ),
}


def create_llm(
    host: str = "localhost:11434",
    models: Optional[Dict[SupportedModel, ModelConfig]] = None,
) -> LLM:
    """Create an LLM manager with shared client and model configurations"""
    client = ollama.Client(host=host)
    model_configs = models or DEFAULT_MODELS
    return LLM(client, model_configs)
