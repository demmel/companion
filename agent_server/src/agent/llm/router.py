"""
Unified LLM router that dispatches to provider implementations
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional
from agent.llm.interface import Message, ImagesInput
from agent.llm.models import SupportedModel, is_ollama_model, is_anthropic_model, OllamaModelConfig, AnthropicModelConfig
from agent.llm.ollama import SerializedOllamaLLM, create_ollama_llm
from agent.llm.anthropic import AnthropicLLM, create_anthropic_llm
from agent.config import config

logger = logging.getLogger(__name__)


@dataclass
class CallStats:
    """Statistics for LLM calls"""
    count: int = 0
    total_time: float = 0.0
    times: List[float] = field(default_factory=list)


class LLM:
    """Unified LLM router that dispatches to appropriate provider."""

    def __init__(self, ollama: SerializedOllamaLLM, anthropic: AnthropicLLM) -> None:
        self._ollama = ollama
        self._anthropic = anthropic
        self._call_stats: Dict[str, CallStats] = {}

    def _get_provider(self, model: SupportedModel) -> SerializedOllamaLLM | AnthropicLLM:
        if is_ollama_model(model):
            return self._ollama
        elif is_anthropic_model(model):
            return self._anthropic
        raise ValueError(f"Unknown model: {model}")

    def _track_call(self, caller: str, duration: float) -> None:
        if caller not in self._call_stats:
            self._call_stats[caller] = CallStats()
        stats = self._call_stats[caller]
        stats.count += 1
        stats.total_time += duration
        stats.times.append(duration)

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        caller: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        start = time.time()
        result = self._get_provider(model).chat(model, messages, temperature, num_predict)
        self._track_call(caller, time.time() - start)
        logger.info(f"LLM chat [{caller}]: {time.time() - start:.2f}s")
        return result

    def chat_streaming(
        self,
        model: SupportedModel,
        messages: List[Message],
        caller: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Iterator[str]:
        start = time.time()
        yield from self._get_provider(model).chat_streaming(model, messages, temperature, num_predict)
        self._track_call(caller, time.time() - start)
        logger.info(f"LLM chat_streaming [{caller}]: {time.time() - start:.2f}s")

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> str:
        start = time.time()
        result = self._get_provider(model).generate(model, prompt, temperature, num_predict, images)
        self._track_call(caller, time.time() - start)
        logger.info(f"LLM generate [{caller}]: {time.time() - start:.2f}s")
        return result

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> Iterator[str]:
        start = time.time()
        yield from self._get_provider(model).generate_streaming(model, prompt, temperature, num_predict, images)
        self._track_call(caller, time.time() - start)
        logger.info(f"LLM generate_streaming [{caller}]: {time.time() - start:.2f}s")

    def models(self) -> Dict[SupportedModel, OllamaModelConfig | AnthropicModelConfig]:
        """Get all model configurations."""
        result: Dict[SupportedModel, OllamaModelConfig | AnthropicModelConfig] = {}
        result.update(self._ollama.models())
        result.update(self._anthropic.models)
        return result

    def estimate_tokens(self, text: str, model: SupportedModel) -> int:
        """Estimate the number of tokens in text for a given model."""
        model_config = self.models().get(model)
        if model_config is None:
            # Default estimate if model not found
            return int(len(text) / 3.4)
        return int(len(text) / model_config.estimated_token_size)

    def is_model_available(self, model: SupportedModel) -> bool:
        return self._get_provider(model).is_model_available(model)

    def pull_model(self, model: SupportedModel) -> bool:
        return self._get_provider(model).pull_model(model)

    def close(self) -> None:
        self._ollama.stop()

    def log_stats_summary(self) -> None:
        if not self._call_stats:
            return
        total_calls = sum(s.count for s in self._call_stats.values())
        total_time = sum(s.total_time for s in self._call_stats.values())
        logger.info(f"LLM stats: {total_calls} calls, {total_time:.2f}s total")
        for caller, stats in sorted(self._call_stats.items(), key=lambda x: x[1].total_time, reverse=True):
            avg = stats.total_time / stats.count if stats.count else 0
            logger.info(f"  {caller}: {stats.count} calls, {stats.total_time:.2f}s, {avg:.2f}s avg")


def create_llm(
    ollama_host: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> LLM:
    """Create a unified LLM router."""
    return LLM(
        ollama=create_ollama_llm(host=ollama_host or config.ollama_host()),
        anthropic=create_anthropic_llm(api_key=anthropic_api_key or config.anthropic_api_key()),
    )
