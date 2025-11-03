"""
Unified LLM router that dispatches to provider implementations
"""

import logging
from typing import Iterator, List, Optional
from agent.llm.interface import Message, CallStats, ImagesInput
from agent.llm.models import SupportedModel, is_ollama_model, is_anthropic_model
from agent.llm.ollama import OllamaLLM, create_ollama_llm
from agent.llm.anthropic import AnthropicLLM, create_anthropic_llm
from agent.config import config

logger = logging.getLogger(__name__)


class LLM:
    """
    Unified LLM router that dispatches calls to appropriate provider.

    This class routes method calls to OllamaLLM or AnthropicLLM based on
    the model parameter. It provides a single interface for all LLM operations
    regardless of provider.
    """

    def __init__(self, ollama: OllamaLLM, anthropic: AnthropicLLM):
        self.ollama = ollama
        self.anthropic = anthropic
        # Create unified models dict for backward compatibility
        self._models = {**ollama.models, **anthropic.models}

    @property
    def models(self):
        """Unified models dict from all providers"""
        return self._models

    def _get_provider(self, model: SupportedModel):
        """Get the appropriate provider for a model"""
        if is_ollama_model(model):
            return self.ollama
        elif is_anthropic_model(model):
            return self.anthropic
        else:
            raise ValueError(
                f"Unknown model: {model}. Not configured for any provider."
            )

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
        """Send chat request, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.chat(
            model=model,
            messages=messages,
            caller=caller,
            stream=stream,
            keep_alive=keep_alive,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            num_predict=num_predict,
            **kwargs,
        )

    def chat_streaming(
        self, model: SupportedModel, messages: List[Message], caller: str, **kwargs
    ) -> Iterator:
        """Streaming chat request, routing to appropriate provider"""
        provider = self._get_provider(model)
        yield from provider.chat_streaming(model, messages, caller, **kwargs)

    def chat_complete(
        self, model: SupportedModel, messages: List[Message], caller: str, **kwargs
    ) -> Optional[str]:
        """Non-streaming chat request, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.chat_complete(model, messages, caller, **kwargs)

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
        images: ImagesInput = None,
        **kwargs,
    ):
        """Send generation request, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.generate(
            model=model,
            prompt=prompt,
            caller=caller,
            stream=stream,
            keep_alive=keep_alive,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            num_predict=num_predict,
            images=images,
            **kwargs,
        )

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        images: ImagesInput = None,
        **kwargs,
    ) -> Iterator:
        """Streaming generation request, routing to appropriate provider"""
        provider = self._get_provider(model)
        yield from provider.generate_streaming(model, prompt, caller, images, **kwargs)

    def generate_complete(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        num_predict: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        images: ImagesInput = None,
        **kwargs,
    ) -> str:
        """Non-streaming generation request, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.generate_complete(
            model, prompt, caller, num_predict, repeat_penalty, images, **kwargs
        )

    def is_model_available(self, model: SupportedModel) -> bool:
        """Check if a model is available, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.is_model_available(model)

    def pull_model(self, model: SupportedModel) -> bool:
        """Pull a model, routing to appropriate provider"""
        provider = self._get_provider(model)
        return provider.pull_model(model)

    def add_call_stats(self, caller: str, duration: float) -> None:
        """Add call statistics (delegates to both providers)"""
        # Note: This is called by the provider's convenience methods,
        # so stats are already tracked on the provider level
        pass

    def reset_call_stats(self) -> None:
        """Reset all LLM call statistics across all providers"""
        self.ollama.reset_call_stats()
        self.anthropic.reset_call_stats()

    def log_stats_summary(self) -> None:
        """Log aggregated statistics from all providers"""
        # Aggregate stats from both providers
        all_stats = {}

        # Collect from Ollama
        for caller, stats in self.ollama.call_stats.items():
            if caller not in all_stats:
                all_stats[caller] = CallStats()
            all_stats[caller].count += stats.count
            all_stats[caller].total_time += stats.total_time
            all_stats[caller].times.extend(stats.times)

        # Collect from Anthropic
        for caller, stats in self.anthropic.call_stats.items():
            if caller not in all_stats:
                all_stats[caller] = CallStats()
            all_stats[caller].count += stats.count
            all_stats[caller].total_time += stats.total_time
            all_stats[caller].times.extend(stats.times)

        # Log aggregated stats
        if not all_stats:
            return

        total_calls = sum(stats.count for stats in all_stats.values())
        total_time = sum(stats.total_time for stats in all_stats.values())

        logger.info(
            f"LLM call statistics: {total_calls} calls, {total_time:.2f}s total"
        )

        # Sort by total time (most expensive first)
        sorted_stats = sorted(
            all_stats.items(), key=lambda x: x[1].total_time, reverse=True
        )

        for caller, stats in sorted_stats:
            avg_time = stats.total_time / stats.count if stats.count > 0 else 0
            logger.info(
                f"  {caller}: {stats.count} calls, {stats.total_time:.2f}s, {avg_time:.2f}s avg"
            )


def create_llm(
    ollama_host: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> LLM:
    """
    Create a unified LLM router with both Ollama and Anthropic providers.

    Args:
        ollama_host: Host for Ollama server (if None, uses config/env var)
        anthropic_api_key: API key for Anthropic (if None, uses config/env var)

    Returns:
        LLM router instance that can handle models from both providers
    """
    # Use config values if not explicitly provided
    ollama_host = ollama_host or config.ollama_host()
    anthropic_api_key = anthropic_api_key or config.anthropic_api_key()

    ollama_llm = create_ollama_llm(host=ollama_host)
    anthropic_llm = create_anthropic_llm(api_key=anthropic_api_key)
    return LLM(ollama=ollama_llm, anthropic=anthropic_llm)
