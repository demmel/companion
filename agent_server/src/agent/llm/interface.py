"""
LLM interface for different providers
"""

import time
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, List, Dict, Optional, Union, Sequence, Any
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type alias for image data
ImageInput = Union[str, bytes, Path]
ImagesInput = Optional[Sequence[ImageInput]]


@dataclass
class CallStats:
    """Statistics for LLM calls by caller"""

    count: int = 0
    total_time: float = 0.0
    times: List[float] = field(default_factory=list)


class Message(BaseModel):
    role: str
    content: str


class ILLM(ABC):
    """Abstract interface for LLM providers"""

    def __init__(self):
        self.call_stats: Dict[str, CallStats] = defaultdict(CallStats)

    @abstractmethod
    def chat(
        self,
        model: Any,  # Provider-specific model type (SupportedModel enum)
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
        pass

    def chat_streaming(
        self, model: Any, messages: List[Message], caller: str, **kwargs
    ) -> Iterator[Any]:
        """Convenience method for streaming chat"""
        start_time = time.time()

        yield from self.chat(model, messages, stream=True, caller=caller, **kwargs)  # type: ignore

        end_time = time.time()
        duration = end_time - start_time
        self.add_call_stats(caller, duration)

    def chat_complete(
        self, model: Any, messages: List[Message], caller: str, **kwargs
    ) -> Optional[str]:
        """Convenience method for non-streaming chat"""
        start_time = time.time()
        response = self.chat(model, messages, stream=False, caller=caller, **kwargs)  # type: ignore

        # Track the call
        duration = time.time() - start_time
        self.add_call_stats(caller, duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")

        return self._extract_content_from_response(response)

    @abstractmethod
    def _extract_content_from_response(self, response: Any) -> Optional[str]:
        """Extract text content from provider-specific response format"""
        pass

    @abstractmethod
    def is_model_available(self, model: Any) -> bool:
        """Check if a model is available"""
        pass

    @abstractmethod
    def pull_model(self, model: Any) -> bool:
        """Pull a model if not available (may not be applicable for all providers)"""
        pass

    @abstractmethod
    def generate(
        self,
        model: Any,
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
        """Send direct generation request (no chat template) to LLM"""
        pass

    def generate_streaming(
        self,
        model: Any,
        prompt: str,
        caller: str,
        images: ImagesInput = None,
        **kwargs,
    ) -> Iterator[Any]:
        """Convenience method for streaming direct generation"""
        start_time = time.time()

        yield from self.generate(model, prompt, caller, stream=True, images=images, **kwargs)  # type: ignore

        end_time = time.time()
        duration = end_time - start_time
        self.add_call_stats(caller, duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")

    def generate_complete(
        self,
        model: Any,
        prompt: str,
        caller: str,
        num_predict: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        images: ImagesInput = None,
        **kwargs,
    ) -> str:
        """Convenience method for non-streaming direct generation"""
        start_time = time.time()
        response = self.generate(
            model,
            prompt,
            caller,
            stream=False,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
            images=images,
            **kwargs,
        )

        # Track the call
        duration = time.time() - start_time
        self.add_call_stats(caller, duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")

        return self._extract_content_from_generate_response(response)

    @abstractmethod
    def _extract_content_from_generate_response(self, response: Any) -> str:
        """Extract text content from provider-specific generate response format"""
        pass

    def add_call_stats(self, caller: str, duration: float) -> None:
        """Add a new call to the statistics"""
        if caller not in self.call_stats:
            self.call_stats[caller] = CallStats()
        stats = self.call_stats[caller]
        stats.count += 1
        stats.total_time += duration
        stats.times.append(duration)

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
