"""
LLM interface for different providers
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union, Sequence
from pathlib import Path
from pydantic import BaseModel

from agent.llm.models import SupportedModel


# Provider-agnostic LLM exceptions
class LLMError(Exception):
    """Base exception for all LLM errors"""
    pass


class LLMAuthenticationError(LLMError):
    """Authentication failed - invalid API key or credentials"""

    def __init__(self, message: str = "Authentication failed: Check your API key"):
        super().__init__(message)


class LLMInsufficientCreditsError(LLMError):
    """Insufficient credits or quota exceeded"""

    def __init__(
        self, message: str = "Insufficient credits: Add credits to your account"
    ):
        super().__init__(message)


class LLMRateLimitError(LLMError):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded: Wait and retry"):
        super().__init__(message)


class LLMAPIError(LLMError):
    """Generic API error"""
    pass


# Type alias for image data
ImageInput = Union[str, bytes, Path]
ImagesInput = Optional[Sequence[ImageInput]]


class Message(BaseModel):
    role: str
    content: str


class ILLM(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Send chat request, return text response."""
        pass

    @abstractmethod
    def chat_streaming(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Iterator[str]:
        """Send chat request, yield text chunks."""
        pass

    @abstractmethod
    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> str:
        """Send generation request, return text response."""
        pass

    @abstractmethod
    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> Iterator[str]:
        """Send generation request, yield text chunks."""
        pass

    @abstractmethod
    def is_model_available(self, model: SupportedModel) -> bool:
        """Check if a model is available"""
        pass

    @abstractmethod
    def pull_model(self, model: SupportedModel) -> bool:
        """Pull a model if not available"""
        pass
