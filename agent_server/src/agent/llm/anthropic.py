"""
Anthropic LLM implementation
"""

import anthropic
import logging
import base64
import mimetypes
import time
from typing import Iterator, List, Dict, Optional, Union, Callable, TypeVar
from pathlib import Path
from agent.llm.interface import (
    ILLM,
    Message,
    ImagesInput,
    ImageInput,
    LLMAuthenticationError,
    LLMInsufficientCreditsError,
    LLMRateLimitError,
    LLMAPIError,
)
from agent.llm.models import SupportedModel, AnthropicModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_rate_limit(
    max_retries: int = 3, base_delay: float = 1.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry on rate limit with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except anthropic.RateLimitError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(f"Rate limit hit, retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
                except anthropic.AuthenticationError as e:
                    raise LLMAuthenticationError(f"Authentication failed: {e}") from e
                except anthropic.PermissionDeniedError as e:
                    raise LLMInsufficientCreditsError(f"Insufficient credits: {e}") from e
                except anthropic.BadRequestError as e:
                    raise LLMAPIError(f"Bad request: {e}") from e
                except anthropic.APIError as e:
                    raise LLMAPIError(f"API error: {e}") from e

            if last_exception:
                raise LLMRateLimitError(f"Rate limit exceeded: {last_exception}") from last_exception
            raise LLMAPIError("Unknown error")

        return wrapper

    return decorator


class AnthropicLLM(ILLM):
    """Anthropic-based LLM implementation"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        models: Dict[SupportedModel, AnthropicModelConfig],
    ) -> None:
        self.client = client
        self.models = models

    def _get_config(self, model: SupportedModel) -> AnthropicModelConfig:
        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available: {list(self.models.keys())}"
            )
        return self.models[model]

    def _encode_image(self, image: ImageInput) -> Dict[str, object]:
        """Encode an image to base64 for Anthropic API."""
        if isinstance(image, (str, Path)):
            file_path = Path(image)
            if not file_path.exists():
                raise FileNotFoundError(f"Image not found: {file_path}")
            with open(file_path, "rb") as f:
                image_data = f.read()
            media_type = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
        elif isinstance(image, bytes):
            image_data = image
            media_type = "image/jpeg"
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(image_data).decode("utf-8"),
            },
        }

    def _build_messages(
        self, messages: List[Message], images: ImagesInput = None
    ) -> tuple[Optional[str], List[Dict[str, object]]]:
        """Build Anthropic message format, extracting system prompt."""
        system_messages = [m for m in messages if m.role == "system"]
        other_messages = [m for m in messages if m.role != "system"]

        system_prompt = (
            "\n\n".join(m.content for m in system_messages)
            if system_messages
            else None
        )

        anthropic_messages: List[Dict[str, object]] = [
            {"role": m.role, "content": m.content} for m in other_messages
        ]

        # Add images to the last user message if present
        if images:
            content_blocks: List[Dict[str, object]] = []
            for img in images:
                try:
                    content_blocks.append(self._encode_image(img))
                except Exception as e:
                    logger.error(f"Failed to encode image: {e}")
            if content_blocks:
                anthropic_messages.append({"role": "user", "content": content_blocks})

        return system_prompt, anthropic_messages

    @retry_on_rate_limit()
    def _call_api(
        self,
        model: SupportedModel,
        system_prompt: Optional[str],
        messages: List[Dict[str, object]],
        temperature: Optional[float],
        num_predict: Optional[int],
    ) -> anthropic.types.Message:
        config = self._get_config(model)
        params: Dict[str, object] = {
            "model": model.value,
            "messages": messages,
            "max_tokens": num_predict or config.max_tokens,
            "temperature": temperature if temperature is not None else config.default_temperature,
        }
        if system_prompt:
            params["system"] = system_prompt
        return self.client.messages.create(**params)  # type: ignore[arg-type]

    @retry_on_rate_limit()
    def _call_api_streaming(
        self,
        model: SupportedModel,
        system_prompt: Optional[str],
        messages: List[Dict[str, object]],
        temperature: Optional[float],
        num_predict: Optional[int],
    ) -> anthropic.MessageStream:
        config = self._get_config(model)
        params: Dict[str, object] = {
            "model": model.value,
            "messages": messages,
            "max_tokens": num_predict or config.max_tokens,
            "temperature": temperature if temperature is not None else config.default_temperature,
        }
        if system_prompt:
            params["system"] = system_prompt
        return self.client.messages.stream(**params)  # type: ignore[arg-type]

    def _extract_text(self, response: anthropic.types.Message) -> str:
        if response.content:
            for block in response.content:
                if block.type == "text":
                    return block.text
        return ""

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        system_prompt, api_messages = self._build_messages(messages)
        response = self._call_api(model, system_prompt, api_messages, temperature, num_predict)
        return self._extract_text(response)

    def chat_streaming(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Iterator[str]:
        system_prompt, api_messages = self._build_messages(messages)
        with self._call_api_streaming(model, system_prompt, api_messages, temperature, num_predict) as stream:
            for text in stream.text_stream:
                yield text

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> str:
        # Simulate generate using chat with a system prompt for direct continuation
        system_msg = Message(
            role="system",
            content="Continue directly without meta-commentary.",
        )
        user_msg = Message(role="user", content=prompt)
        system_prompt, api_messages = self._build_messages([system_msg, user_msg], images)
        response = self._call_api(model, system_prompt, api_messages, temperature, num_predict)
        return self._extract_text(response)

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> Iterator[str]:
        system_msg = Message(
            role="system",
            content="Continue directly without meta-commentary.",
        )
        user_msg = Message(role="user", content=prompt)
        system_prompt, api_messages = self._build_messages([system_msg, user_msg], images)
        with self._call_api_streaming(model, system_prompt, api_messages, temperature, num_predict) as stream:
            for text in stream.text_stream:
                yield text

    def is_model_available(self, model: SupportedModel) -> bool:
        return model in self.models

    def pull_model(self, model: SupportedModel) -> bool:
        return True  # No-op for API models


DEFAULT_ANTHROPIC_MODELS = {
    SupportedModel.CLAUDE_SONNET_4_5: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_SONNET_4_5,
    ),
    SupportedModel.CLAUDE_OPUS_4_1: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_OPUS_4_1,
    ),
    SupportedModel.CLAUDE_HAIKU_4_5: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_HAIKU_4_5,
        max_tokens=8192,
    ),
}


def create_anthropic_llm(
    api_key: Optional[str] = None,
    models: Optional[Dict[SupportedModel, AnthropicModelConfig]] = None,
) -> AnthropicLLM:
    """Create an Anthropic LLM instance."""
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return AnthropicLLM(client, models or DEFAULT_ANTHROPIC_MODELS)
