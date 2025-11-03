"""
Anthropic LLM implementation
"""

import anthropic
import logging
import base64
import mimetypes
import time
from typing import Iterator, List, Dict, Optional, Union, Any, Callable, TypeVar
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


def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator to retry Anthropic API calls on rate limit errors with exponential backoff.
    Translates Anthropic-specific exceptions to provider-agnostic LLM exceptions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except anthropic.RateLimitError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Rate limit retry exhausted after {max_retries} attempts"
                        )
                        # Translate to provider-agnostic exception with chaining
                        raise LLMRateLimitError(
                            f"Rate limit exceeded after {max_retries} retries: {e}"
                        ) from e
                except anthropic.AuthenticationError as e:
                    # Authentication failed - translate and fail immediately
                    logger.error(f"Authentication error: {e}")
                    raise LLMAuthenticationError(f"Authentication failed: {e}") from e
                except anthropic.PermissionDeniedError as e:
                    # Permission denied - usually means insufficient credits
                    logger.error(
                        f"Permission denied (likely insufficient credits): {e}"
                    )
                    raise LLMInsufficientCreditsError(
                        f"Insufficient credits or quota exceeded: {e}"
                    ) from e
                except anthropic.BadRequestError as e:
                    # Bad request - fail immediately
                    logger.error(f"Bad request error: {e}")
                    raise LLMAPIError(f"Bad request: {e}") from e
                except anthropic.APIError as e:
                    # Generic API errors - could be temporary, log and raise
                    logger.error(f"Anthropic API error: {type(e).__name__}: {e}")
                    raise LLMAPIError(f"API error: {e}") from e

            # Should never reach here, but just in case
            if last_exception:
                raise LLMRateLimitError(
                    f"Rate limit exceeded after {max_retries} retries: {last_exception}"
                ) from last_exception

            # We really shouldn't get here, but just in case
            raise LLMAPIError("Unknown error during Anthropic API call")

        return wrapper

    return decorator


class AnthropicLLM(ILLM):
    """Anthropic-based LLM implementation"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        models: Dict[SupportedModel, AnthropicModelConfig],
    ):
        super().__init__()
        self.client = client
        self.models = models

    def _encode_image(self, image: ImageInput) -> Dict[str, Any]:
        """
        Encode an image to base64 format for Anthropic API.

        Args:
            image: Can be a file path (str or Path), or raw bytes

        Returns:
            Dict with Anthropic image format: {"type": "image", "source": {...}}
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            # Read from file
            file_path = Path(image)
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")

            with open(file_path, "rb") as f:
                image_data = f.read()

            # Detect media type from file extension
            media_type = mimetypes.guess_type(str(file_path))[0]
            if not media_type or not media_type.startswith("image/"):
                # Default to jpeg if can't determine
                media_type = "image/jpeg"
        elif isinstance(image, bytes):
            # Raw bytes - assume jpeg
            image_data = image
            media_type = "image/jpeg"
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Encode to base64
        base64_data = base64.b64encode(image_data).decode("utf-8")

        # Return in Anthropic's format
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }

    def _create_content_with_images(
        self, text: str, images: Optional[ImagesInput] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Create message content with optional images.

        If no images, returns simple string.
        If images present, returns list of content blocks.

        Args:
            text: The text content
            images: Optional sequence of images

        Returns:
            Either a string (no images) or list of content blocks (with images)
        """
        if not images:
            return text

        # Build content blocks with images first, then text
        content_blocks: List[Dict[str, Any]] = []

        # Add image blocks
        for image in images:
            try:
                image_block = self._encode_image(image)
                content_blocks.append(image_block)
            except Exception as e:
                logger.error(f"Failed to encode image: {e}")
                # Continue with other images

        # Add text block
        if text:
            content_blocks.append({"type": "text", "text": text})

        return content_blocks

    def _make_api_call(self, stream: bool, **api_params):
        """
        Internal method to make the actual API call with retry logic.
        Wrapped separately so retry decorator can be applied.
        """

        @retry_on_rate_limit(max_retries=3, base_delay=1.0)
        def _call():
            if stream:
                return self.client.messages.stream(**api_params)
            else:
                return self.client.messages.create(**api_params)

        return _call()

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        caller: str,
        stream: bool = False,
        keep_alive: Optional[str] = None,  # Not used for Anthropic
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,  # Not used for Anthropic
        num_predict: Optional[int] = None,
        **kwargs,
    ):
        """Send chat request to Anthropic API with error handling and retry logic"""

        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available models: {list(self.models.keys())}"
            )

        config = self.models[model]

        # Check for special _raw_user_content parameter (used for images)
        raw_user_content = kwargs.pop("_raw_user_content", None)

        # Convert Message objects to Anthropic format
        # Anthropic expects system messages separately
        system_messages = [msg for msg in messages if msg.role == "system"]
        non_system_messages = [msg for msg in messages if msg.role != "system"]

        system_prompt = (
            "\n\n".join(msg.content for msg in system_messages)
            if system_messages
            else None
        )

        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in non_system_messages
        ]

        # If we have raw user content (with images), append it
        if raw_user_content is not None:
            anthropic_messages.append({"role": "user", "content": raw_user_content})

        # Build API parameters
        # Note: Anthropic doesn't allow both temperature and top_p
        api_params = {
            "model": model.value,
            "messages": anthropic_messages,
            "max_tokens": num_predict or config.max_tokens,
            "top_k": top_k or config.default_top_k,
            **kwargs,
        }

        # Use temperature OR top_p, not both (Anthropic restriction)
        # Prefer temperature if provided, otherwise use top_p
        if temperature is not None:
            api_params["temperature"] = temperature
        elif top_p is not None:
            api_params["top_p"] = top_p
        else:
            # Default to temperature
            api_params["temperature"] = config.default_temperature

        if system_prompt:
            api_params["system"] = system_prompt

        # Make API call with retry logic
        # Errors propagate naturally and are handled at the call site
        return self._make_api_call(stream, **api_params)

    def _extract_content_from_response(
        self, response: anthropic.types.Message
    ) -> Optional[str]:
        """Extract text content from Anthropic chat response"""
        # Anthropic returns content as a list of content blocks
        if response.content and len(response.content) > 0:
            # Get the first text block
            for block in response.content:
                if block.type == "text":
                    return block.text
        return None

    def is_model_available(self, model: SupportedModel) -> bool:
        """Check if a model is available (always True for API-based models)"""
        return model in self.models

    def pull_model(self, model: SupportedModel) -> bool:
        """Pull a model (no-op for API-based models)"""
        return True

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        stream: bool = False,
        keep_alive: Optional[str] = None,  # Not used for Anthropic
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,  # Not used for Anthropic
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
        **kwargs,
    ):
        """
        Simulate direct generation using chat with special system prompt.

        Anthropic doesn't have a separate generate API, so we simulate it
        by using chat with a system prompt that instructs Claude to continue
        the thought directly without meta-commentary.

        Supports images via Anthropic's vision API.
        """

        # Create system prompt for direct continuation
        system_message = Message(
            role="system",
            content="You are continuing your own internal thought/narrative. Complete the following as a direct continuation without meta-commentary, explanations, or formatting.",
        )

        # Create user message with optional images
        # Note: We need to handle this specially since Message expects string content
        # but Anthropic API needs content blocks when images are present
        user_content = self._create_content_with_images(prompt, images)

        # For now, we'll create the message with string content and handle
        # the conversion in chat() if needed
        if isinstance(user_content, str):
            user_message = Message(role="user", content=user_content)
            messages_to_send = [system_message, user_message]
        else:
            # We have content blocks - need to bypass Message class
            # and send raw message dict to chat
            messages_to_send = [system_message]
            # Add a special marker to handle this in chat
            kwargs["_raw_user_content"] = user_content

        # Use chat with the special system prompt
        return self.chat(
            model=model,
            messages=messages_to_send,
            caller=caller,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_predict=num_predict,
            **kwargs,
        )

    def _extract_content_from_generate_response(
        self, response: anthropic.types.Message
    ) -> str:
        """Extract text content from Anthropic generate response (same as chat)"""
        content = self._extract_content_from_response(response)
        return content or ""

    def chat_streaming(
        self, model: SupportedModel, messages: List[Message], caller: str, **kwargs
    ) -> Iterator[str]:
        """Override streaming to handle Anthropic's stream format"""
        import time

        start_time = time.time()

        with self.chat(model, messages, stream=True, caller=caller, **kwargs) as stream:  # type: ignore
            for text in stream.text_stream:
                yield text

        end_time = time.time()
        duration = end_time - start_time
        self.add_call_stats(caller, duration)

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        caller: str,
        images: ImagesInput = None,
        **kwargs,
    ) -> Iterator[Dict[str, str]]:
        """Override generate streaming to match Ollama's response format"""
        # Generate uses chat internally, so we can reuse chat_streaming
        import time

        start_time = time.time()

        system_message = Message(
            role="system",
            content="You are continuing your own internal thought/narrative. Complete the following as a direct continuation without meta-commentary, explanations, or formatting.",
        )

        # Handle images in user message
        user_content = self._create_content_with_images(prompt, images)

        # Prepare messages and kwargs for chat_streaming
        if isinstance(user_content, str):
            # Simple text message
            messages_to_send = [
                system_message,
                Message(role="user", content=user_content),
            ]
        else:
            # Content blocks with images
            messages_to_send = [system_message]
            kwargs["_raw_user_content"] = user_content

        # Yield chunks in Ollama-compatible format (dict with "response" key)
        for text_chunk in self.chat_streaming(
            model, messages_to_send, caller, **kwargs
        ):
            yield {"response": text_chunk}

        end_time = time.time()
        duration = end_time - start_time
        self.add_call_stats(caller, duration)

        logger.info(f"LLM call [{caller}]: {duration:.2f}s")


# Default Anthropic model configurations
DEFAULT_ANTHROPIC_MODELS = {
    SupportedModel.CLAUDE_SONNET_4_5: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_SONNET_4_5,
        max_tokens=4096,
        default_temperature=0.3,
        default_top_p=0.9,
        default_top_k=50,
        context_window=200000,
        estimated_token_size=3.4,
    ),
    SupportedModel.CLAUDE_OPUS_4_1: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_OPUS_4_1,
        max_tokens=4096,
        default_temperature=0.3,
        default_top_p=0.9,
        default_top_k=50,
        context_window=200000,
        estimated_token_size=3.4,
    ),
    SupportedModel.CLAUDE_HAIKU_4_5: AnthropicModelConfig(
        model=SupportedModel.CLAUDE_HAIKU_4_5,
        max_tokens=8192,
        default_temperature=0.3,
        default_top_p=0.9,
        default_top_k=50,
        context_window=200000,
        estimated_token_size=3.4,
    ),
}


def create_anthropic_llm(
    api_key: Optional[str] = None,
    models: Optional[Dict[SupportedModel, AnthropicModelConfig]] = None,
) -> AnthropicLLM:
    """Create an Anthropic LLM instance"""
    # If no API key provided, anthropic.Anthropic() will look for ANTHROPIC_API_KEY env var
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    model_configs = models or DEFAULT_ANTHROPIC_MODELS
    return AnthropicLLM(client, model_configs)
