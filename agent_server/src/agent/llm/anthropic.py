"""
Anthropic LLM implementation
"""

import anthropic
import logging
import base64
from typing import Iterator, List, Dict, Optional
from pathlib import Path
from agent.llm.interface import ILLM, Message, ImagesInput, ImageInput
from agent.llm.models import SupportedModel, AnthropicModelConfig

logger = logging.getLogger(__name__)


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
        """Send chat request to Anthropic API"""

        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available models: {list(self.models.keys())}"
            )

        config = self.models[model]

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

        # Build API parameters
        api_params = {
            "model": model.value,
            "messages": anthropic_messages,
            "max_tokens": num_predict or config.max_tokens,
            "temperature": temperature or config.default_temperature,
            "top_p": top_p or config.default_top_p,
            "top_k": top_k or config.default_top_k,
            **kwargs,
        }

        if system_prompt:
            api_params["system"] = system_prompt

        if stream:
            return self.client.messages.stream(**api_params)
        else:
            return self.client.messages.create(**api_params)

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
        """

        # Create system prompt for direct continuation
        system_message = Message(
            role="system",
            content="You are continuing your own internal thought/narrative. Complete the following as a direct continuation without meta-commentary, explanations, or formatting.",
        )

        # Handle images in user message
        user_content = prompt
        if images:
            # For now, we'll add image support later if needed
            # Anthropic requires base64 encoded images in a specific format
            logger.warning(
                "Image support in generate() not yet implemented for Anthropic"
            )

        user_message = Message(role="user", content=user_content)

        # Use chat with the special system prompt
        return self.chat(
            model=model,
            messages=[system_message, user_message],
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
    ) -> Iterator[str]:
        """Override generate streaming to handle Anthropic's stream format"""
        # Generate uses chat internally, so we can reuse chat_streaming
        import time

        start_time = time.time()

        system_message = Message(
            role="system",
            content="You are continuing your own internal thought/narrative. Complete the following as a direct continuation without meta-commentary, explanations, or formatting.",
        )
        user_message = Message(role="user", content=prompt)

        yield from self.chat_streaming(
            model, [system_message, user_message], caller, **kwargs
        )

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
