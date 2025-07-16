"""
LLM client for interfacing with Ollama
"""

import ollama
from typing import Iterator, List, Dict, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum


class Message(BaseModel):
    role: str
    content: str


class LLMClient:
    """Client for interfacing with Ollama LLM"""

    def __init__(
        self,
        client: ollama.Client,
        model: str,
        context_window: int = 32768,
    ):
        self.context_window = context_window
        self.client = client
        self.model = model

    def chat(self, messages: List[Message]) -> Iterator[ollama.ChatResponse]:
        """Send streaming chat request to LLM"""
        # Convert messages to dict format
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        response = self.client.chat(
            model=self.model,
            messages=message_dicts,
            stream=True,
            options={
                "num_gpu": -1,  # Use all available GPU layers
                "num_thread": 16,  # More CPU threads for large context processing
                "num_ctx": self.context_window,  # Configurable context window
                "temperature": 0.3,  # Lower temp for better coherence
                "top_p": 0.8,  # Slightly more focused sampling
                "top_k": 40,  # Add top-k sampling for stability
                "repeat_penalty": 1.1,  # Prevent repetition
                "num_predict": 512,  # Allow longer responses
            },
            keep_alive="30m",  # Keep model loaded for 10 minutes
        )

        return response

    def chat_complete(self, messages: List[Message]) -> str:
        """Send non-streaming chat request for background operations like summarization"""

        # Convert messages to dict format
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        response = self.client.chat(
            model=self.model,
            messages=message_dicts,
            stream=False,
            options={
                "num_gpu": -1,  # Use all available GPU layers
                "num_thread": 16,  # More CPU threads for large context processing
                "num_ctx": self.context_window,  # Configurable context window
                "temperature": 0.3,  # Lower temp for better coherence
                "top_p": 0.8,  # Slightly more focused sampling
                "top_k": 40,  # Add top-k sampling for stability
                "repeat_penalty": 1.1,  # Prevent repetition
                "num_predict": 512,  # Allow longer responses
            },
            keep_alive="10m",  # Keep model loaded for 10 minutes
        )

        return response["message"]["content"]

    def is_available(self) -> bool:
        """Check if the model is available"""
        try:
            models = self.client.list()
            model_names = [model["name"] for model in models["models"]]
            return self.model in model_names
        except:
            return False

    def pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            self.client.pull(self.model)
            return True
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False


class SupportedModel(str, Enum):
    """Supported models with backend-specific identifiers"""

    MISTRAL_SMALL = "huihui_ai/mistral-small-abliterated"
    MISTRAL_NEMO = "mistral-nemo:latest"
    DOLPHIN_MISTRAL_NEMO = "CognitiveComputations/dolphin-mistral-nemo:latest"
    LLAMA_8B = "llama3.1:8b"
    GEMMA_27B = "aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m"
    DEEPSEEK_R1_14B = "huihui_ai/deepseek-r1-abliterated:14b"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    model: SupportedModel
    keep_alive: str = "30m"
    default_temperature: float = 0.1
    default_top_p: float = 0.8
    default_top_k: int = 40
    default_repeat_penalty: float = 1.1
    default_num_predict: int = 4096
    context_window: int = 32768


class LLM:
    """Centralized LLM management with model-specific configurations"""

    def __init__(
        self, client: ollama.Client, models: Dict[SupportedModel, ModelConfig]
    ):
        self.client = client
        self.models = models

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
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
        self, model: SupportedModel, messages: List[Message], **kwargs
    ) -> Iterator[ollama.ChatResponse]:
        """Convenience method for streaming chat"""
        return self.chat(model, messages, stream=True, **kwargs)  # type: ignore

    def chat_complete(
        self, model: SupportedModel, messages: List[Message], **kwargs
    ) -> Optional[str]:
        """Convenience method for non-streaming chat"""
        response: ollama.ChatResponse = self.chat(
            model, messages, stream=False, **kwargs
        )  # type: ignore
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
        default_top_k=50,
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
