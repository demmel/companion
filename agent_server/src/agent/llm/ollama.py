"""
Ollama LLM implementation
"""

import ollama
import logging
from typing import Iterator, List, Dict, Optional
from agent.llm.interface import ILLM, Message, ImagesInput
from agent.llm.models import SupportedModel, OllamaModelConfig

logger = logging.getLogger(__name__)


class OllamaLLM(ILLM):
    """Ollama-based LLM implementation"""

    def __init__(
        self, client: ollama.Client, models: Dict[SupportedModel, OllamaModelConfig]
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

    def _extract_content_from_response(
        self, response: ollama.ChatResponse
    ) -> Optional[str]:
        """Extract text content from Ollama chat response"""
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
        images: ImagesInput = None,
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
            images=[ollama.Image(value=image) for image in images] if images else None,
        )

    def _extract_content_from_generate_response(
        self, response: ollama.GenerateResponse
    ) -> str:
        """Extract text content from Ollama generate response"""
        return response["response"]  # type: ignore


# Default Ollama model configurations
DEFAULT_OLLAMA_MODELS = {
    SupportedModel.LLAMA_8B: OllamaModelConfig(
        model=SupportedModel.LLAMA_8B,
    ),
    SupportedModel.GEMMA_27B: OllamaModelConfig(
        model=SupportedModel.GEMMA_27B,
    ),
    SupportedModel.MISTRAL_SMALL: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL,
        estimated_token_size=3.4,
    ),
    SupportedModel.MISTRAL_SMALL_3_2: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL_3_2,
        estimated_token_size=3.4,
    ),
    SupportedModel.MISTRAL_SMALL_3_2_Q4: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL_3_2_Q4,
        estimated_token_size=3.4,
        default_temperature=0.15,
    ),
    SupportedModel.MISTRAL_SMALL_3_2_Q8: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL_3_2_Q8,
        estimated_token_size=3.4,
        default_temperature=0.15,
    ),
    SupportedModel.DOLPHIN_MISTRAL_NEMO: OllamaModelConfig(
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO,
    ),
    SupportedModel.MISTRAL_NEMO: OllamaModelConfig(
        model=SupportedModel.MISTRAL_NEMO,
    ),
    SupportedModel.DEEPSEEK_R1_14B: OllamaModelConfig(
        model=SupportedModel.DEEPSEEK_R1_14B,
        default_temperature=0.6,
        default_repeat_penalty=1.2,
        default_top_p=0.95,
    ),
    SupportedModel.RP_MAX: OllamaModelConfig(
        model=SupportedModel.RP_MAX,
    ),
}


def create_ollama_llm(
    host: str = "localhost:11434",
    models: Optional[Dict[SupportedModel, OllamaModelConfig]] = None,
) -> OllamaLLM:
    """Create an Ollama LLM instance"""
    client = ollama.Client(host=host)
    model_configs = models or DEFAULT_OLLAMA_MODELS
    return OllamaLLM(client, model_configs)
