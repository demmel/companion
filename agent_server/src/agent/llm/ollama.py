"""
Ollama LLM implementation
"""

import ollama
import logging
import threading
import queue
from typing import Callable, Iterator, List, Dict, Optional, TypeVar, Union, cast

from agent.llm.interface import ILLM, Message, ImagesInput
from agent.llm.models import SupportedModel, OllamaModelConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OllamaLLM(ILLM):
    """Ollama-based LLM implementation"""

    def __init__(
        self, client: ollama.Client, models: Dict[SupportedModel, OllamaModelConfig]
    ) -> None:
        self.client = client
        self.models = models

    def _get_config(self, model: SupportedModel) -> OllamaModelConfig:
        if model not in self.models:
            raise ValueError(
                f"Model {model} not configured. Available: {list(self.models.keys())}"
            )
        return self.models[model]

    def _build_options(
        self,
        config: OllamaModelConfig,
        temperature: Optional[float],
        num_predict: Optional[int],
    ) -> Dict[str, object]:
        return {
            "num_gpu": -1,
            "num_thread": 32,
            "num_ctx": config.context_window,
            "temperature": temperature if temperature is not None else config.default_temperature,
            "top_p": config.default_top_p,
            "top_k": config.default_top_k,
            "repeat_penalty": config.default_repeat_penalty,
            "num_predict": num_predict if num_predict is not None else config.default_num_predict,
        }

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        config = self._get_config(model)
        options = self._build_options(config, temperature, num_predict)
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        response = self.client.chat(
            model=model.value,
            messages=message_dicts,
            stream=False,
            options=options,
            keep_alive=config.keep_alive,
        )
        return response["message"]["content"]  # type: ignore[return-value]

    def chat_streaming(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Iterator[str]:
        config = self._get_config(model)
        options = self._build_options(config, temperature, num_predict)
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]

        for chunk in self.client.chat(
            model=model.value,
            messages=message_dicts,
            stream=True,
            options=options,
            keep_alive=config.keep_alive,
        ):
            content = chunk["message"]["content"]
            if content:
                yield content

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> str:
        config = self._get_config(model)
        options = self._build_options(config, temperature, num_predict)

        response = self.client.generate(
            model=model.value,
            prompt=prompt,
            stream=False,
            options=options,
            keep_alive=config.keep_alive,
            images=[ollama.Image(value=img) for img in images] if images else None,
        )
        return response["response"]  # type: ignore[return-value]

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> Iterator[str]:
        config = self._get_config(model)
        options = self._build_options(config, temperature, num_predict)

        for chunk in self.client.generate(
            model=model.value,
            prompt=prompt,
            stream=True,
            options=options,
            keep_alive=config.keep_alive,
            images=[ollama.Image(value=img) for img in images] if images else None,
        ):
            text = chunk["response"]
            if text:
                yield text

    def is_model_available(self, model: SupportedModel) -> bool:
        try:
            models = self.client.list()
            model_names = [m["name"] for m in models["models"]]
            return model.value in model_names
        except Exception:
            return False

    def pull_model(self, model: SupportedModel) -> bool:
        try:
            self.client.pull(model.value)
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False


class SerializedOllamaLLM(ILLM):
    """
    Wrapper that serializes all requests through a background worker.
    Prevents concurrent Ollama requests which cause performance degradation.
    """

    def __init__(self, inner: OllamaLLM) -> None:
        self._inner = inner
        self._queue: queue.Queue[Optional[Callable[[], None]]] = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="SerializedOllamaWorker"
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                break
            fn()

    def stop(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5.0)

    def models(self) -> Dict[SupportedModel, OllamaModelConfig]:
        return self._inner.models

    def _run_sync(self, fn: Callable[[], T]) -> T:
        result: List[T] = []
        error: List[Exception] = []
        done = threading.Event()

        def work() -> None:
            try:
                result.append(fn())
            except Exception as e:
                error.append(e)
            done.set()

        self._queue.put(work)
        done.wait()
        if error:
            raise error[0]
        return result[0]

    def _run_streaming(self, fn: Callable[[], Iterator[T]]) -> Iterator[T]:
        chunks: queue.Queue[Union[T, Exception, None]] = queue.Queue()

        def work() -> None:
            try:
                for chunk in fn():
                    chunks.put(chunk)
                chunks.put(None)
            except Exception as e:
                chunks.put(e)

        self._queue.put(work)

        while True:
            item = chunks.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def chat(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        return self._run_sync(
            lambda: self._inner.chat(model, messages, temperature, num_predict)
        )

    def chat_streaming(
        self,
        model: SupportedModel,
        messages: List[Message],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> Iterator[str]:
        return self._run_streaming(
            lambda: self._inner.chat_streaming(model, messages, temperature, num_predict)
        )

    def generate(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> str:
        return self._run_sync(
            lambda: self._inner.generate(model, prompt, temperature, num_predict, images)
        )

    def generate_streaming(
        self,
        model: SupportedModel,
        prompt: str,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        images: ImagesInput = None,
    ) -> Iterator[str]:
        return self._run_streaming(
            lambda: self._inner.generate_streaming(model, prompt, temperature, num_predict, images)
        )

    def is_model_available(self, model: SupportedModel) -> bool:
        return self._inner.is_model_available(model)

    def pull_model(self, model: SupportedModel) -> bool:
        return self._inner.pull_model(model)


# Default Ollama model configurations
DEFAULT_OLLAMA_MODELS = {
    SupportedModel.LLAMA_3B: OllamaModelConfig(model=SupportedModel.LLAMA_3B),
    SupportedModel.LLAMA_8B: OllamaModelConfig(model=SupportedModel.LLAMA_8B),
    SupportedModel.GEMMA_27B: OllamaModelConfig(model=SupportedModel.GEMMA_27B),
    SupportedModel.MISTRAL_SMALL: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL, estimated_token_size=3.4
    ),
    SupportedModel.MISTRAL_SMALL_3_2: OllamaModelConfig(
        model=SupportedModel.MISTRAL_SMALL_3_2, estimated_token_size=3.4
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
        model=SupportedModel.DOLPHIN_MISTRAL_NEMO
    ),
    SupportedModel.MISTRAL_NEMO: OllamaModelConfig(model=SupportedModel.MISTRAL_NEMO),
    SupportedModel.DEEPSEEK_R1_14B: OllamaModelConfig(
        model=SupportedModel.DEEPSEEK_R1_14B,
        default_temperature=0.6,
        default_repeat_penalty=1.2,
        default_top_p=0.95,
    ),
    SupportedModel.RP_MAX: OllamaModelConfig(model=SupportedModel.RP_MAX),
}


def create_ollama_llm(
    host: str = "localhost:11434",
    models: Optional[Dict[SupportedModel, OllamaModelConfig]] = None,
) -> SerializedOllamaLLM:
    """Create a serialized Ollama LLM instance."""
    client = ollama.Client(host=host)
    model_configs = models or DEFAULT_OLLAMA_MODELS
    inner = OllamaLLM(client, model_configs)
    return SerializedOllamaLLM(inner)
