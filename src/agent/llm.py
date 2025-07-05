"""
LLM client for interfacing with Ollama
"""

import ollama
from typing import Iterator, List
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class LLMClient:
    """Client for interfacing with Ollama LLM"""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "localhost:11434",
        context_window: int = 32768,
    ):
        self.model = model
        self.host = host
        self.context_window = context_window
        self.client = ollama.Client(host=host)

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
            keep_alive="10m",  # Keep model loaded for 10 minutes
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
