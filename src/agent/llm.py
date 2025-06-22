"""
LLM client for interfacing with Ollama
"""

import ollama
from typing import Dict, List, Optional, Any
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

    def chat(self, messages: List[Message], stream: bool = False) -> str:
        """Send chat request to LLM"""
        try:
            # Convert messages to dict format
            message_dicts = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            response = self.client.chat(
                model=self.model,
                messages=message_dicts,
                stream=stream,
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

            if stream:
                return response
            else:
                return response["message"]["content"]

        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate response from prompt"""
        try:
            response = self.client.generate(
                model=self.model, prompt=prompt, system=system
            )
            return response["response"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

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
