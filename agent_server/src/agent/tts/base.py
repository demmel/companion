"""Abstract base class and types for TTS providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    """Result from TTS generation."""

    audio_data: bytes
    sample_rate: int
    format: str  # "wav"
    duration_ms: int


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: Path | None = None,
    ) -> TTSResult:
        """Generate audio from text.

        Args:
            text: The text to synthesize (may contain paralinguistic tags).
            reference_audio: Optional reference audio for voice cloning.

        Returns:
            TTSResult containing the generated audio.
        """
        pass
