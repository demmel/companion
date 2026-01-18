"""Chatterbox TTS provider implementation."""

import io
import wave
from pathlib import Path

from ..base import TTSProvider, TTSResult


class ChatterboxProvider(TTSProvider):
    """TTS provider using Chatterbox.

    Chatterbox supports:
    - Paralinguistic tags like [laugh], [sigh], [happy], [whispering]
    - Few-shot voice cloning from reference audio
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the Chatterbox provider.

        Args:
            device: The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self._model: object | None = None

    def _ensure_model(self) -> None:
        """Lazy-load the model."""
        if self._model is None:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)

            # Fix dtype mismatch: norm_loudness multiplies by float64, which promotes
            # the entire wav array to float64, causing downstream torch stft errors
            import numpy as np

            original_norm = self._model.norm_loudness

            def patched_norm(
                wav: "np.ndarray", sr: int, target_lufs: int = -27
            ) -> "np.ndarray":
                result = original_norm(wav, sr, target_lufs)
                return result.astype(np.float32) if result.dtype == np.float64 else result

            self._model.norm_loudness = patched_norm

    @property
    def name(self) -> str:
        return "chatterbox"

    def generate(
        self,
        text: str,
        reference_audio: Path | None = None,
    ) -> TTSResult:
        """Generate audio from text using Chatterbox.

        Args:
            text: The text to synthesize (may contain paralinguistic tags).
            reference_audio: Optional reference audio for voice cloning.

        Returns:
            TTSResult containing the generated audio.
        """
        self._ensure_model()
        assert self._model is not None

        # Text already has paralinguistic tags from LLM rewriting
        wav = self._model.generate(
            text=text,
            audio_prompt_path=str(reference_audio) if reference_audio else None,
        )

        # Convert tensor to WAV bytes and calculate duration
        audio_data, num_samples = self._tensor_to_wav_bytes(wav, sample_rate=24000)
        duration_ms = int(num_samples / 24000 * 1000)

        return TTSResult(
            audio_data=audio_data,
            sample_rate=24000,
            format="wav",
            duration_ms=duration_ms,
        )

    def _tensor_to_wav_bytes(
        self, wav_tensor: object, sample_rate: int = 24000
    ) -> tuple[bytes, int]:
        """Convert a PyTorch tensor to WAV bytes.

        Args:
            wav_tensor: The audio tensor.
            sample_rate: The sample rate.

        Returns:
            Tuple of (WAV file bytes, number of samples).
        """
        import numpy as np
        import torch

        if isinstance(wav_tensor, torch.Tensor):
            wav_np = wav_tensor.cpu().numpy()
        else:
            wav_np = np.array(wav_tensor)

        # Ensure 1D
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()

        num_samples = len(wav_np)

        # Normalize to int16 range
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767).astype(np.int16)

        # Write to WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(wav_int16.tobytes())

        return buffer.getvalue(), num_samples
