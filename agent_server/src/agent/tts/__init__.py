"""TTS (Text-to-Speech) module for the agent system.

This module provides:
- TTSService: Production service with background rendering
- TTSProvider: Abstract base class for TTS providers
- ChatterboxProvider: Chatterbox TTS implementation

Usage:
    from agent.tts import TTSService
    from agent.tts.providers import ChatterboxProvider

    provider = ChatterboxProvider(device="cuda")
    service = TTSService(
        provider=provider,
        reference_audio=Path("path/to/reference.wav"),
        output_dir=Path("generated_audio"),
    )
    service.start()

    # Queue audio for background rendering
    service.queue_render("action_123", "Hello world!", "happy")

    # Check status
    status = service.get_audio_status("action_123")

    # Get audio when ready
    if status == RenderStatus.READY:
        audio_path = service.get_audio_path("action_123")
"""

from .base import TTSProvider, TTSResult
from .service import RenderStatus, TTSService

__all__ = [
    "TTSProvider",
    "TTSResult",
    "TTSService",
    "RenderStatus",
]
