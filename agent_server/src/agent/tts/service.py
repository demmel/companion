"""Production TTS service with background rendering."""

import io
import logging
import queue
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from agent.llm.interface import Message
from agent.llm.models import SupportedModel
from agent.llm.router import LLM

from .base import TTSProvider
from .text_processor import normalize_text_for_tts, split_into_paragraphs

logger = logging.getLogger(__name__)


def _convert_wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV audio bytes to MP3.

    Requires ffmpeg. By default, pydub finds ffmpeg on the system PATH.
    Set FFMPEG_PATH env var to override with a custom path.
    """
    import os
    from pydub import AudioSegment

    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path

    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3", bitrate="128k")
    return mp3_buffer.getvalue()


def rewrite_for_tts(
    text: str, tone: str | None, llm: LLM, model: SupportedModel
) -> str:
    """Use LLM to rewrite text for expressive TTS.

    - Removes emojis (cannot be spoken)
    - Removes narrative descriptions ("I purr", "she whispers")
    - Converts action markers (*laughs*) to tags ([laugh])
    - Inserts paralinguistic tags throughout based on tone
    """
    prompt = f"""Prepare this text for expressive text-to-speech by inserting paralinguistic tags.

RULES:
1. REMOVE emojis - they cannot be spoken
2. REMOVE narrative descriptions of speech (like "I purr", "she whispers", "my voice carries warmth")
3. CONVERT action markers (*text*) to tags when possible:
   - *laughs*, *chuckles* -> [laugh] or [chuckle]
   - *sighs* -> [sigh]
   - *gasps* -> [gasp]
   - *whispers* -> [whispering]
4. REMOVE unconvertible action markers (*leans forward*, *smiles*, *tilts head*)
5. LIBERALLY INSERT tags throughout based on emotional content and tone:
   - Excited/happy text: add [happy], [chuckle], [laugh] at natural pause points
   - Wistful/longing text: add [sigh] before reflective phrases
   - Intimate/secretive: add [whispering] for tender moments
   - Surprised: add [gasp] before revelations
   - For text longer than 2 sentences, insert AT LEAST 3-5 tags spread throughout

Supported tags:
   Sounds: [laugh], [chuckle], [sigh], [gasp], [cough], [sniff], [groan], [shush], [clear throat], [crying]
   Emotions: [happy], [angry], [fear], [surprised], [sarcastic], [dramatic], [whispering]

Tone: {tone or 'neutral'}

Original text:
{text}

Output ONLY the processed text with tags inserted. No explanations."""

    response = llm.chat(
        model=model,
        messages=[Message(role="user", content=prompt)],
        caller="tts_rewrite",
    )
    return response.strip() if response else text


class RenderStatus(Enum):
    """Status of an audio render request."""

    PENDING = "pending"
    RENDERING = "rendering"
    READY = "ready"
    ERROR = "error"


@dataclass
class RenderRequest:
    """A request to render audio for a speak action."""

    action_id: str
    text: str
    tone: str | None


class TTSService:
    """Production TTS service with background rendering.

    This service manages a background worker thread that processes TTS requests
    asynchronously. Audio files are saved to disk and can be retrieved by action ID.
    """

    def __init__(
        self,
        provider: TTSProvider,
        reference_audio: Path,
        output_dir: Path,
        llm: LLM,
        tts_rewrite_model: SupportedModel,
        on_audio_ready: Callable[[str, str], None] | None = None,
    ):
        """Initialize the TTS service.

        Args:
            provider: The TTS provider to use for generation.
            reference_audio: Path to the reference audio for voice cloning.
            output_dir: Directory to save generated audio files.
            llm: The LLM instance for text rewriting.
            tts_rewrite_model: The model to use for TTS text rewriting.
            on_audio_ready: Optional callback when audio is ready (action_id, audio_url).
        """
        self.provider = provider
        self.reference_audio = reference_audio
        self.output_dir = output_dir
        self.llm = llm
        self.tts_rewrite_model = tts_rewrite_model
        self.on_audio_ready = on_audio_ready

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Request queue and status tracking
        self._render_queue: queue.Queue[RenderRequest | None] = queue.Queue()
        self._status: dict[str, RenderStatus] = {}
        self._status_lock = threading.Lock()

        # Worker thread
        self._worker_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the background render worker."""
        if self._worker_thread is not None:
            logger.warning("TTS service already started")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="tts-worker", daemon=True
        )
        self._worker_thread.start()
        logger.info("TTS service started")

    def stop(self) -> None:
        """Stop the background worker gracefully."""
        if self._worker_thread is None:
            return

        logger.info("Stopping TTS service...")
        self._running = False

        # Send sentinel to unblock the worker
        self._render_queue.put(None)

        # Wait for worker to finish
        self._worker_thread.join(timeout=5.0)
        self._worker_thread = None
        logger.info("TTS service stopped")

    def queue_render(self, action_id: str, text: str, tone: str | None) -> None:
        """Queue audio for background rendering.

        Args:
            action_id: Unique identifier for this speak action.
            text: The text to synthesize.
            tone: Optional emotional tone hint.
        """
        # Set initial status
        with self._status_lock:
            self._status[action_id] = RenderStatus.PENDING

        # Add to queue
        request = RenderRequest(action_id=action_id, text=text, tone=tone)
        self._render_queue.put(request)
        logger.info(f"Queued TTS render for action {action_id}")

    def get_audio_path(self, action_id: str) -> Path | None:
        """Get path to rendered audio, or None if not ready.

        Args:
            action_id: The action identifier.

        Returns:
            Path to the audio file if ready, None otherwise.
        """
        audio_path = self.output_dir / f"{action_id}.mp3"
        if audio_path.exists():
            return audio_path
        return None

    def get_audio_status(self, action_id: str) -> RenderStatus:
        """Check render status for an action.

        Args:
            action_id: The action identifier.

        Returns:
            Current render status.
        """
        with self._status_lock:
            return self._status.get(action_id, RenderStatus.ERROR)

    def _worker_loop(self) -> None:
        """Background worker loop that processes render requests."""
        logger.info("TTS worker started")

        while self._running:
            try:
                # Get next request (blocks with timeout)
                try:
                    request = self._render_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check for sentinel (stop signal)
                if request is None:
                    break

                # Process the request
                self._process_request(request)

            except Exception as e:
                logger.error(f"TTS worker error: {e}", exc_info=True)

        logger.info("TTS worker stopped")

    def _process_request(self, request: RenderRequest) -> None:
        """Process a single render request.

        Args:
            request: The render request to process.
        """
        action_id = request.action_id

        # Update status to rendering
        with self._status_lock:
            self._status[action_id] = RenderStatus.RENDERING

        try:
            logger.info(f"Rendering audio for action {action_id}")

            # LLM rewrite to insert paralinguistic tags
            text = rewrite_for_tts(
                request.text, request.tone, self.llm, self.tts_rewrite_model
            )

            # Normalize text
            text = normalize_text_for_tts(text)

            # Split into paragraphs for long text
            paragraphs = split_into_paragraphs(text)

            # Generate audio - no longer passing emotion
            if len(paragraphs) <= 1:
                # Short text - generate directly
                result = self.provider.generate(
                    text=text,
                    reference_audio=self.reference_audio,
                )
                audio_data = result.audio_data
            else:
                # Long text - generate paragraph by paragraph and concatenate
                logger.info(f"Splitting into {len(paragraphs)} paragraphs")
                audio_data = self._generate_paragraphs(paragraphs)

            # Convert WAV to MP3 for compression
            mp3_data = _convert_wav_to_mp3(audio_data)

            # Save MP3 file
            output_path = self.output_dir / f"{action_id}.mp3"
            output_path.write_bytes(mp3_data)

            # Update status
            with self._status_lock:
                self._status[action_id] = RenderStatus.READY

            logger.info(f"Audio ready for action {action_id}: {output_path}")

            # Notify callback if set
            if self.on_audio_ready:
                audio_url = f"/generated_audio/{action_id}.mp3"
                self.on_audio_ready(action_id, audio_url)

        except Exception as e:
            logger.error(f"Failed to render audio for action {action_id}: {e}")
            with self._status_lock:
                self._status[action_id] = RenderStatus.ERROR

    def _generate_paragraphs(self, paragraphs: list[str]) -> bytes:
        """Generate audio for multiple paragraphs and concatenate.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            Concatenated WAV audio data.
        """
        all_audio_data: list[bytes] = []

        for i, paragraph in enumerate(paragraphs):
            logger.debug(f"Generating paragraph {i + 1}/{len(paragraphs)}")
            result = self.provider.generate(
                text=paragraph,
                reference_audio=self.reference_audio,
            )

            # Extract raw PCM from WAV
            with wave.open(io.BytesIO(result.audio_data), "rb") as wf:
                all_audio_data.append(wf.readframes(wf.getnframes()))

        # Combine all audio into single WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)  # 16-bit
            out_wf.setframerate(24000)  # Chatterbox uses 24kHz
            for audio_chunk in all_audio_data:
                out_wf.writeframes(audio_chunk)

        return buffer.getvalue()
