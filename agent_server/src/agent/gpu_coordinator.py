"""Process-wide GPU compute coordinator.

The companion runs three GPU consumers that must never compute at the same time
on a single 24GB card, because overlapping GPU compute causes severe thrashing:

- the Ollama LLM (resident in VRAM, the hot path),
- the Chatterbox TTS model (in-process torch, offloaded to CPU when idle),
- the SDXL image pipeline (in-process diffusers, offloaded to CPU when idle).

This module provides a single global lock that every GPU operation acquires for
its full duration via :meth:`GpuCoordinator.lease`, guaranteeing mutual
exclusion. The lease also moves swappable models on/off the GPU and clears the
torch allocator cache on release so idle models do not crowd VRAM.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class GpuCoordinator:
    """Serializes all GPU compute through one lock and enforces CPU offload.

    A lease must NEVER nest another lease on the same thread: the lock is plain
    (non-reentrant) so accidental nesting deadlocks loudly rather than silently
    permitting overlap. Leases wrap only the raw GPU op (LLM request, audio
    synthesis, diffusion forward), never an LLM call that itself takes a lease.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def lease(
        self,
        name: str,
        to_gpu: Optional[Callable[[], None]] = None,
        to_cpu: Optional[Callable[[], None]] = None,
    ) -> Iterator[None]:
        """Hold the global GPU lock for the duration of a GPU operation.

        Args:
            name: Short identifier for the consumer (for logging/debugging).
            to_gpu: Optional callback that moves the model onto the GPU. Called
                after the lock is acquired, before the operation runs.
            to_cpu: Optional callback that moves the model back to CPU. Called
                after the operation completes (even on error), before the cache
                is cleared and the lock is released.
        """
        with self._lock:
            logger.debug("GPU lease acquired: %s", name)
            try:
                if to_gpu is not None:
                    to_gpu()
                yield
            finally:
                if to_cpu is not None:
                    to_cpu()
                self._empty_cache()
                logger.debug("GPU lease released: %s", name)

    def _empty_cache(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_shared: Optional[GpuCoordinator] = None


def get_gpu_coordinator() -> GpuCoordinator:
    """Return the process-wide GPU coordinator singleton."""
    global _shared
    if _shared is None:
        _shared = GpuCoordinator()
    return _shared
