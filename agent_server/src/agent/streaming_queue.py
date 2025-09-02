"""
Simple queue-based streaming for callbacks.
"""

import queue
import threading
from typing import Iterator, TypeVar, Optional, Generic

T = TypeVar("T")


class StreamingQueue(Generic[T]):
    """Queue-based streaming that works with synchronous callbacks"""

    def __init__(self):
        self._queue: queue.Queue[Optional[T]] = queue.Queue()
        self._finished = False

    def emit(self, event: T) -> None:
        """Emit an event from a synchronous callback"""
        if not self._finished:
            self._queue.put(event)

    def finish(self) -> None:
        """Signal that no more events will be emitted"""
        self._finished = True
        self._queue.put(None)  # Sentinel to stop iteration

    def stream_while(self, work_func) -> Iterator[T]:
        """
        Stream events while work_func executes.

        work_func should call emit() during execution and will run in a separate thread.
        """
        # Start work in background thread
        thread = threading.Thread(target=self._run_work, args=(work_func,))
        thread.start()

        try:
            # Yield events as they arrive
            while True:
                event = self._queue.get()
                if event is None:  # Sentinel - work is done
                    break
                yield event
        finally:
            # Ensure thread completes
            thread.join()

    def _run_work(self, work_func):
        """Run work function and signal completion"""
        try:
            work_func()
        finally:
            self.finish()
