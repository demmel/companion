from contextlib import contextmanager
import logging
import time


@contextmanager
def timeit(label: str, level: int = logging.INFO):
    """Context manager to time a block of code and log the duration."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logging.log(level, f"[{label}] {duration:.4f}s")
