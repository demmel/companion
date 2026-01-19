"""
ChromaDB wrapper for Windows file handle cleanup.

ChromaDB's PersistentClient doesn't properly release file handles on Windows,
which prevents deleting the data directory. This module provides a wrapper
that tracks open clients and performs cleanup when the last one closes.
"""

import gc
import time

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

_open_count = 0


class ChromaClient:
    """
    Wrapper around ChromaDB PersistentClient that handles Windows file cleanup.

    Tracks all open instances and only performs expensive cleanup (cache clear,
    gc, sleep) when the last instance is closed.

    Access the underlying client via the `client` property.
    """

    def __init__(self, path: str):
        global _open_count
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        _open_count += 1

    def client(self) -> ClientAPI:
        """The underlying ChromaDB client."""
        return self._client

    def close(self) -> None:
        """Close this client and release file handles when all clients are closed."""
        from chromadb.api.client import Client as RawChromaClient

        global _open_count

        # Stop this client's system
        try:
            self._client._system.stop()  # type: ignore[attr-defined]
        except (KeyError, AttributeError):
            pass

        _open_count -= 1

        # Only do expensive cleanup when last client closes
        if _open_count == 0:
            RawChromaClient.clear_system_cache()
            gc.collect()
            gc.collect()  # Second pass for circular references
            time.sleep(0.1)  # Give Windows time to release handles
