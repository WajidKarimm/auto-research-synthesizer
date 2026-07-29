"""Phase 6 cache package: lightweight TTL cache with a Redis-shaped seam later."""

from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small in-memory TTL cache for single-process development runs."""

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: dict[str, CacheEntry[T]] = {}

    def get(self, key: str) -> T | None:
        entry = self._items.get(key)
        if entry is None:
            return None

        if entry.expires_at <= monotonic():
            self._items.pop(key, None)
            return None

        return entry.value

    def set(self, key: str, value: T) -> None:
        self._prune()
        self._items[key] = CacheEntry(
            value=value,
            expires_at=monotonic() + self.ttl_seconds,
        )

    def clear(self) -> None:
        self._items.clear()

    def _prune(self) -> None:
        now = monotonic()
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)

        while len(self._items) >= self.max_entries:
            oldest_key = min(self._items, key=lambda key: self._items[key].expires_at)
            self._items.pop(oldest_key, None)
