from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from cachetools import TTLCache

# Defaults: cache up to 256 distinct queries, each valid for 6 hours.
# Override via env vars MCP_CACHE_MAX / MCP_CACHE_TTL_SECONDS.
_DEFAULT_MAX = 256
_DEFAULT_TTL = 6 * 60 * 60  # 6 hours in seconds


@dataclass
class CachedResult:
    figure_json: str
    plot_spec: dict
    data_profile: dict


class QueryCache:
    """
    Thread-safe LRU+TTL cache for pipeline results.

    Key   — normalized query string (lowercased, stripped)
    Value — CachedResult (figure JSON + plot spec + data profile)

    Entries are evicted automatically when:
    - The cache is full (LRU — least recently used goes first)
    - The TTL expires (stale data is never served)
    """

    def __init__(self, maxsize: int = _DEFAULT_MAX, ttl: int = _DEFAULT_TTL) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> CachedResult | None:
        key = _make_key(query)
        with self._lock:
            return self._cache.get(key)

    def set(self, query: str, result: CachedResult) -> None:
        key = _make_key(query)
        with self._lock:
            self._cache[key] = result

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def maxsize(self) -> int:
        return self._cache.maxsize

    @property
    def ttl(self) -> float:
        return self._cache.ttl


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_key(query: str) -> str:
    """Normalize the query and return a short SHA-256 hex digest."""
    normalized = query.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()
