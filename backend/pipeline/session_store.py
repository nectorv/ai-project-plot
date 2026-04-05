from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

# Keep the last N turns per session (~10 exchanges) and cap total sessions
# to avoid unbounded memory growth.
_MAX_TURNS = 20
_MAX_SESSIONS = 500


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str


class ConversationStore:
    """
    Thread-safe in-memory store mapping session_id → list of conversation turns.

    Each turn is either a user query or the title of the chart the assistant
    produced. This is enough context for the resolver to rewrite follow-up
    queries without storing full chart data.
    """

    def __init__(
        self,
        max_turns: int = _MAX_TURNS,
        max_sessions: int = _MAX_SESSIONS,
    ) -> None:
        self._store: dict[str, list[Turn]] = {}
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def add_turn(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> None:
        with self._lock:
            if session_id not in self._store:
                if len(self._store) >= self._max_sessions:
                    # Evict the oldest session (insertion-order dict)
                    oldest = next(iter(self._store))
                    del self._store[oldest]
                self._store[session_id] = []

            self._store[session_id].append(Turn(role=role, content=content))

            # Trim oldest turns if over the per-session limit
            if len(self._store[session_id]) > self._max_turns:
                self._store[session_id] = self._store[session_id][-self._max_turns :]

    def get_history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._store)
