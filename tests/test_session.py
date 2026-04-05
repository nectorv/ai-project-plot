"""
Unit tests for ConversationStore and resolve_query.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.session_store import ConversationStore, Turn
from pipeline.resolver import resolve_query


# ---------------------------------------------------------------------------
# ConversationStore tests
# ---------------------------------------------------------------------------
def test_store_starts_empty():
    store = ConversationStore()
    assert store.get_history("any") == []
    assert store.session_count == 0


def test_add_and_retrieve_turn():
    store = ConversationStore()
    store.add_turn("s1", "user", "show GDP in France")
    history = store.get_history("s1")
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "show GDP in France"


def test_multiple_turns_preserved_in_order():
    store = ConversationStore()
    store.add_turn("s1", "user", "first query")
    store.add_turn("s1", "assistant", "showed chart A")
    store.add_turn("s1", "user", "second query")
    history = store.get_history("s1")
    assert [t.content for t in history] == ["first query", "showed chart A", "second query"]


def test_clear_removes_session():
    store = ConversationStore()
    store.add_turn("s1", "user", "hello")
    store.clear("s1")
    assert store.get_history("s1") == []
    assert store.session_count == 0


def test_clear_nonexistent_session_is_safe():
    store = ConversationStore()
    store.clear("does_not_exist")  # should not raise


def test_max_turns_trims_oldest():
    store = ConversationStore(max_turns=4)
    for i in range(6):
        store.add_turn("s1", "user", f"query {i}")
    history = store.get_history("s1")
    assert len(history) == 4
    assert history[0].content == "query 2"   # oldest 2 trimmed
    assert history[-1].content == "query 5"


def test_max_sessions_evicts_oldest():
    store = ConversationStore(max_sessions=2)
    store.add_turn("s1", "user", "first")
    store.add_turn("s2", "user", "second")
    store.add_turn("s3", "user", "third")   # should evict s1
    assert store.session_count == 2
    assert store.get_history("s1") == []    # evicted
    assert store.get_history("s2") != []
    assert store.get_history("s3") != []


def test_get_history_returns_copy():
    """Mutating the returned list must not affect the store."""
    store = ConversationStore()
    store.add_turn("s1", "user", "hello")
    history = store.get_history("s1")
    history.clear()
    assert len(store.get_history("s1")) == 1


def test_separate_sessions_are_independent():
    store = ConversationStore()
    store.add_turn("s1", "user", "GDP France")
    store.add_turn("s2", "user", "CO2 USA")
    assert store.get_history("s1")[0].content == "GDP France"
    assert store.get_history("s2")[0].content == "CO2 USA"


# ---------------------------------------------------------------------------
# resolve_query tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolve_returns_query_unchanged_when_no_history():
    result = await resolve_query("show GDP in France", [])
    assert result == "show GDP in France"


@pytest.mark.asyncio
async def test_resolve_calls_llm_when_history_present():
    history = [
        Turn(role="user", content="show GDP in France from 2000 to 2025"),
        Turn(role="assistant", content="Showed chart: GDP Evolution in France (2000-2025)"),
    ]
    mock_response = MagicMock()
    mock_response.content = "show GDP in France and Germany from 2000 to 2025"

    with patch("pipeline.resolver.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(return_value=mock_response)
        result = await resolve_query("now add Germany", history)

    assert result == "show GDP in France and Germany from 2000 to 2025"
    instance.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_falls_back_to_original_on_empty_llm_response():
    history = [Turn(role="user", content="show GDP in France")]
    mock_response = MagicMock()
    mock_response.content = ""   # LLM returned nothing

    with patch("pipeline.resolver.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(return_value=mock_response)
        result = await resolve_query("now add Germany", history)

    assert result == "now add Germany"


@pytest.mark.asyncio
async def test_resolve_only_uses_last_6_turns():
    """Resolver must only pass the last 6 turns to the LLM, not the full history."""
    history = [Turn(role="user", content=f"query {i}") for i in range(10)]
    mock_response = MagicMock()
    mock_response.content = "resolved query"

    with patch("pipeline.resolver.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        instance.ainvoke = AsyncMock(return_value=mock_response)
        await resolve_query("follow-up", history)

    prompt_sent = instance.ainvoke.call_args[0][0]
    # Only queries 4-9 should appear (last 6)
    assert "query 0" not in prompt_sent
    assert "query 4" in prompt_sent
    assert "query 9" in prompt_sent
