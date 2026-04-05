"""
Unit tests for MCPClientPool.

All tests mock _create_client so no real subprocess is spawned.
We test the pool mechanics: initialization, acquire/release,
concurrency bounding, and broken-client replacement.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.mcp_pool import MCPClientPool


def _make_client():
    """Return a (mock_client, mock_tools) pair."""
    return MagicMock(), [MagicMock()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _pool_of(size: int) -> MCPClientPool:
    pool = MCPClientPool(size=size)
    with patch.object(pool, "_create_client", new_callable=AsyncMock) as m:
        m.side_effect = [_make_client() for _ in range(size)]
        await pool.initialize()
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pool_initializes_correct_size():
    pool = await _pool_of(3)
    assert pool.available == 3


@pytest.mark.asyncio
async def test_acquire_reduces_available_count():
    pool = await _pool_of(2)
    async with pool.acquire():
        assert pool.available == 1


@pytest.mark.asyncio
async def test_release_restores_available_count():
    pool = await _pool_of(2)
    async with pool.acquire():
        pass  # released on exit
    assert pool.available == 2


@pytest.mark.asyncio
async def test_acquire_yields_client_and_tools():
    pool = await _pool_of(1)
    async with pool.acquire() as (client, tools):
        assert client is not None
        assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_pool_blocks_when_exhausted():
    """
    With a pool of size 1, a second concurrent acquire must wait
    until the first one is released.
    """
    pool = await _pool_of(1)
    order: list[str] = []

    async def task_a():
        async with pool.acquire():
            order.append("a_start")
            await asyncio.sleep(0.05)
            order.append("a_end")

    async def task_b():
        # Give task_a time to grab the only client first
        await asyncio.sleep(0.01)
        async with pool.acquire():
            order.append("b_start")

    await asyncio.gather(task_a(), task_b())
    assert order == ["a_start", "a_end", "b_start"], (
        "task_b must not start until task_a releases the client"
    )


@pytest.mark.asyncio
async def test_concurrent_requests_bounded_by_pool_size():
    """
    With pool size 2, at most 2 tasks should hold a client at the same time.
    """
    pool = await _pool_of(2)
    max_concurrent = 0
    current = 0

    async def task():
        nonlocal max_concurrent, current
        async with pool.acquire():
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.02)
            current -= 1

    await asyncio.gather(*[task() for _ in range(5)])
    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_broken_client_is_replaced():
    """
    When acquire() context raises an exception, the pool should replace
    the broken client with a fresh one so pool size is preserved.
    """
    pool = await _pool_of(2)

    fresh_client = _make_client()
    with patch.object(pool, "_create_client", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = fresh_client
        try:
            async with pool.acquire():
                raise RuntimeError("simulated tool failure")
        except RuntimeError:
            pass
        # Give the background replacement a moment to run
        await asyncio.sleep(0.05)

    assert pool.available == 2
    assert mock_create.call_count == 1  # one replacement was spawned


@pytest.mark.asyncio
async def test_close_drains_pool():
    pool = await _pool_of(3)
    await pool.close()
    assert pool.available == 0
