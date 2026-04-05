from __future__ import annotations

import asyncio
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

_UVX_PATH = shutil.which("uvx") or str(Path(sys.executable).parent / "uvx")


class MCPClientPool:
    """
    A fixed-size pool of MCP clients (and their pre-fetched tools).

    At startup, `size` subprocesses are spawned and kept alive.
    Each request borrows a (client, tools) pair via `acquire()`,
    uses it, then returns it automatically when done.

    If all clients are busy, the caller waits until one is freed.
    If a client errors during use, a fresh one is spawned to replace it.
    """

    def __init__(self, size: int = 3) -> None:
        self._size = size
        self._queue: asyncio.Queue[tuple[MultiServerMCPClient, list[Any]]] = asyncio.Queue()

    async def initialize(self) -> None:
        """Spawn all pool clients. Called once at server startup."""
        for _ in range(self._size):
            client, tools = await self._create_client()
            await self._queue.put((client, tools))

    async def _create_client(self) -> tuple[MultiServerMCPClient, list[Any]]:
        dc_api_key = os.getenv("DC_API_KEY", "")
        client = MultiServerMCPClient(
            {
                "datacommons": {
                    "command": _UVX_PATH,
                    "args": ["datacommons-mcp", "serve", "stdio"],
                    "transport": "stdio",
                    "env": {**os.environ, "DC_API_KEY": dc_api_key},
                }
            }
        )
        if hasattr(client, "get_tools_sync"):
            tools = client.get_tools_sync()
        else:
            tools = await client.get_tools()
        return client, tools

    @asynccontextmanager
    async def acquire(self):
        """
        Borrow a (client, tools) pair from the pool.

        Usage:
            async with pool.acquire() as (client, tools):
                ...

        The pair is returned automatically on exit.
        If an exception occurs, a fresh client is spawned to replace
        the potentially broken one before it re-enters the pool.
        """
        client, tools = await self._queue.get()
        healthy = True
        try:
            yield client, tools
        except Exception:
            healthy = False
            raise
        finally:
            if healthy:
                await self._queue.put((client, tools))
            else:
                # Replace broken client with a fresh one
                try:
                    new_client, new_tools = await self._create_client()
                    await self._queue.put((new_client, new_tools))
                except Exception as spawn_exc:
                    # Log and accept reduced pool size rather than crashing
                    print(f"[MCPClientPool] Failed to replace broken client: {spawn_exc}")

    @property
    def available(self) -> int:
        """Number of clients currently available in the pool."""
        return self._queue.qsize()

    async def close(self) -> None:
        """Drain the pool and close all clients. Called at server shutdown."""
        while not self._queue.empty():
            client, _ = self._queue.get_nowait()
            for close_method in ("aclose", "close"):
                fn = getattr(client, close_method, None)
                if fn:
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass
                    break
