from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

CACHE_MAX = int(os.getenv("CACHE_MAX", "256"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", str(6 * 60 * 60)))

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pipeline import (
    CachedResult,
    ConversationStore,
    MCPClientPool,
    PipelineState,
    QueryCache,
    QueryRequest,
    extract_datacommons,
    normalize_and_validate,
    plan_plot,
    render_plot,
    resolve_query,
    route_task,
)

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_cwd = Path(__file__).parent.resolve()
_dotenv_path = next(
    (p / ".env" for p in [_cwd, *_cwd.parents] if (p / ".env").exists()),
    None,
)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=True)

# ---------------------------------------------------------------------------
# Lifespan — pool and conversation store are created once at startup
# ---------------------------------------------------------------------------
POOL_SIZE = int(os.getenv("MCP_POOL_SIZE", "3"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Initializing MCP client pool (size={POOL_SIZE})...")
    pool = MCPClientPool(size=POOL_SIZE)
    await pool.initialize()
    app.state.pool = pool

    app.state.store = ConversationStore()
    app.state.cache = QueryCache(maxsize=CACHE_MAX, ttl=CACHE_TTL)
    print(f"[startup] Pool ready — {pool.available} clients available.")
    print(f"[startup] Query cache: max={CACHE_MAX} entries, ttl={CACHE_TTL}s.")
    yield

    print("[shutdown] Closing MCP client pool...")
    await pool.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Project Plot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"

def _step(label: str) -> str:
    return _sse({"type": "step", "label": label})

def _error(message: str) -> str:
    return _sse({"type": "error", "message": message})

def _result(figure_json: str, plot_spec: dict, data_profile: dict) -> str:
    return _sse({
        "type": "result",
        "figure": figure_json,
        "spec": plot_spec,
        "data_profile": data_profile,
    })


# ---------------------------------------------------------------------------
# Pipeline generator
# ---------------------------------------------------------------------------
async def _run_pipeline(
    original_query: str,
    session_id: str,
    pool: MCPClientPool,
    store: ConversationStore,
    cache: QueryCache,
):
    # Always send session_id first so the client can persist it
    yield _sse({"type": "session", "session_id": session_id})

    # --- Step 0: Resolve query using conversation history ---
    history = store.get_history(session_id)
    if history:
        yield _step("Resolving query with conversation context...")
        resolved_query = await resolve_query(original_query, history)
        if resolved_query != original_query:
            yield _sse({"type": "resolved", "original": original_query, "resolved": resolved_query})
    else:
        resolved_query = original_query

    # --- Cache check ---
    cached = cache.get(resolved_query)
    if cached:
        yield _step("Cache hit — returning stored result instantly.")
        yield _result(cached.figure_json, cached.plot_spec, cached.data_profile)
        chart_title = cached.data_profile.get("title") or resolved_query
        store.add_turn(session_id, "user", original_query)
        store.add_turn(session_id, "assistant", f"Showed chart: {chart_title}")
        return

    state = PipelineState(request=resolved_query)

    # --- Step 1: Route ---
    yield _step("Routing task...")
    state = route_task(state)

    # --- Step 2: Extract ---
    yield _step(f"Fetching data (pool: {pool.available}/{POOL_SIZE} free)...")
    try:
        async with pool.acquire() as (_, tools):
            state = await extract_datacommons(state, tools)
    except Exception as exc:
        yield _error(f"Data extraction failed: {exc}")
        return

    # --- Step 3: Normalize ---
    yield _step("Normalizing and validating data...")
    state = normalize_and_validate(state)
    if not state.validation["pass"]:
        yield _error(f"Validation failed: {state.validation['reason']}")
        return

    # --- Step 4: Plan plot ---
    yield _step("Planning the best chart type...")
    try:
        state = await asyncio.to_thread(plan_plot, state)
    except Exception as exc:
        yield _error(f"Plot planning failed: {exc}")
        return

    # --- Step 5: Render ---
    yield _step("Rendering chart...")
    try:
        state = render_plot(state)
    except Exception as exc:
        yield _error(f"Render failed: {exc}")
        return

    yield _result(state.figure_json, state.plot_spec, state.data_profile)

    # Store result in cache for future identical queries
    cache.set(resolved_query, CachedResult(
        figure_json=state.figure_json,
        plot_spec=state.plot_spec,
        data_profile=state.data_profile,
    ))

    # Save the exchange to history so future queries can reference it
    chart_title = (state.data_profile or {}).get("title") or resolved_query
    store.add_turn(session_id, "user", original_query)
    store.add_turn(session_id, "assistant", f"Showed chart: {chart_title}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_endpoint(request: QueryRequest, req: Request):
    pool: MCPClientPool = req.app.state.pool
    store: ConversationStore = req.app.state.store
    cache: QueryCache = req.app.state.cache

    session_id = request.session_id or str(uuid.uuid4())

    return StreamingResponse(
        _run_pipeline(request.q, session_id, pool, store, cache),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str, req: Request):
    req.app.state.store.clear(session_id)
    return {"cleared": session_id}


@app.get("/health")
async def health(req: Request):
    pool: MCPClientPool = req.app.state.pool
    store: ConversationStore = req.app.state.store
    cache: QueryCache = req.app.state.cache
    return {
        "status": "ok",
        "dc_api_key_set": bool(os.getenv("DC_API_KEY")),
        "pool_size": POOL_SIZE,
        "pool_available": pool.available,
        "active_sessions": store.session_count,
        "cache_entries": cache.size,
        "cache_max": cache.maxsize,
        "cache_ttl_seconds": cache.ttl,
    }
