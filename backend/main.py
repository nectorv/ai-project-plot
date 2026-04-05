from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pipeline import (
    MCPClientPool,
    PipelineState,
    QueryRequest,
    extract_datacommons,
    normalize_and_validate,
    plan_plot,
    render_plot,
    route_task,
)

# ---------------------------------------------------------------------------
# Load .env — walk up from backend/ to find it
# ---------------------------------------------------------------------------
_cwd = Path(__file__).parent.resolve()
_dotenv_path = next(
    (p / ".env" for p in [_cwd, *_cwd.parents] if (p / ".env").exists()),
    None,
)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=True)

# ---------------------------------------------------------------------------
# Lifespan — pool is created once at startup, closed at shutdown
# ---------------------------------------------------------------------------
POOL_SIZE = int(os.getenv("MCP_POOL_SIZE", "3"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Initializing MCP client pool (size={POOL_SIZE})...")
    pool = MCPClientPool(size=POOL_SIZE)
    await pool.initialize()
    app.state.pool = pool
    print(f"[startup] Pool ready — {pool.available} clients available.")
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
async def _run_pipeline(query: str, pool: MCPClientPool):
    state = PipelineState(request=query)

    # --- Step 1: Route ---
    yield _step("Routing task...")
    state = route_task(state)

    # --- Step 2: Extract (borrows a client from the pool) ---
    yield _step(f"Fetching data from Data Commons (pool: {pool.available}/{POOL_SIZE} free)...")
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_endpoint(request: QueryRequest, req: Request):
    pool: MCPClientPool = req.app.state.pool
    return StreamingResponse(
        _run_pipeline(request.q, pool),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health(req: Request):
    pool: MCPClientPool = req.app.state.pool
    return {
        "status": "ok",
        "dc_api_key_set": bool(os.getenv("DC_API_KEY")),
        "pool_size": POOL_SIZE,
        "pool_available": pool.available,
    }
