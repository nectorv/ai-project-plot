from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pipeline import (
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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Project Plot API")

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
    """Format a dict as an SSE data line."""
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
async def _run_pipeline(query: str):
    state = PipelineState(request=query)

    # --- Step 1: Route ---
    yield _step("Routing task...")
    state = route_task(state)

    # --- Step 2: Extract ---
    yield _step("Fetching data from Data Commons (this may take ~15s)...")
    try:
        state = await extract_datacommons(state)
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
        # plan_plot calls llm.invoke() synchronously — offload to a thread
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
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    return StreamingResponse(
        _run_pipeline(request.q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind proxy
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "dc_api_key_set": bool(os.getenv("DC_API_KEY"))}
