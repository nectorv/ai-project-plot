from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI

from .models import GenericExtractionResult, PipelineState

_TRANSIENT_ERROR_KEYWORDS = (
    "504", "gateway timeout", "503", "502",
    "target_read_timeout", "connection", "timeout",
)

_SYSTEM_PROMPT = (
    "You are a data assistant using Data Commons tools. "
    "Extract data for the user's request and return ONLY the required structured schema. "
    "If user asks for trends over time, return data_type='timeseries' with rows containing time/value and optional columns. "
    "If user asks for comparisons across entities/categories, include ALL requested entities in rows and include an entity/place column. "
    "If user asks for one value, return data_type='scalar' and set scalar_value/scalar_unit. "
    "If user asks for a dataset/listing, return data_type='table' with rows/columns. "
    "Keep missing values as null and do not include commentary outside schema."
)


def _is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(keyword in msg for keyword in _TRANSIENT_ERROR_KEYWORDS)


async def _ainvoke_with_retry(
    agent, inputs: dict, max_attempts: int = 3, base_delay: float = 5.0
) -> dict:
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await agent.ainvoke(inputs)
        except (ToolException, Exception) as exc:
            if _is_transient_error(exc):
                last_exc = exc
                if attempt < max_attempts:
                    wait = base_delay * (2 ** (attempt - 1))
                    print(
                        f"[retry {attempt}/{max_attempts}] Transient DC error: "
                        f"{str(exc)[:120]}. Retrying in {wait:.0f}s..."
                    )
                    await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Data Commons API failed after {max_attempts} attempts. Last error: {last_exc}"
    ) from last_exc


def _extract_tool_name_from_call(call: Any) -> str:
    if isinstance(call, dict):
        name = call.get("name")
        if not name and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
        return (name or "").strip()
    return (getattr(call, "name", "") or "").strip()


def _collect_called_tool_names(response: dict[str, Any]) -> set[str]:
    messages = response.get("messages", [])
    called: set[str] = set()

    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                name = _extract_tool_name_from_call(call)
                if name:
                    called.add(name)
            for call in (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", []) or []:
                name = _extract_tool_name_from_call(call)
                if name:
                    called.add(name)
        elif isinstance(msg, ToolMessage):
            name = (getattr(msg, "name", "") or "").strip()
            if name:
                called.add(name)
        elif isinstance(msg, dict):
            for call in msg.get("tool_calls", []) or []:
                name = _extract_tool_name_from_call(call)
                if name:
                    called.add(name)
            role = (msg.get("role") or msg.get("type") or "").lower()
            if role in {"tool", "toolmessage"}:
                name = (msg.get("name") or msg.get("tool_name") or "").strip()
                if name:
                    called.add(name)

    for step in response.get("intermediate_steps", []) or []:
        try:
            action = step[0]
        except Exception:
            action = None
        name = (getattr(action, "tool", "") or "").strip() if action is not None else ""
        if name:
            called.add(name)

    return called


def _assert_dc_tool_called(response: dict, expected_tool_names: set[str] | None = None) -> list[str]:
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected agent response format; cannot verify tool usage.")

    called = _collect_called_tool_names(response)

    if expected_tool_names:
        expected = {n.strip() for n in expected_tool_names if n and n.strip()}
        used_expected = bool(called & expected)
    else:
        used_expected = bool(called)

    if not used_expected:
        raise RuntimeError(
            "No expected Data Commons tool call detected in the agent trace. "
            f"Detected tool names: {sorted(called)}"
        )
    return sorted(called)


def _count_distinct_entities(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    entity_col = next(
        (c for c in ["place", "entity", "country", "name", "location"] if c in rows[0]),
        None,
    )
    if not entity_col:
        return 0
    values = {
        str(row.get(entity_col)).strip()
        for row in rows
        if row.get(entity_col) is not None
    }
    return len({v for v in values if v and v.lower() != "none"})


async def extract_datacommons(state: PipelineState, tools: list[Any]) -> PipelineState:
    """
    Run the Data Commons extraction agent using a pre-fetched list of tools.
    Tools (and the underlying MCP subprocess) are provided by MCPClientPool,
    so no subprocess is spawned here.
    """
    expected_tool_names = {
        (getattr(t, "name", "") or "").strip()
        for t in tools
        if (getattr(t, "name", "") or "").strip()
    }

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=GenericExtractionResult,
    )

    comparison_hint = (
        "This is a comparative request. Return data for each compared entity "
        "(do not collapse to a single entity)."
        if state.is_comparison
        else ""
    )

    request_with_task = (
        f"Task type hint: {state.task_type}. "
        f"User request: {state.request}\n"
        f"{comparison_hint}"
    )

    response = await _ainvoke_with_retry(agent, {"messages": [("user", request_with_task)]})
    called_tools = _assert_dc_tool_called(response, expected_tool_names)

    if state.is_comparison:
        structured = response["structured_response"]
        rows = structured.rows if hasattr(structured, "rows") else []
        if structured.data_type in {"timeseries", "categorical"} and _count_distinct_entities(rows) < 2:
            retry_prompt = (
                f"User request: {state.request}\n"
                "Mandatory: return at least two distinct entities in rows with an explicit "
                "place/entity column. If one entity has missing values, still include rows "
                "for available years/categories for all requested entities."
            )
            retry_response = await _ainvoke_with_retry(agent, {"messages": [("user", retry_prompt)]})
            called_tools = _assert_dc_tool_called(retry_response, expected_tool_names)
            response = retry_response

    state.raw_payload = response
    state.tool_verification = {
        "dc_tool_called": True,
        "called_tools": called_tools,
        "expected_tools": sorted(expected_tool_names),
    }
    return state
