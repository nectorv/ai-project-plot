from __future__ import annotations

import pandas as pd

from .models import PipelineState

_NON_NUMERIC_COLS = {"time", "date", "period", "place", "entity", "name", "metric", "unit", "source"}


def normalize_and_validate(state: PipelineState) -> PipelineState:
    structured = state.raw_payload["structured_response"]
    payload = structured.model_dump()
    data_type = payload.get("data_type", "table")

    if data_type == "scalar":
        df = pd.DataFrame(
            [
                {
                    "metric": payload.get("title", "value"),
                    "value": payload.get("scalar_value"),
                    "unit": payload.get("scalar_unit"),
                }
            ]
        )
    else:
        rows = payload.get("rows", [])
        if not rows:
            state.validation = {
                "pass": False,
                "reason": f"No rows returned for data_type='{data_type}'.",
            }
            return state

        df = pd.DataFrame(rows)
        for col in payload.get("columns", []):
            if col not in df.columns:
                df[col] = None

    # Cast numeric columns
    for col in df.columns:
        if col.lower() in _NON_NUMERIC_COLS:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.6 * len(df))):
            df[col] = converted

    # Sort timeseries
    if data_type == "timeseries" and "time" in df.columns:
        parsed = pd.to_datetime(df["time"], errors="coerce")
        if parsed.notna().any():
            df = df.assign(_sort=parsed).sort_values("_sort").drop(columns=["_sort"])
        else:
            numeric_time = pd.to_numeric(df["time"], errors="coerce")
            if numeric_time.notna().any():
                df = df.assign(_sort=numeric_time).sort_values("_sort").drop(columns=["_sort"])
            else:
                df = df.sort_values("time")

    df = df.reset_index(drop=True)

    state.data_type = data_type
    state.normalized_data = df
    state.data_profile = {
        "data_type": data_type,
        "title": payload.get("title"),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isnull().sum().to_dict(),
    }
    state.validation = {"pass": True}
    return state
