from __future__ import annotations

import warnings

import pandas as pd
from langchain_openai import ChatOpenAI

from .models import PlotSpec, PipelineState

def _default_plot_spec(state: PipelineState, df: pd.DataFrame) -> PlotSpec:
    data_type = state.data_type or "table"
    request_lower = state.request.lower()
    columns = list(df.columns)
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in columns if c not in numeric_cols]
    pie_requested = any(w in request_lower for w in ["pie", "share", "composition", "breakdown", "portion"])

    if data_type == "timeseries" and "time" in columns and "value" in columns:
        color_col = next(
            (c for c in ["place", "entity", "country", "name", "location"]
             if c in columns and df[c].nunique(dropna=True) > 1),
            None,
        )
        return PlotSpec(chart_type="line", x="time", y="value", color=color_col, title=state.request)

    if data_type == "scalar" and "metric" in columns and "value" in columns:
        return PlotSpec(chart_type="bar", x="metric", y="value", title=state.request)

    if numeric_cols and text_cols:
        chart_type = "pie" if pie_requested and df[text_cols[0]].nunique() <= 12 else "bar"
        return PlotSpec(
            chart_type=chart_type,
            x=text_cols[0],
            y=numeric_cols[0],
            title=state.request,
            sort_by_x=chart_type != "pie",
        )

    if len(numeric_cols) >= 2:
        return PlotSpec(chart_type="scatter", x=numeric_cols[0], y=numeric_cols[1], title=state.request)

    return PlotSpec(chart_type="table", title=state.request)


def plan_plot(state: PipelineState) -> PipelineState:
    df = state.normalized_data
    if df is None or df.empty:
        raise ValueError("No normalized data available for plotting.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    plot_planner = llm.with_structured_output(PlotSpec)
    prompt = f"""
You are a plotting planner.
Choose the most appropriate visualization for the user's request.
Return only a PlotSpec.

Chart guidance:
- Use line for trends over time.
- For multi-entity timeseries (e.g., France vs Germany), keep x=time, y=value, and set color to entity/place.
- Use bar for categorical comparisons.
- Use scatter for relationships between two numeric variables.
- Use pie only for part-to-whole/category share views with a small number of categories.
- Use table when a chart would be unclear.

Task type: {state.data_type}
User request: {state.request}
Available columns: {list(df.columns)}
Data types: {df.dtypes.astype(str).to_dict()}
Sample rows: {df.head(5).to_dict(orient='records')}
""".strip()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*", category=UserWarning)
            plot_spec = plot_planner.invoke(prompt)
    except Exception:
        plot_spec = _default_plot_spec(state, df)

    if plot_spec.chart_type != "table" and (not plot_spec.x or not plot_spec.y):
        plot_spec = _default_plot_spec(state, df)

    # Ensure color is set for comparison timeseries
    if state.is_comparison and state.data_type == "timeseries" and not plot_spec.color:
        for candidate in ["place", "entity", "country", "name", "location"]:
            if candidate in df.columns and df[candidate].nunique(dropna=True) > 1:
                plot_spec.color = candidate
                break

    state.plot_spec = plot_spec.model_dump()
    return state
