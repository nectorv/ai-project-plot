from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


class GenericExtractionResult(BaseModel):
    data_type: Literal["timeseries", "table", "categorical", "scalar"] = Field(
        description="Detected result type."
    )
    title: str = Field(description="Short title for the extracted data")
    columns: list[str] = Field(
        default_factory=list,
        description="Column names for row-based results",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Row records for timeseries/table/categorical outputs",
    )
    scalar_value: float | int | str | None = Field(
        default=None,
        description="Scalar result when data_type='scalar'",
    )
    scalar_unit: str | None = Field(default=None)
    notes: str | None = Field(default=None)


class PlotSpec(BaseModel):
    chart_type: Literal["line", "bar", "scatter", "pie", "table"] = Field(
        description="Best chart type for the user's request and available columns."
    )
    x: str | None = Field(default=None, description="Column to use on the x-axis or pie labels")
    y: str | None = Field(default=None, description="Column to use on the y-axis or pie values")
    title: str = Field(description="Readable chart title")
    x_label: str | None = Field(default=None, description="Optional x-axis label")
    y_label: str | None = Field(default=None, description="Optional y-axis label")
    color: str | None = Field(
        default=None,
        description="Optional column to use for color grouping when relevant",
    )
    sort_by_x: bool = Field(
        default=True,
        description="Whether the dataframe should be sorted by x before plotting",
    )


class QueryRequest(BaseModel):
    q: str = Field(description="Natural language data question from the user")


class SSEEvent(BaseModel):
    type: Literal["step", "result", "error"]
    label: str | None = None        # for type="step"
    figure: str | None = None       # for type="result" — Plotly JSON string
    spec: dict | None = None        # for type="result" — PlotSpec dict
    message: str | None = None      # for type="error"


# In-memory pipeline state, passed between steps within a single request.
# normalized_data holds a pandas DataFrame and is never serialized.
class PipelineState:
    def __init__(self, request: str) -> None:
        self.request: str = request

        # route_task
        self.task_type: str | None = None
        self.is_comparison: bool = False
        self.source_plan: str | None = None

        # extract_datacommons
        self.raw_payload: dict | None = None
        self.tool_verification: dict | None = None

        # normalize_and_validate
        self.normalized_data: pd.DataFrame | None = None
        self.data_type: str | None = None
        self.data_profile: dict | None = None
        self.validation: dict | None = None

        # plan_plot
        self.plot_spec: dict | None = None

        # render_plot
        self.figure_json: str | None = None
