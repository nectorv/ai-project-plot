from __future__ import annotations

import plotly.express as px

from .models import PlotSpec, PipelineState


def render_plot(state: PipelineState) -> PipelineState:
    df = state.normalized_data.copy()
    plot_spec = PlotSpec.model_validate(state.plot_spec)

    if plot_spec.chart_type == "table":
        # Serialize as records for the frontend to display
        state.figure_json = df.to_json(orient="records")
        return state

    required_cols = {plot_spec.x, plot_spec.y}
    if plot_spec.color and plot_spec.chart_type != "pie":
        required_cols.add(plot_spec.color)

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {sorted(missing)}")

    if plot_spec.sort_by_x and plot_spec.chart_type != "pie":
        df = df.sort_values(plot_spec.x)

    if plot_spec.chart_type == "pie":
        fig = px.pie(df, names=plot_spec.x, values=plot_spec.y, title=plot_spec.title)
    else:
        chart_builders = {"line": px.line, "bar": px.bar, "scatter": px.scatter}
        kwargs = {
            "data_frame": df,
            "x": plot_spec.x,
            "y": plot_spec.y,
            "color": plot_spec.color,
            "title": plot_spec.title,
            "labels": {
                plot_spec.x: plot_spec.x_label or plot_spec.x,
                plot_spec.y: plot_spec.y_label or plot_spec.y,
            },
        }
        if plot_spec.chart_type in {"line", "scatter"}:
            kwargs["markers"] = True

        fig = chart_builders[plot_spec.chart_type](**kwargs)

    fig.update_layout(template="plotly_white")
    state.figure_json = fig.to_json()
    return state
