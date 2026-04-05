from .models import PipelineState


def route_task(state: PipelineState) -> PipelineState:
    req = state.request.lower()

    time_keywords = [
        "over time", "trend", "from", "to", "year", "monthly",
        "quarterly", "timeseries", "time series", "evolution",
    ]
    scalar_keywords = ["latest", "current", "what is", "how much", "single value", "now"]
    categorical_keywords = [
        "by", "compare", "across", "top", "ranking", "rank",
        "which country", "which state", "vs", "versus", "compared to",
    ]
    table_keywords = ["table", "rows", "list all", "dataset", "return data"]

    state.is_comparison = any(
        word in req
        for word in [" vs ", " versus ", "compared to", "compare", "comparison", "against"]
    )

    if any(word in req for word in time_keywords):
        state.task_type = "timeseries"
    elif any(word in req for word in scalar_keywords):
        state.task_type = "scalar"
    elif any(word in req for word in categorical_keywords):
        state.task_type = "categorical"
    elif any(word in req for word in table_keywords):
        state.task_type = "table"
    else:
        state.task_type = "table"

    state.source_plan = "datacommons"
    return state
