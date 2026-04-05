from .models import PipelineState

_COMPARISON_KEYWORDS = [" vs ", " versus ", "compared to", "compare", "comparison", "against"]


def route_task(state: PipelineState) -> PipelineState:
    req = state.request.lower()
    state.is_comparison = any(word in req for word in _COMPARISON_KEYWORDS)
    return state
