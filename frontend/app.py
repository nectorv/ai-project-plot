from __future__ import annotations

import json

import plotly.io as pio
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Project Plot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Project Plot")
st.caption("Ask any data question — get a chart. Follow up to refine.")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None          # assigned by backend on first query

if "history" not in st.session_state:
    st.session_state.history = []               # list of {query, resolved, title, figure_json, spec, profile}

# ---------------------------------------------------------------------------
# Sidebar — conversation history + controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Conversation")

    if st.session_state.history:
        for i, turn in enumerate(st.session_state.history):
            with st.expander(f"#{i + 1} — {turn['query'][:50]}...", expanded=False):
                if turn.get("resolved") and turn["resolved"] != turn["query"]:
                    st.caption(f"Resolved to: _{turn['resolved']}_")
                st.write(f"**Chart:** {turn.get('title', '—')}")
    else:
        st.caption("No history yet. Ask a question to start.")

    st.divider()
    if st.button("Clear session", disabled=not st.session_state.session_id):
        sid = st.session_state.session_id
        try:
            requests.delete(f"{BACKEND_URL}/session/{sid}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = None
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Show the evolution of GDP in France vs Germany from 2000 to 2025",
    "Compare CO2 emissions per capita for the USA, China, and India over the last 20 years",
    "What is the current unemployment rate in Spain?",
    "Show life expectancy trends in Japan and Brazil from 1990 to 2020",
]

with st.expander("Example queries", expanded=not st.session_state.history):
    for example in EXAMPLES:
        if st.button(example, key=f"ex_{example}"):
            st.session_state["query_input"] = example
            st.rerun()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
placeholder = (
    "Follow up, e.g. 'Now add Italy' or 'Do the same for CO2'..."
    if st.session_state.history
    else "e.g. Show the evolution of GDP in France vs Germany from 2000 to 2025"
)

query = st.text_area(
    "Your data question",
    value=st.session_state.get("query_input", ""),
    placeholder=placeholder,
    height=80,
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Generate", type="primary", disabled=not query.strip())

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run and query.strip():
    st.divider()

    status_box = st.empty()
    progress_log = st.empty()
    chart_area = st.empty()
    table_area = st.empty()

    steps: list[str] = []
    resolved_query: str = query.strip()
    current_turn: dict = {"query": query.strip(), "resolved": query.strip()}

    def _render_log():
        progress_log.markdown("\n".join(f"- {s}" for s in steps))

    status_box.info("Starting pipeline...")

    try:
        with requests.post(
            f"{BACKEND_URL}/query",
            json={"q": query.strip(), "session_id": st.session_state.session_id},
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                if not raw_line.startswith("data: "):
                    continue

                event = json.loads(raw_line[6:])
                event_type = event.get("type")

                if event_type == "session":
                    st.session_state.session_id = event["session_id"]

                elif event_type == "resolved":
                    resolved_query = event["resolved"]
                    current_turn["resolved"] = resolved_query
                    steps.append(f"Resolved: _{resolved_query}_")
                    _render_log()

                elif event_type == "step":
                    steps.append(event["label"])
                    status_box.info(event["label"])
                    _render_log()

                elif event_type == "error":
                    status_box.error(f"Error: {event['message']}")
                    progress_log.empty()
                    break

                elif event_type == "result":
                    progress_log.empty()
                    status_box.success("Done!")

                    spec = event.get("spec", {})
                    profile = event.get("data_profile", {})
                    figure_json = event.get("figure")
                    chart_title = profile.get("title", resolved_query)

                    # Render chart or table
                    if spec.get("chart_type") == "table" or not figure_json:
                        import pandas as pd
                        rows = json.loads(figure_json) if figure_json else []
                        table_area.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        fig = pio.from_json(figure_json)
                        chart_area.plotly_chart(fig, use_container_width=True)

                    with st.expander("Data profile", expanded=False):
                        st.json(profile)
                    with st.expander("Plot spec", expanded=False):
                        st.json(spec)

                    # Save turn to local history
                    current_turn.update({
                        "title": chart_title,
                        "figure_json": figure_json,
                        "spec": spec,
                        "profile": profile,
                    })
                    st.session_state.history.append(current_turn)
                    # Clear the input box for the next follow-up
                    st.session_state["query_input"] = ""

    except requests.exceptions.ConnectionError:
        status_box.error(
            f"Cannot connect to the backend at `{BACKEND_URL}`. "
            "Make sure the FastAPI server is running:\n\n"
            "```bash\ncd backend && uvicorn main:app --reload\n```"
        )
    except requests.exceptions.Timeout:
        status_box.error("Request timed out. Data Commons or the LLM took too long.")
    except Exception as exc:
        status_box.error(f"Unexpected error: {exc}")
