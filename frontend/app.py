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
st.caption("Ask any data question — get a chart. Powered by Data Commons + GPT-4o.")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Show the evolution of GDP in France vs Germany from 2000 to 2025",
    "Compare CO2 emissions per capita for the USA, China, and India over the last 20 years",
    "What is the current unemployment rate in Spain?",
    "Show life expectancy trends in Japan and Brazil from 1990 to 2020",
    "Compare renewable energy share across EU countries",
]

with st.expander("Example queries", expanded=False):
    for example in EXAMPLES:
        if st.button(example, key=example):
            st.session_state["query_input"] = example

query = st.text_area(
    "Your data question",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. Show the evolution of GDP in France vs Germany from 2000 to 2025",
    height=80,
    key="query_input",
)

run = st.button("Generate chart", type="primary", disabled=not query.strip())

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

    def _render_log():
        progress_log.markdown(
            "\n".join(f"- {s}" for s in steps),
            unsafe_allow_html=False,
        )

    status_box.info("Starting pipeline...")

    try:
        with requests.post(
            f"{BACKEND_URL}/query",
            json={"q": query.strip()},
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue

                # SSE lines look like:  b"data: {...}"
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                if not raw_line.startswith("data: "):
                    continue

                event = json.loads(raw_line[6:])
                event_type = event.get("type")

                if event_type == "step":
                    label = event["label"]
                    steps.append(label)
                    status_box.info(label)
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

                    # Chart or table
                    if spec.get("chart_type") == "table" or not figure_json:
                        import pandas as pd
                        rows = json.loads(figure_json) if figure_json else []
                        table_area.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        fig = pio.from_json(figure_json)
                        chart_area.plotly_chart(fig, use_container_width=True)

                    # Data profile expander
                    with st.expander("Data profile", expanded=False):
                        st.json(profile)

                    with st.expander("Plot spec", expanded=False):
                        st.json(spec)

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
