from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import openmeteo_requests
import requests_cache
import wbdata
import yfinance as yf
from fredapi import Fred
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langchain_openai import ChatOpenAI
from pytrends.request import TrendReq
from retry_requests import retry

from .models import GenericExtractionResult, PipelineState

_TRANSIENT_ERROR_KEYWORDS = (
    "504", "gateway timeout", "503", "502",
    "target_read_timeout", "connection", "timeout",
)

_SYSTEM_PROMPT = (
    "You are a data assistant with access to multiple data sources. "
    "Choose the best source for each request:\n"
    "- Data Commons tools: global statistics, demographics, health, education, emissions, population.\n"
    "- yahoo_finance_tool: stock prices (OHLCV), company financials, market cap, earnings. "
    "Use for any question about a publicly traded company or stock ticker.\n"
    "- openmeteo_tool: historical and forecast weather/climate data — temperature, precipitation, wind. "
    "Use for any meteorological or climate question tied to a specific location and date range.\n"
    "- google_trends_tool: relative search interest over time for keywords. "
    "Use for questions about popularity, search trends, or public interest in topics.\n"
    "- world_bank_tool: World Bank development indicators — GDP, poverty, health, education, country-level economic data.\n"
    "- fred_tool: US macroeconomic time series — interest rates, inflation, GDP, unemployment, money supply.\n"
    "- eurostat_tool: European Union statistical data — EU member state comparisons, Eurostat datasets.\n\n"
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
                        f"[retry {attempt}/{max_attempts}] Transient error: "
                        f"{str(exc)[:120]}. Retrying in {wait:.0f}s..."
                    )
                    await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Data extraction failed after {max_attempts} attempts. Last error: {last_exc}"
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


def _assert_any_tool_called(response: dict, expected_tool_names: set[str] | None = None) -> list[str]:
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
            "No data source tool call detected in the agent trace. "
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


# ---------------------------------------------------------------------------
# LangChain custom tools for data sources not covered by MCP
# ---------------------------------------------------------------------------

@tool
def yahoo_finance_tool(ticker: str, period: str = "1y", interval: str = "1mo") -> str:
    """
    Fetch historical OHLCV price data for a stock ticker.
    Args:
        ticker: Stock symbol, e.g. 'AAPL', 'TSLA', 'MSFT'.
        period: History length — '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.
        interval: Bar size — '1d', '1wk', '1mo'.
    Returns JSON with keys: ticker, period, interval, rows (list of dicts with date/open/high/low/close/volume).
    """
    try:
        tk = yf.Ticker(ticker.upper())
        hist = tk.history(period=period, interval=interval)
        if hist.empty:
            return json.dumps({"error": f"No data returned for ticker {ticker}"})
        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else "Datetime"
        hist[date_col] = hist[date_col].astype(str).str[:10]
        rows = (
            hist[[date_col, "Open", "High", "Low", "Close", "Volume"]]
            .rename(columns={date_col: "date"})
            .to_dict(orient="records")
        )
        return json.dumps({"ticker": ticker.upper(), "period": period, "interval": interval, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def openmeteo_tool(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: str = "temperature_2m_max,temperature_2m_min,precipitation_sum",
) -> str:
    """
    Fetch historical daily weather data from Open-Meteo (no API key required).
    Args:
        latitude: Decimal latitude of the location.
        longitude: Decimal longitude of the location.
        start_date: ISO date string, e.g. '2020-01-01'.
        end_date: ISO date string, e.g. '2023-12-31'.
        variables: Comma-separated Open-Meteo daily variable names.
    Returns JSON with keys: latitude, longitude, rows (list of dicts with date + variable columns).
    """
    try:
        cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
        om = openmeteo_requests.Client(session=retry_session)

        var_list = [v.strip() for v in variables.split(",")]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": var_list,
        }
        responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        r = responses[0]
        daily = r.Daily()
        n = int((daily.TimeEnd() - daily.Time()) / daily.Interval())
        dates = [
            datetime.fromtimestamp(daily.Time() + i * daily.Interval()).strftime("%Y-%m-%d")
            for i in range(n)
        ]
        rows = []
        for i, date in enumerate(dates):
            row = {"date": date}
            for j, var in enumerate(var_list):
                val = daily.Variables(j).ValuesAsNumpy()[i]
                row[var] = None if val != val else round(float(val), 4)  # NaN → None
            rows.append(row)
        return json.dumps({"latitude": latitude, "longitude": longitude, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def google_trends_tool(keywords: str, timeframe: str = "today 5-y", geo: str = "") -> str:
    """
    Fetch relative Google search interest over time via pytrends (no API key required).
    Args:
        keywords: Comma-separated search terms, e.g. 'Python,JavaScript,Rust'. Max 5.
        timeframe: pytrends timeframe string, e.g. 'today 5-y', '2020-01-01 2024-01-01', 'today 12-m'.
        geo: ISO 3166-1 alpha-2 country code for regional filtering, e.g. 'US'. Empty for worldwide.
    Returns JSON with keys: keywords, timeframe, geo, rows (list of dicts with date + keyword columns).
    """
    try:
        kw_list = [k.strip() for k in keywords.split(",")][:5]
        pt = TrendReq(hl="en-US", tz=360)
        pt.build_payload(kw_list, timeframe=timeframe, geo=geo)
        df = pt.interest_over_time()
        if df.empty:
            return json.dumps({"error": "No trend data returned."})
        df = df.reset_index()
        df["date"] = df["date"].astype(str).str[:10]
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        rows = df.to_dict(orient="records")
        return json.dumps({"keywords": kw_list, "timeframe": timeframe, "geo": geo, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def world_bank_tool(indicator: str, countries: str, start_year: int = 2000, end_year: int = 2024) -> str:
    """
    Fetch World Bank development indicators for one or more countries.
    Args:
        indicator: World Bank indicator code, e.g. 'NY.GDP.PCAP.CD' (GDP per capita), 'SP.POP.TOTL' (population).
        countries: Comma-separated ISO 3-letter country codes, e.g. 'FRA,DEU,USA'.
        start_year: First year of data (inclusive).
        end_year: Last year of data (inclusive).
    Returns JSON with keys: indicator, rows (list of dicts with country/date/value).
    """
    try:
        country_list = [c.strip() for c in countries.split(",")]
        data = wbdata.get_dataframe(
            {indicator: "value"},
            country=country_list,
            date=(str(start_year), str(end_year)),
        )
        if data is None or data.empty:
            return json.dumps({"error": f"No World Bank data for indicator {indicator}"})
        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]
        # wbdata returns a MultiIndex (country, date); after reset_index columns are 'country', 'date', 'value'
        rows = data.dropna(subset=["value"]).to_dict(orient="records")
        return json.dumps({"indicator": indicator, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def fred_tool(series_id: str, start_date: str = "2000-01-01", end_date: str = "") -> str:
    """
    Fetch a US macroeconomic time series from FRED (Federal Reserve Economic Data).
    Args:
        series_id: FRED series identifier, e.g. 'UNRATE' (unemployment), 'CPIAUCSL' (CPI), 'DGS10' (10-yr Treasury).
        start_date: ISO date string, e.g. '2000-01-01'.
        end_date: ISO date string, e.g. '2024-12-31'. Defaults to today if empty.
    Returns JSON with keys: series_id, rows (list of dicts with date/value).
    """
    try:
        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key or api_key == "your_fred_api_key_here":
            return json.dumps({"error": "FRED_API_KEY not configured. Add it to .env."})
        fred = Fred(api_key=api_key)
        kwargs: dict[str, Any] = {"observation_start": start_date}
        if end_date:
            kwargs["observation_end"] = end_date
        series = fred.get_series(series_id, **kwargs)
        if series is None or series.empty:
            return json.dumps({"error": f"No FRED data for series {series_id}"})
        rows = [
            {"date": str(date.date()), "value": None if val != val else round(float(val), 6)}
            for date, val in series.items()
        ]
        return json.dumps({"series_id": series_id, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def eurostat_tool(dataset_code: str, filters: str = "") -> str:
    """
    Fetch a dataset from Eurostat (European Union statistics).
    Args:
        dataset_code: Eurostat dataset code, e.g. 'une_rt_a' (unemployment rate annual), 'nama_10_gdp' (GDP).
        filters: Optional JSON string of filter parameters, e.g. '{"geo": ["DE", "FR"], "unit": ["PC_ACT"]}'. Empty for no filter.
    Returns JSON with keys: dataset_code, rows (list of dicts).
    """
    try:
        import eurostat  # local import — only needed when this tool is called
        filter_pars = json.loads(filters) if filters.strip() else {}
        df = eurostat.get_data_df(dataset_code, filter_pars=filter_pars)
        if df is None or df.empty:
            return json.dumps({"error": f"No Eurostat data for dataset {dataset_code}"})
        # Melt wide year columns to long format for easier downstream use
        id_cols = [c for c in df.columns if not str(c).isdigit() and not str(c).startswith("19") and not str(c).startswith("20")]
        year_cols = [c for c in df.columns if c not in id_cols]
        if year_cols:
            df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="time", value_name="value")
        df = df.dropna(subset=["value"])
        rows = df.head(500).to_dict(orient="records")  # cap at 500 rows
        return json.dumps({"dataset_code": dataset_code, "rows": rows})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


_LANGCHAIN_TOOLS = [
    yahoo_finance_tool,
    openmeteo_tool,
    google_trends_tool,
    world_bank_tool,
    fred_tool,
    eurostat_tool,
]


async def extract_datacommons(state: PipelineState, tools: list[Any]) -> PipelineState:
    """
    Run the data extraction agent using MCP tools (from the pool) plus custom LangChain tools.
    The agent selects the best source(s) for the user's request automatically.
    """
    expected_tool_names = {
        (getattr(t, "name", "") or "").strip()
        for t in tools
        if (getattr(t, "name", "") or "").strip()
    }
    # Include custom LangChain tool names in the expected set
    expected_tool_names |= {t.name for t in _LANGCHAIN_TOOLS}

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    all_tools = list(tools) + _LANGCHAIN_TOOLS
    agent = create_agent(
        llm,
        tools=all_tools,
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
        f"User request: {state.request}\n"
        f"{comparison_hint}"
    )

    response = await _ainvoke_with_retry(agent, {"messages": [("user", request_with_task)]})
    called_tools = _assert_any_tool_called(response, expected_tool_names)

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
            called_tools = _assert_any_tool_called(retry_response, expected_tool_names)
            response = retry_response

    state.raw_payload = response
    state.tool_verification = {
        "tool_called": True,
        "called_tools": called_tools,
        "expected_tools": sorted(expected_tool_names),
    }
    return state
