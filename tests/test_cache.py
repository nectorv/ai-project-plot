"""
Unit tests for QueryCache.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.cache import CachedResult, QueryCache, _make_key


def _result(title: str = "test") -> CachedResult:
    return CachedResult(
        figure_json='{"data": []}',
        plot_spec={"chart_type": "line"},
        data_profile={"title": title},
    )


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------
def test_key_is_case_insensitive():
    assert _make_key("GDP in France") == _make_key("gdp in france")

def test_key_strips_whitespace():
    assert _make_key("  GDP in France  ") == _make_key("GDP in France")

def test_different_queries_have_different_keys():
    assert _make_key("GDP in France") != _make_key("GDP in Germany")


# ---------------------------------------------------------------------------
# Basic get/set
# ---------------------------------------------------------------------------
def test_miss_returns_none():
    cache = QueryCache()
    assert cache.get("unknown query") is None

def test_set_then_get_returns_result():
    cache = QueryCache()
    cache.set("gdp in france", _result("GDP France"))
    hit = cache.get("gdp in france")
    assert hit is not None
    assert hit.data_profile["title"] == "GDP France"

def test_get_is_case_insensitive():
    cache = QueryCache()
    cache.set("GDP in France", _result())
    assert cache.get("gdp in france") is not None
    assert cache.get("  GDP IN FRANCE  ") is not None

def test_size_increases_on_set():
    cache = QueryCache()
    assert cache.size == 0
    cache.set("query one", _result())
    assert cache.size == 1
    cache.set("query two", _result())
    assert cache.size == 2

def test_duplicate_set_does_not_increase_size():
    cache = QueryCache()
    cache.set("same query", _result("v1"))
    cache.set("same query", _result("v2"))
    assert cache.size == 1
    assert cache.get("same query").data_profile["title"] == "v2"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------
def test_lru_evicts_when_full():
    cache = QueryCache(maxsize=2, ttl=60)
    cache.set("query a", _result("A"))
    cache.set("query b", _result("B"))
    cache.set("query c", _result("C"))   # should evict oldest (A)
    assert cache.size == 2
    assert cache.get("query a") is None  # evicted
    assert cache.get("query b") is not None
    assert cache.get("query c") is not None


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------
def test_ttl_expires_entry():
    cache = QueryCache(maxsize=10, ttl=1)  # 1 second TTL
    cache.set("short lived query", _result())
    assert cache.get("short lived query") is not None
    time.sleep(1.1)
    assert cache.get("short lived query") is None  # expired
