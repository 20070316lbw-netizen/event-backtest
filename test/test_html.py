"""HTML 报告: 自包含、数据可解析, 三套模板都能出。"""
from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np
from helpers import make_market

from event_backtest import (
    MarketConfig,
    Strategy,
    load_strategy,
    plot_comparison_html,
    plot_tearsheet_html,
    plot_walk_forward_html,
    run,
)
from event_backtest.fees import FeeSchedule
from event_backtest.rules import MarketRules
from event_backtest.strategy import WalkForward
from event_backtest.walkforward import run_walk_forward

_PAYLOAD = re.compile(
    r'<script id="report-data" type="application/json">(.*?)</script>', re.S)


class _BuyOnce(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 3)


def _result():
    close = np.array([[10.0], [10.2], [10.5], [10.1], [10.8], [11.0]])
    market = make_market(close, tickers=["AAA"])
    return run(_BuyOnce(), market, initial_cash=1000.0,
               fees=FeeSchedule(), rules=MarketRules())


def _walk_forward():
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.01, size=(120, 1)), axis=0)
    market = make_market(close, tickers=["AAA"])
    spec = replace(load_strategy("volume_breakout"),
                   walk_forward=WalkForward(folds=2, start=None, train_days=20,
                                            test_days=10, embargo_days=3))
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    return run_walk_forward(cfg, spec, market=market)


def _payload(page: str) -> dict:
    match = _PAYLOAD.search(page)
    assert match is not None
    return json.loads(match.group(1))


def _assert_self_contained(page: str) -> None:
    assert page.startswith("<!doctype html>")
    # SVG 的命名空间不是网络请求; 除此之外不许有任何外链, 否则断网打不开
    stripped = page.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in stripped and "https://" not in stripped
    assert "<style>" in page and "<script>" in page


def test_tearsheet_html_self_contained_and_parseable(tmp_path):
    report = plot_tearsheet_html(_result(), market="us", title="单元测试")
    assert report.kind == "tearsheet"
    _assert_self_contained(report.html)

    data = _payload(report.html)
    assert data["title"] == "单元测试"
    assert data["kind"] == "tearsheet"
    assert len(data["nav"]["labels"]) == 6
    assert data["nav"]["series"][0]["name"] == "策略"
    assert data["drawdown"]["series"][0]["values"][-1] == 0.0
    # NaN 的指标(这段行情没有下行波动, Sortino 是 NaN)不进卡片, 也不能变成 JSON 里的 NaN
    assert data["kpis"] and all(kpi["value"] is not None for kpi in data["kpis"])
    assert any(section["title"] == "执行" for section in data["sections"])
    assert any(row[0] == "委托笔数" for section in data["sections"]
               for row in section["rows"])

    path = report.save(tmp_path / "deep" / "tearsheet.html")
    assert path.exists() and path.stat().st_size > 8000


def test_comparison_html_has_one_series_per_scenario():
    result = _result()
    report = plot_comparison_html({"a": result, "b": result}, market="us")
    assert report.kind == "comparison"
    _assert_self_contained(report.html)

    data = _payload(report.html)
    assert [s["name"] for s in data["nav"]["series"]] == ["a", "b"]
    assert data["table"]["sortable"] is True
    assert data["table"]["sortKey"] == "total_return"
    assert {row["name"] for row in data["table"]["rows"]} == {"a", "b"}


def test_walk_forward_html_has_folds():
    result = _walk_forward()
    report = plot_walk_forward_html(result)
    assert report.kind == "walk_forward"
    _assert_self_contained(report.html)

    data = _payload(report.html)
    assert len(data["folds"]["rows"]) == len(result.folds) == 2
    assert data["fold_bars"]["labels"] == ["折0", "折1"]
    assert len(data["nav"]["labels"]) == len(result.oos_nav)
    assert data["kpis"]


def test_html_payload_never_contains_nan_literals():
    """json.dumps(allow_nan=False) 是硬约束: 页面里的 JSON 不允许 NaN / Infinity。"""
    pages = (plot_tearsheet_html(_result(), market="us").html,
             plot_comparison_html({"a": _result()}, market="us").html,
             plot_walk_forward_html(_walk_forward()).html)
    for page in pages:
        payload = _PAYLOAD.search(page).group(1)
        assert "NaN" not in payload and "Infinity" not in payload
