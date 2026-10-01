"""结果 API: 相等语义、上下文 meta、Run 容器、save_run 与 run.json 清单。"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from helpers import make_market

from event_backtest import Run, Strategy, load_config, run
from event_backtest.fees import FeeSchedule
from event_backtest.results import jsonable, save_run
from event_backtest.rules import MarketRules


class _BuyOnce(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 3)


def _result(cash: float = 1000.0):
    close = np.array([[10.0], [10.2], [10.5], [10.1], [10.8], [11.0]])
    market = make_market(close, tickers=["AAA"])
    return run(_BuyOnce(), market, initial_cash=cash, fees=FeeSchedule(), rules=MarketRules())


def test_result_equality_is_identity_and_hashable():
    first, second = _result(), _result()
    # 以前这里会抛 "The truth value of a Series is ambiguous"
    assert first == first and first != second
    assert len({first, second}) == 2
    assert make_market([[1.0], [2.0]]) != make_market([[1.0], [2.0]])


def test_run_carries_context_and_metrics():
    result = _result()
    result.meta.update({"market": "us", "benchmark": "AAA"})
    run_obj = Run(result, strategy="buy_once")
    assert run_obj.market == "us" and run_obj.benchmark == "AAA"
    assert run_obj.summary()["total_return"] > 0
    assert "execution_rate" in run_obj.execution()
    assert {title for title, _ in run_obj.sections()} >= {"收益", "执行"}
    assert run_obj.values()["total_return"] == pytest.approx(run_obj.summary()["total_return"])
    assert run_obj.benchmark_curve() is not None       # 基准就在这份行情里


def test_config_wins_over_meta():
    result = _result()
    result.meta.update({"market": "cn", "benchmark": None})
    pair = Run(result, config=replace(load_config("us"), benchmark="AAA"))
    assert pair.market == "us" and pair.benchmark == "AAA"


def test_study_requires_declarative_spec():
    with pytest.raises(ValueError, match="事件研究"):
        Run(_result()).study()


def test_save_run_writes_files_and_manifest(tmp_path):
    result = _result()
    result.meta.update({"market": "us"})
    out = save_run(tmp_path, result, market="us", strategy="buy_once")
    for name in ("nav.parquet", "fills.parquet", "orders.parquet", "trades.parquet",
                 "performance.csv", "trade_stats.csv", "execution.csv", "run.json"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert manifest["market"] == "us" and manifest["strategy"] == "buy_once"
    assert manifest["bars"] == 6
    assert manifest["metrics"]["total_return"] is not None
    assert "nav.parquet" in manifest["files"]
    assert isinstance(manifest["package_version"], str) and manifest["package_version"]


def test_run_save_can_add_html(tmp_path):
    result = _result()
    result.meta.update({"market": "us"})
    out = Run(result).save(tmp_path / "with_html", html=True, title="t")
    assert (out / "tearsheet.html").exists() and (out / "run.json").exists()


def test_jsonable_handles_numpy_nan_and_nested():
    payload = jsonable({"a": np.float64(1.5), "b": float("nan"), "c": np.int64(3),
                        "d": (1, np.bool_(True)), "e": pd.Timestamp("2024-01-02"),
                        "f": {"g": [np.float64(2.0)]}})
    assert json.loads(json.dumps(payload)) == {
        "a": 1.5, "b": None, "c": 3, "d": [1, True],
        "e": "2024-01-02T00:00:00", "f": {"g": [2.0]}}
