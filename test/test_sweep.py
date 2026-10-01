"""参数扫描: 点路径打补丁、网格搜索选优、作为 walk-forward 的 select。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from helpers import make_market

from event_backtest.config import MarketConfig
from event_backtest.strategy import parse_strategy
from event_backtest.sweep import expand_grid, make_grid_select, search, set_path
from event_backtest.walkforward import run_walk_forward

_RAW = {"name": "t", "factors": [{"momentum": {"window": 4}}],
        "signal": {"all": [{"factor": "momentum", "operation": "greater_than", "value": 0.0}]},
        "exit": {"hold_bars": 5}}
_GRID = {"signal.all[0].value": [0.0, 0.01, 0.05]}


def _market(days: int = 120):
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.01, size=(days, 1)), axis=0)
    volume = np.ones((days, 1))
    volume[::10] *= 5.0
    market = make_market(close, volume=volume, tickers=["AAA"])
    return replace(market, ts=pd.date_range("2024-01-01", periods=days, freq="D").to_numpy())


def test_set_path_and_expand_grid():
    node = {"factors": [{"momentum": {"window": 8}}], "exit": {"hold_bars": 4}}
    set_path(node, "factors[0].momentum.window", 16)
    set_path(node, "exit.hold_bars", 8)
    assert node["factors"][0]["momentum"]["window"] == 16
    assert node["exit"]["hold_bars"] == 8
    assert len(expand_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6
    assert expand_grid({}) == [{}]          # 空网格也会跑一次(不扫)


def test_search_picks_best_by_metric():
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    outcome = search(_RAW, _GRID, cfg, _market(), metric="sharpe")
    assert len(outcome.table) == 3
    best_row = outcome.table.sort_values("sharpe", ascending=False).iloc[0]
    assert outcome.best_params == {"signal.all[0].value": best_row["signal.all[0].value"]}


def test_grid_select_used_in_walk_forward():
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    spec = parse_strategy({**_RAW, "walk_forward": {"folds": 2, "train_days": 40,
                                                    "test_days": 20, "embargo_days": 3}})
    recorded: list[dict[str, object]] = []
    select = make_grid_select(_RAW, _GRID, cfg, metric="sharpe", record=recorded)
    result = run_walk_forward(cfg, spec, market=_market(150), select=select)
    assert len(result.folds) == 2
    assert len(recorded) == 2               # 每折训练段选一次
    for params in recorded:
        assert params["signal.all[0].value"] in _GRID["signal.all[0].value"]
