"""声明式策略: 加载校验 + 端到端冒烟。"""
from __future__ import annotations

import numpy as np
import pytest
from helpers import make_market

from event_backtest.engine import run
from event_backtest.fees import FeeSchedule
from event_backtest.rules import MarketRules
from event_backtest.strategy import (
    DeclarativeStrategy,
    StrategyError,
    load_strategy,
    parse_strategy,
)


def test_load_volume_breakout():
    spec = load_strategy("volume_breakout")
    assert spec.name == "放量突破"
    assert [r.name for r in spec.factors] == ["relative_volume", "momentum"]
    assert spec.signal is not None and spec.signal.trigger == "enter"
    assert spec.exit.hold_bars == 16
    assert spec.sizing.max_positions == 3


def test_unknown_factor(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("name: t\nfactors:\n  - nope\n", encoding="utf-8")
    with pytest.raises(StrategyError):
        load_strategy(path)


def test_signal_references_disabled_factor(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: t\nfactors:\n  - momentum: {window: 8}\n"
        "signal:\n  all:\n    - {factor: relative_volume, operation: greater_than, value: 2}\n",
        encoding="utf-8")
    with pytest.raises(StrategyError):
        load_strategy(path)


def test_bad_factor_params(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("name: t\nfactors:\n  - momentum: {window: 8, nope: 1}\n", encoding="utf-8")
    with pytest.raises(StrategyError):
        load_strategy(path)


def test_declarative_run_smoke():
    spec = load_strategy("volume_breakout")
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.01, size=(60, 2)), axis=0)
    volume = np.ones((60, 2))
    volume[40:, :] = 5.0
    market = make_market(close, volume=volume, tickers=["T0", "T1"])
    result = run(DeclarativeStrategy(spec), market, initial_cash=100_000,
                 fees=FeeSchedule(), rules=MarketRules())
    assert len(result.nav) == 60
    assert bool(result.nav.notna().all())


def test_parse_strategy_from_dict():
    raw = {"name": "t", "factors": [{"momentum": {"window": 8}}],
           "signal": {"all": [{"factor": "momentum", "operation": "greater_than", "value": 0}]}}
    spec = parse_strategy(raw)
    assert spec.name == "t"
    assert [r.name for r in spec.factors] == ["momentum"]
    with pytest.raises(StrategyError):
        parse_strategy({"factors": []})          # 缺 name
