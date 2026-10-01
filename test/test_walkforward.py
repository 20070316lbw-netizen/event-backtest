"""Walk-forward: 折划分、embargo 隔离、样本外拼接、参数不污染。"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from helpers import make_market

from event_backtest.config import MarketConfig
from event_backtest.figure import plot_walk_forward
from event_backtest.market import slice_dates, slice_market
from event_backtest.strategy import WalkForward, load_strategy
from event_backtest.walkforward import (
    WalkForwardError,
    check_data,
    fold_ranges,
    run_walk_forward,
)


def _market(days: int, tickers=("AAA",)):
    """造一段日线行情, 时间轴用真实日期, 方便断言折边界。"""
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.01, size=(days, len(tickers))), axis=0)
    volume = np.ones((days, len(tickers)))
    volume[::10] *= 5.0
    market = make_market(close, volume=volume, tickers=list(tickers))
    return replace(market, ts=pd.date_range("2024-01-01", periods=days, freq="D").to_numpy())


def _wf(**over):
    values = {"folds": 2, "start": date(2024, 1, 1), "train_days": 20,
              "test_days": 10, "embargo_days": 3, **over}
    return WalkForward(**values)


def test_fold_ranges_are_contiguous_with_embargo():
    market = _market(100)
    wf = _wf()
    ranges = fold_ranges(wf, market)

    assert len(ranges) == 2
    # 第 0 折: 训练 day0~19, 隔离 3 天, 测试 day23~32(2024-01-24 ~ 2024-02-02)
    assert ranges[0].test_days == (pd.Timestamp("2024-01-24"), pd.Timestamp("2024-02-02"))
    # 训练结束到测试开始正好隔 embargo 根 bar(这里是日线)
    assert ranges[0].test_bars[0] - ranges[0].train_bars[1] == wf.embargo_days
    # 各折测试段首尾相接、互不重叠
    assert ranges[0].test_bars[1] == ranges[1].test_bars[0]


def test_check_data_rejects_short_history():
    with pytest.raises(WalkForwardError):
        check_data(_wf(), _market(20), name="t")


def test_slice_market_keeps_security_fields():
    market = _market(30, tickers=("AAA", "BBB"))
    part = slice_market(market, 5, 15)
    assert part.shape == (10, 2)
    assert part.tickers == ("AAA", "BBB")
    assert part.ts[0] == market.ts[5]
    assert part.ts[-1] == market.ts[14]


def test_run_walk_forward_stitches_and_does_not_leak_test_into_train():
    market = _market(120)
    spec = replace(load_strategy("volume_breakout"), walk_forward=_wf())
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)

    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy(fold_spec, train_market):
        """记录选择器能看到的训练段首尾时间。"""
        index = pd.DatetimeIndex(train_market.ts)
        seen.append((index.min(), index.max()))
        return fold_spec

    result = run_walk_forward(cfg, spec, market=market, select=spy)

    assert len(result.folds) == 2
    # 样本外净值 = 各折测试段拼起来
    assert len(result.oos_nav) == sum(len(f.result.nav) for f in result.folds)
    # 每折净值只落在该折测试区间内
    for fold_result in result.folds:
        idx = fold_result.result.nav.index
        assert idx.min() >= fold_result.fold.test_days[0]
        assert idx.max() <= fold_result.fold.test_days[1]
    # 参数不污染: 选择器看到的就是本折训练段(滚动窗口), 严格早于测试段
    for fold_result, (first, latest) in zip(result.folds, seen, strict=True):
        assert first == fold_result.fold.train_days[0]
        assert latest == fold_result.fold.train_days[1]
        assert latest < fold_result.fold.test_days[0]


def test_run_walk_forward_requires_wf_section():
    # momentum_only 没有 walk_forward 段
    cfg = MarketConfig(market="us", db_path="unused.db")
    with pytest.raises(WalkForwardError):
        run_walk_forward(cfg, load_strategy("momentum_only"), market=_market(50))


def test_parse_walk_forward_from_yaml(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: t\n"
        "walk_forward: {folds: 2, start: 2024-01-01, train_days: 20,"
        " test_days: 10, embargo_days: 3}\n"
        "factors:\n  - momentum: {window: 8}\n",
        encoding="utf-8")
    spec = load_strategy(path)
    assert spec.walk_forward is not None
    assert spec.walk_forward.required_days == 20 + 3 + 2 * 10


def test_slice_dates_is_inclusive_and_copies():
    market = _market(30)
    part = slice_dates(market, "2024-01-05", "2024-01-10")
    assert len(part.ts) == 6                        # 1/5 ~ 1/10, 两端含
    assert part.ts[0] == pd.Timestamp("2024-01-05")
    assert part.ts[-1] == pd.Timestamp("2024-01-10")
    # 复制语义: 改切片不影响原行情
    original = float(market.close[4, 0])
    part.close[0, 0] = -1.0
    assert float(market.close[4, 0]) == original


def test_slice_dates_empty_raises():
    with pytest.raises(ValueError):
        slice_dates(_market(10), "2030-01-01", "2030-02-01")


def test_walk_forward_start_defaults_to_first_day(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "name: t\n"
        "walk_forward: {folds: 2, train_days: 20, test_days: 10, embargo_days: 3}\n"
        "factors:\n  - momentum: {window: 8}\n",
        encoding="utf-8")
    spec = load_strategy(path)
    assert spec.walk_forward is not None
    assert spec.walk_forward.start is None
    # start 省略 -> 从数据第一天起算, 第 0 折测试仍是 day 23..32
    assert fold_ranges(spec.walk_forward, _market(100))[0].test_bars[0] == 23


def test_plot_walk_forward():
    market = _market(120)
    spec = replace(load_strategy("volume_breakout"), walk_forward=_wf())
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    fig = plot_walk_forward(run_walk_forward(cfg, spec, market=market))
    assert len(fig.axes) == 2
