"""组合指标: 日频聚合、按市场年化、基准对比与 beta。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from event_backtest.metrics import (
    daily_nav,
    periods_per_year,
    summarize,
)


def test_daily_nav_collapses_intraday():
    index = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 15:00", "2024-01-03 15:00"])
    nav = pd.Series([100.0, 110.0, 121.0], index=index)
    daily = daily_nav(nav)
    assert list(daily.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert list(daily) == [110.0, 121.0]


def test_periods_per_year_by_market():
    assert periods_per_year("cn") == 242
    assert periods_per_year("us") == 252


def test_summarize_annualizes_on_daily_nav():
    # 一年左右的日频净值涨 10%, 年化应该也约 10%(而不是被按 bar 数放大/缩小)
    index = pd.date_range("2024-01-01", periods=243, freq="D")
    nav = pd.Series(np.linspace(100, 110, 243), index=index)
    summary = summarize(nav, market="cn")
    assert summary["total_return"] == pytest.approx(0.10)
    assert summary["annual_return"] == pytest.approx(0.10, abs=0.01)


def test_intraday_annualization_not_bar_based():
    # 30 分钟净值: 一年 242 天 x 8 根 = 1936 根; 若按 bar 数(252)年化会严重错
    days = 60
    index = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=10 + 0.5 * b) for d in pd.date_range("2024-01-01", periods=days)
         for b in range(8)])
    nav = pd.Series(np.linspace(100, 110, len(index)), index=index)
    summary = summarize(nav, market="cn")
    # 60 个交易日涨 10% -> 年化约 (1.1)^(242/60)-1 ≈ 47%, 绝不是按 252 根算出来的小数字
    assert summary["days"] == days
    assert summary["annual_return"] > 0.3


def test_summarize_with_benchmark():
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    nav = pd.Series(np.linspace(100, 120, 100), index=index)
    bench = pd.Series(np.linspace(100, 110, 100), index=index)
    summary = summarize(nav, market="us", benchmark_nav=bench)
    assert summary["benchmark_return"] == pytest.approx(0.10)
    assert summary["excess_return"] == pytest.approx(0.10)


def test_summarize_beta():
    rng = np.random.default_rng(0)
    ret = rng.normal(0.001, 0.01, 200)
    index = pd.date_range("2024-01-01", periods=200, freq="D")
    nav = pd.Series(100 * np.cumprod(1 + ret), index=index)
    bench = pd.Series(100 * np.cumprod(1 + 2 * ret), index=index)   # 两倍波动 -> beta 0.5
    summary = summarize(nav, market="us", benchmark_nav=bench)
    assert summary["beta"] == pytest.approx(0.5, abs=0.05)
