"""MarketData 构建: 列对齐、复权因子、涨跌停、零成交量 bar。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from event_backtest.market import build_cn_market, build_us_market


def _cn_frames():
    bars = pd.DataFrame([
        {"ts": "2024-01-02", "ticker": "AAA.SH", "open": 10.0, "high": 10.5, "low": 9.8,
         "close": 10.2, "volume": 1000, "amount": 10200},
        {"ts": "2024-01-03", "ticker": "AAA.SH", "open": 10.2, "high": 10.6, "low": 10.1,
         "close": 10.5, "volume": 1200, "amount": 12600},
        {"ts": "2024-01-02", "ticker": "510300.SH", "open": 4.0, "high": 4.1, "low": 3.9,
         "close": 4.0, "volume": 500, "amount": 2000},
        {"ts": "2024-01-03", "ticker": "510300.SH", "open": 4.0, "high": 4.0, "low": 4.0,
         "close": 4.0, "volume": 0, "amount": 0},
    ])
    daily = pd.DataFrame([
        {"date": "2024-01-02", "ticker": "AAA.SH", "close": 10.2, "pre_close": 10.0,
         "is_suspended": False, "is_st": False},
        {"date": "2024-01-03", "ticker": "AAA.SH", "close": 10.5, "pre_close": 10.2,
         "is_suspended": False, "is_st": False},
        {"date": "2024-01-02", "ticker": "510300.SH", "close": 4.0, "pre_close": 4.0,
         "is_suspended": False, "is_st": False},
        {"date": "2024-01-03", "ticker": "510300.SH", "close": 4.0, "pre_close": 4.0,
         "is_suspended": False, "is_st": False},
    ])
    basic = pd.DataFrame([
        {"ticker": "AAA.SH", "name": "测试股票", "sec_type": "stock"},
        {"ticker": "510300.SH", "name": "沪深300ETF", "sec_type": "etf"},
    ])
    return bars, daily, basic


def test_build_cn_market_aligns_and_limits():
    bars, daily, basic = _cn_frames()
    m = build_cn_market(bars, daily, basic, tickers=["AAA.SH", "510300.SH"])
    assert m.shape == (2, 2)
    assert m.tickers == ("AAA.SH", "510300.SH")
    assert m.limit_up[0, 0] == pytest.approx(11.0)   # 前收 10.0, 主板 10%
    assert m.limit_down[0, 0] == pytest.approx(9.0)
    assert m.limit_up[0, 1] == pytest.approx(4.4)    # ETF 10%
    assert list(m.t0) == [False, False]


def test_build_cn_market_zero_volume_bar_is_not_tradable():
    bars, daily, basic = _cn_frames()
    m = build_cn_market(bars, daily, basic, tickers=["AAA.SH", "510300.SH"])
    assert np.isnan(m.close[1, 1])
    assert bool(m.tradable[1, 1]) is False
    assert bool(m.tradable[0, 1]) is True


def test_build_us_market_uses_adj_close():
    prices = pd.DataFrame([
        {"date": "2024-01-02", "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "adj_close": 50.25, "volume": 1000},
        {"date": "2024-01-03", "ticker": "AAPL", "open": 100.5, "high": 102.0, "low": 100.0,
         "close": 101.0, "adj_close": 50.5, "volume": 1200},
    ])
    m = build_us_market(prices, tickers=["AAPL"])
    assert m.shape == (2, 1)
    assert m.adj_factor[0, 0] == pytest.approx(0.5)
    assert m.adj_close[1, 0] == pytest.approx(50.5)
    assert m.pre_close[1, 0] == pytest.approx(100.5)
    assert np.isnan(m.pre_close[0, 0])
    assert np.isnan(m.limit_up).all()
    assert bool(m.t0[0]) is True
    assert bool(m.tradable.all())
