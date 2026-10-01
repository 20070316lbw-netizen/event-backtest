"""市场规则: 涨跌停分板块 / ST 随日期 / T+0 / 最小价位 / 四舍五入。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from event_backtest.rules import CNMarketRules, USMarketRules, default_rules, limit_prices


def test_cn_limit_pct_by_board():
    r = CNMarketRules()
    assert r.limit_pct("600519.SH", "stock", "贵州茅台", False) == 0.10
    assert r.limit_pct("300750.SZ", "stock", "宁德时代", False) == 0.20
    assert r.limit_pct("688981.SH", "stock", "中芯国际", False) == 0.20
    assert r.limit_pct("830799.BJ", "stock", "艾融软件", False) == 0.30
    assert r.limit_pct("159915.SZ", "etf", "创业板ETF", False) == 0.20
    assert r.limit_pct("510300.SH", "etf", "沪深300ETF", False) == 0.10


def test_cn_st_limit_changes_with_date_and_etf_ignores_st():
    r = CNMarketRules()
    assert r.limit_pct("600243.SH", "stock", "*ST某某", True, date(2026, 7, 3)) == 0.05
    assert r.limit_pct("600243.SH", "stock", "*ST某某", True, date(2026, 7, 6)) == 0.10
    assert r.limit_pct("510300.SH", "etf", "沪深300ETF", True) == 0.10


def test_cn_t0_and_lot_and_tick():
    r = CNMarketRules()
    assert r.is_t0("518880.SH", "etf", "华安黄金ETF") is True
    assert r.is_t0("510300.SH", "etf", "沪深300ETF") is False
    assert r.is_t0("600519.SH", "stock", "贵州茅台") is False
    assert r.lot_size == 100
    assert r.tick("etf") == 0.001
    assert r.tick("stock") == 0.01


def test_overrides_win_over_keywords():
    r = CNMarketRules(limit_overrides={"159915.SZ": 0.10}, t0_overrides={"510300.SH": True})
    assert r.limit_pct("159915.SZ", "etf", "创业板ETF", False) == 0.10
    assert r.is_t0("510300.SH", "etf", "沪深300ETF") is True


def test_us_rules_are_permissive():
    r = USMarketRules()
    assert r.limit_pct("AAPL", "stock", "Apple", False) == 0.0
    assert r.is_t0("AAPL", "stock", "Apple") is True
    assert r.lot_size == 1


def test_limit_prices_round_half_up():
    up, down = limit_prices(np.array([10.05]), np.array([0.10]), np.array([0.01]))
    assert float(up[0]) == pytest.approx(11.06)
    assert float(down[0]) == pytest.approx(9.05)


def test_default_rules_and_unknown_market():
    assert isinstance(default_rules("cn"), CNMarketRules)
    assert isinstance(default_rules("us"), USMarketRules)
    with pytest.raises(ValueError):
        default_rules("hk")


def test_st_limits_must_be_sorted():
    with pytest.raises(ValueError):
        CNMarketRules(st_limits=((date(2026, 1, 1), 0.05), (date(2020, 1, 1), 0.10)))
