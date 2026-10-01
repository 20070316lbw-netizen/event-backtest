"""滑点: 固定价差 / 成交量比例 / 接入撮合。"""
from __future__ import annotations

import pytest
from helpers import make_market

from event_backtest.engine import run
from event_backtest.fees import FeeSchedule
from event_backtest.rules import MarketRules
from event_backtest.slippage import (
    FixedSlippage,
    VolumeShareSlippage,
    make_slippage,
)
from event_backtest.strategy.base import Strategy


def test_fixed_slippage_prices():
    s = FixedSlippage(spread=0.02)
    assert s.apply(10.0, is_buy=True, bar_volume=1000, qty=100)[0] == pytest.approx(10.01)
    assert s.apply(10.0, is_buy=False, bar_volume=1000, qty=100)[0] == pytest.approx(9.99)


def test_volume_share_caps_and_impacts():
    s = VolumeShareSlippage(volume_limit=0.1, price_impact=0.5)
    price, cap = s.apply(10.0, is_buy=True, bar_volume=1000, qty=500)
    assert cap == 100                                   # 0.1 * 1000
    share = 100 / 1000
    assert price == pytest.approx(10.0 * (1 + 0.5 * share**2))
    sell_price, _ = s.apply(10.0, is_buy=False, bar_volume=1000, qty=100)
    assert sell_price < 10.0


def test_volume_share_zero_volume():
    assert VolumeShareSlippage().apply(10.0, is_buy=True, bar_volume=0.0, qty=100) == (10.0, 0)


def test_make_slippage_unknown():
    with pytest.raises(ValueError):
        make_slippage("nope")


class _BuyOnce(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 10)


def test_slippage_applied_in_engine():
    m = make_market([[10.0], [10.0]])
    base = run(_BuyOnce(), m, initial_cash=10000, fees=FeeSchedule(), rules=MarketRules())
    slipped = run(_BuyOnce(), m, initial_cash=10000, fees=FeeSchedule(), rules=MarketRules(),
                  slippage=FixedSlippage(spread=0.1))
    assert base.fills.iloc[0]["price"] == pytest.approx(10.0)
    assert slipped.fills.iloc[0]["price"] == pytest.approx(10.05)
