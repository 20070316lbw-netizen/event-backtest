"""回合交易: 期末未平仓用盯市价补一行, 统计里单独计数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_market

from event_backtest import Strategy, run
from event_backtest.fees import FeeSchedule
from event_backtest.metrics import last_marks, trade_stats, trades
from event_backtest.rules import MarketRules


class _BuyAndHold(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 10)


class _RoundTrip(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 10)
        if ctx.bar == 2:
            ctx.order(0, -10)


def _result(strategy):
    market = make_market([[10.0], [11.0], [12.0], [11.5]], tickers=["AAA"])
    return run(strategy, market, initial_cash=10_000.0, fees=FeeSchedule(),
               rules=MarketRules())


def test_without_marks_only_closed_round_trips():
    result = _result(_BuyAndHold())
    assert trades(result.fills).empty                      # 老行为: 没平仓就没有行


def test_marks_add_open_position_row():
    result = _result(_BuyAndHold())
    frame = trades(result.fills, marks=last_marks(result.market))
    assert len(frame) == 1
    row = frame.iloc[0]
    assert bool(row["open"]) is True
    assert pd.isna(row["exit_ts"]) and np.isnan(row["bars"])
    assert row["exit_px"] == pytest.approx(11.5)           # 最后一个收盘价
    assert row["entry_px"] == pytest.approx(11.0)          # bar0 下单, bar1 开盘成交
    assert row["pnl"] == pytest.approx(10 * (11.5 - 11.0))  # 浮动盈亏(买入零费用)


def test_trade_stats_counts_open_separately():
    result = _result(_BuyAndHold())
    stats = trade_stats(trades(result.fills, marks=last_marks(result.market)))
    assert stats["trades"] == 0 and stats["open_trades"] == 1
    assert np.isnan(stats["win_rate"])                     # 未平仓不进胜率

    closed = trade_stats(trades(_result(_RoundTrip()).fills,
                                marks=last_marks(_result(_RoundTrip()).market)))
    assert closed["trades"] == 1 and closed["open_trades"] == 0
    assert closed["win_rate"] == pytest.approx(1.0)


def test_last_marks_uses_last_finite_close():
    market = make_market([[1.0, 5.0], [np.nan, 6.0], [np.nan, np.nan]],
                         tickers=["A", "B"])
    marks = last_marks(market)
    assert marks["A"] == pytest.approx(1.0)                # 只有第一根有价
    assert marks["B"] == pytest.approx(6.0)                # 最后一根没有就往前找
