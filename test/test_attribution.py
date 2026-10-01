"""信号归因与建仓口径: event_id 串联 + 整手取整 + 单笔仓位比例。"""
from __future__ import annotations

import numpy as np
from helpers import make_market

from event_backtest import Strategy, load_strategy, run
from event_backtest.fees import FeeSchedule
from event_backtest.rules import CNMarketRules, MarketRules
from event_backtest.strategy.declarative import DeclarativeStrategy


def _rising_market_with_volume_spike(bars: int = 40, spike: int = 20):
    """一路慢涨 + 某根放量 -> 触发放量突破。"""
    close = 10.0 * np.cumprod(np.full((bars, 1), 1.005))
    volume = np.full((bars, 1), 1000.0)
    volume[spike] = 10_000.0
    return make_market(close, volume=volume, tickers=["AAA"])


class _Scripted(Strategy):
    def __init__(self, actions):
        self.actions = actions

    def handle_data(self, ctx):
        action = self.actions.get(ctx.bar)
        if action is not None:
            action(ctx)


def test_event_id_flows_to_orders_fills_and_trades():
    market = _rising_market_with_volume_spike()
    result = run(DeclarativeStrategy(load_strategy("volume_breakout")), market,
                 initial_cash=1_000_000.0, fees=FeeSchedule(), rules=MarketRules())

    assert len(result.orders) > 0
    assert (result.orders["event_id"].astype(str).str.len() > 0).all()
    assert (result.fills["event_id"].astype(str).str.len() > 0).all()
    assert result.orders["event_id"].iloc[0].startswith("放量突破|volume_breakout|AAA|")

    from event_backtest.metrics import last_marks, trades
    frame = trades(result.fills, marks=last_marks(result.market))
    assert (frame["entry_event"].astype(str).str.len() > 0).all()


def test_entry_percent_defaults_to_max_positions_share():
    """不写 sizing.percent 时按 max_positions 等权分摊, 而不是全池 1/N。"""
    market = _rising_market_with_volume_spike()
    result = run(DeclarativeStrategy(load_strategy("volume_breakout")), market,
                 initial_cash=1_000_000.0, fees=FeeSchedule(), rules=MarketRules())
    buy = result.fills[result.fills["side"] == "buy"].iloc[0]
    invested = float(buy["amount"]) * float(buy["price"])
    # max_positions=3 -> 单笔约 1/3 净值; 老口径是全池 1/1(只有一只证券) = 100%
    assert 0.25 < invested / 1_000_000.0 < 0.40


def test_target_percent_rounds_to_lot_and_clears_odd_lots():
    """A 股目标仓位单必须按整手委托; 清仓时连零股一起卖。"""
    market = make_market([[10.0], [10.0], [10.0]], tickers=["AAA"])
    strategy = _Scripted({0: lambda ctx: ctx.order_target_percent(0, 1 / 3)})
    result = run(strategy, market, initial_cash=1_000_000.0, fees=FeeSchedule(),
                 rules=CNMarketRules())
    order = result.orders.iloc[0]
    assert order["amount"] % 100 == 0                      # 不再是 33333 股
    assert order["filled"] % 100 == 0
    assert not (result.orders["reason"] == "数量不足一手").any()


def test_clear_position_sells_everything():
    market = make_market([[10.0], [10.0], [10.0], [10.0]], tickers=["AAA"])
    strategy = _Scripted({
        0: lambda ctx: ctx.order(0, 150),                  # 150 股 -> 成交 100 股
        1: lambda ctx: ctx.order_target_percent(0, 0.0),   # 清仓: 把 100 股全卖掉
    })
    result = run(strategy, market, initial_cash=100_000.0, fees=FeeSchedule(),
                 rules=CNMarketRules())
    sells = result.fills[result.fills["side"] == "sell"]
    assert len(sells) == 1 and int(sells.iloc[0]["amount"]) == 100
    assert result.portfolio.total[0] == 0
