"""引擎行为: 下一根撮合、整手、涨跌停、T+1 / T+0、费用。"""
from __future__ import annotations

import numpy as np
import pytest

from event_backtest import MarketData, Strategy, run
from event_backtest.fees import FeeSchedule
from event_backtest.rules import CNMarketRules


def _market(close, *, open_=None, high=None, low=None, volume=None, pre_close=None,
            limit_up=None, limit_down=None, is_new_day=None, t0=False,
            tickers=("AAA.SH",), sec_type="stock") -> MarketData:
    close = np.asarray(close, float)
    if close.ndim == 1:
        close = close[:, None]
    T, N = close.shape

    def arr(x, default):
        if x is None:
            return np.full((T, N), default, float)
        a = np.asarray(x, float)
        return a[:, None] if a.ndim == 1 else a

    o = close.copy() if open_ is None else arr(open_, np.nan)
    h = close.copy() if high is None else arr(high, np.nan)
    low_ = close.copy() if low is None else arr(low, np.nan)
    vol = np.full((T, N), 1000.0) if volume is None else arr(volume, np.nan)
    if pre_close is None:
        pre = np.vstack([np.full((1, N), np.nan), close[:-1]])
    else:
        pre = arr(pre_close, np.nan)
    nd = np.ones(T, bool) if is_new_day is None else np.asarray(is_new_day, bool)
    return MarketData(
        ts=np.arange(T).astype("datetime64[D]").astype("datetime64[ns]"),
        tickers=tuple(tickers), open=o, high=h, low=low_, close=close,
        volume=vol, amount=np.full((T, N), np.nan), pre_close=pre,
        adj_factor=np.ones((T, N)), limit_up=arr(limit_up, np.inf),
        limit_down=arr(limit_down, -np.inf),
        tradable=~np.isnan(close) & (np.nan_to_num(vol) > 0), is_new_day=nd,
        t0=np.full(N, t0, bool), sec_type=tuple([sec_type] * N),
    )


class _Scripted(Strategy):
    """按 bar 下标执行预设动作的策略。"""

    def __init__(self, actions):
        self.actions = actions

    def handle_data(self, ctx):
        action = self.actions.get(ctx.bar)
        if action is not None:
            action(ctx)


def test_market_order_fills_at_next_bar_open():
    m = _market([[10.0], [10.2], [10.5]], open_=[[10.0], [10.3], [10.5]])
    strategy = _Scripted({0: lambda ctx: ctx.order(0, 100)})
    res = run(strategy, m, initial_cash=10_000, fees=FeeSchedule())
    assert len(res.fills) == 1
    fill = res.fills.iloc[0]
    assert int(fill["bar"]) == 1
    assert fill["price"] == pytest.approx(10.3)
    assert fill["side"] == "buy"
    assert res.portfolio.total[0] == 100


def test_buy_is_rounded_to_lot():
    m = _market([[10.0], [10.0]])
    strategy = _Scripted({0: lambda ctx: ctx.order(0, 150)})
    res = run(strategy, m, initial_cash=100_000, fees=FeeSchedule(), rules=CNMarketRules())
    assert res.portfolio.total[0] == 100   # 150 -> 一手 100


def test_buy_blocked_on_limit_up():
    m = _market([[10.0], [11.22]], open_=[[10.0], [11.22]], high=[[10.0], [11.22]],
                low=[[10.0], [11.22]], pre_close=[[10.0], [10.2]],
                limit_up=[[np.inf], [11.22]])
    strategy = _Scripted({0: lambda ctx: ctx.order(0, 100)})
    res = run(strategy, m, initial_cash=100_000, fees=FeeSchedule(), rules=CNMarketRules())
    assert len(res.fills) == 0
    assert res.portfolio.total[0] == 0


def test_t_plus_one_blocks_same_day_sell():
    # bar0/bar1 同一天(第一天), bar2 仍同一天, bar3 次日
    m = _market([[10.0], [10.0], [10.0], [10.0]], is_new_day=[True, False, False, True])
    strategy = _Scripted({
        0: lambda ctx: ctx.order(0, 100),   # 次日(bar1)买入
        1: lambda ctx: ctx.order(0, -100),  # 当天(bar2)想卖, 应被 T+1 拒绝
    })
    res = run(strategy, m, initial_cash=100_000, fees=FeeSchedule(), rules=CNMarketRules())
    assert len(res.fills) == 1
    assert res.portfolio.total[0] == 100   # 卖单被拒, 持仓还在


def test_t0_allows_same_day_sell():
    m = _market([[10.0], [10.0], [10.0], [10.0]], is_new_day=[True, False, False, True],
                t0=True, sec_type="etf")
    strategy = _Scripted({
        0: lambda ctx: ctx.order(0, 100),
        1: lambda ctx: ctx.order(0, -100),
    })
    res = run(strategy, m, initial_cash=100_000, fees=FeeSchedule(), rules=CNMarketRules())
    assert len(res.fills) == 2
    assert res.portfolio.total[0] == 0


def test_fill_price_must_be_known():
    """写错的口径直接报错, 不再被静默当成 vwap。"""
    m = _market([[10.0], [10.0]])
    with pytest.raises(ValueError, match="fill_price"):
        run(_Scripted({}), m, initial_cash=1000.0, fill_price="close")


def test_vwap_requires_amount_data():
    """行情没有成交额时用 vwap 会全量拒单, 应该在开跑前报错。"""
    m = _market([[10.0], [10.0]])   # _market 的 amount 全是 NaN
    with pytest.raises(ValueError, match="vwap"):
        run(_Scripted({}), m, initial_cash=1000.0, fill_price="vwap")


def test_sell_pays_stamp_duty():
    m = _market([[10.0], [10.0], [10.0]], is_new_day=[True, True, True])
    fees = FeeSchedule(commission=0.0, stamp_duty=0.001)
    strategy = _Scripted({
        0: lambda ctx: ctx.order(0, 100),
        1: lambda ctx: ctx.order(0, -100),
    })
    res = run(strategy, m, initial_cash=100_000, fees=fees, rules=CNMarketRules())
    sell = res.fills[res.fills["side"] == "sell"].iloc[0]
    assert sell["fee"] == pytest.approx(100 * 10.0 * 0.001)
