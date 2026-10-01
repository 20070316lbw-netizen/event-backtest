"""委托簿记: 拒单原因、期末未成交、部分成交、执行率与 tag。"""
from __future__ import annotations

import io

import numpy as np
import pytest
from helpers import make_market
from rich.console import Console

from event_backtest import Strategy, execution_stats, reject_reasons, run
from event_backtest.broker import ORDER_COLUMNS
from event_backtest.fees import FeeSchedule
from event_backtest.report import print_result
from event_backtest.results import save_run
from event_backtest.rules import CNMarketRules, MarketRules


class _Scripted(Strategy):
    """按 bar 下标执行预设动作的策略。"""

    def __init__(self, actions):
        self.actions = actions

    def handle_data(self, ctx):
        action = self.actions.get(ctx.bar)
        if action is not None:
            action(ctx)


def test_rejected_order_records_reason():
    """一字涨停买不进: 单子被拒, 原因落进委托明细。"""
    m = make_market([[10.0], [11.22]], limit_up=[[np.inf], [11.22]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 100)}), m, initial_cash=100_000,
              fees=FeeSchedule(), rules=CNMarketRules())
    assert len(res.fills) == 0
    assert len(res.orders) == 1
    row = res.orders.iloc[0]
    assert row["status"] == "rejected"
    assert row["reason"] == "涨停买不进"
    assert row["side"] == "buy" and row["amount"] == 100 and row["filled"] == 0
    assert row["tag"] == ""


def test_order_still_open_at_end_is_cancelled_with_reason():
    m = make_market([[10.0], [10.0], [10.0]])
    res = run(_Scripted({2: lambda ctx: ctx.order(0, 10)}), m, initial_cash=1000.0,
              fees=FeeSchedule(), rules=MarketRules())
    assert len(res.fills) == 0
    row = res.orders.iloc[0]
    assert row["status"] == "cancelled"
    assert row["reason"] == "回测结束未成交"


def test_partial_fill_keeps_filled_amount_and_execution_rate():
    """现金只够一半: 先部分成交, 余量下根重试时因现金不足被拒。"""
    m = make_market([[10.0], [10.0], [10.0], [10.0]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 1000)}), m, initial_cash=1000.0,
              fees=FeeSchedule(), rules=MarketRules())
    assert len(res.fills) == 1 and res.fills.iloc[0]["amount"] == 100
    row = res.orders.iloc[0]
    assert row["filled"] == 100 and row["amount"] == 1000
    assert row["status"] == "rejected" and row["reason"] == "现金不足"

    stats = execution_stats(res.orders, res.fills, res.nav)
    assert stats["orders"] == 1
    assert stats["filled_orders"] == 0        # 没完全成交
    assert stats["rejected_orders"] == 1
    assert stats["execution_rate"] == pytest.approx(0.1)
    assert stats["turnover"] == pytest.approx(1.0)
    assert stats["cost_ratio"] == pytest.approx(0.0)


def test_reject_reasons_counted_by_reason():
    m = make_market([[10.0], [10.0], [10.0]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 100),
                         1: lambda ctx: ctx.order(0, 100)}),
              m, initial_cash=500.0, fees=FeeSchedule(), rules=MarketRules())
    reasons = reject_reasons(res.orders)
    assert reasons.sum() == 2
    assert reasons.index[0] == "现金不足"


def test_tag_is_carried_to_order_record():
    m = make_market([[10.0], [10.0]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 5, tag="exit:hold_bars")}), m,
              initial_cash=1000.0, fees=FeeSchedule(), rules=MarketRules())
    row = res.orders.iloc[0]
    assert row["tag"] == "exit:hold_bars"
    assert row["status"] == "filled"


def test_no_orders_gives_empty_frame_and_nan_stats():
    m = make_market([[10.0], [10.0]])
    res = run(_Scripted({}), m, initial_cash=1000.0, fees=FeeSchedule(),
              rules=MarketRules())
    assert res.orders.empty
    assert list(res.orders.columns) == list(ORDER_COLUMNS)
    assert reject_reasons(res.orders).empty
    stats = execution_stats(res.orders, res.fills, res.nav)
    assert stats["orders"] == 0
    assert np.isnan(stats["execution_rate"])
    assert np.isnan(stats["turnover"])


def test_vwap_fills_when_amount_is_available():
    m = make_market([[10.0], [10.0]], volume=[[100.0], [100.0]],
                    amount=[[1000.0], [1000.0]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 5)}), m, initial_cash=1000.0,
              fees=FeeSchedule(), rules=MarketRules(), fill_price="vwap")
    assert len(res.fills) == 1
    assert res.fills.iloc[0]["price"] == pytest.approx(10.0)


def test_print_result_shows_execution_and_reject_reasons():
    m = make_market([[10.0], [11.22]], limit_up=[[np.inf], [11.22]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 100)}), m, initial_cash=100_000,
              fees=FeeSchedule(), rules=CNMarketRules())
    buffer = io.StringIO()
    print_result(res, market="cn", console=Console(file=buffer, width=100, no_color=True))
    text = buffer.getvalue()
    assert "执行率" in text and "拒单原因" in text and "涨停买不进" in text


def test_save_run_includes_orders_and_execution(tmp_path):
    m = make_market([[10.0], [10.0], [10.0]])
    res = run(_Scripted({0: lambda ctx: ctx.order(0, 5)}), m, initial_cash=1000.0,
              fees=FeeSchedule(), rules=MarketRules())
    out = save_run(tmp_path, res, market="us", strategy="test")
    assert (out / "orders.parquet").exists()
    assert (out / "execution.csv").exists()
    assert (out / "nav.parquet").exists()
    assert (out / "run.json").exists()
