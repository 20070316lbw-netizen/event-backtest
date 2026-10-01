"""盯市回归: 缺价 bar(停牌 / 零成交量)不能让持仓按 0 计, 净值不应瞬间跳水。"""
from __future__ import annotations

import numpy as np
import pytest
from helpers import make_market

from event_backtest.engine import run
from event_backtest.fees import FeeSchedule
from event_backtest.rules import MarketRules
from event_backtest.strategy.base import Strategy


class _BuyOnce(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 10)


def test_missing_bar_keeps_last_mark():
    # 第 2 根 bar 缺价(停牌 / 零成交量), 持仓应沿用上一个有效收盘价
    close = np.array([[10.0], [10.0], [np.nan], [10.0]])
    market = make_market(close, tickers=["AAA"])
    result = run(_BuyOnce(), market, initial_cash=1000.0,
                 fees=FeeSchedule(), rules=MarketRules())
    nav = result.nav.to_numpy()
    assert result.portfolio.total[0] == 10
    assert nav[1] == pytest.approx(1000.0)
    assert nav[2] == pytest.approx(nav[1])   # 缺价不跳水
