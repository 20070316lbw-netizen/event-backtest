"""基准: 把市场行情里的某只证券当成"买入持有"基准。

配置里的 benchmark 字段之前只解析不用; 现在 tearsheet 用它画对比线, 汇总里用它算
超额与 Beta。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_backtest.engine import BacktestResult
from event_backtest.metrics import buy_and_hold_nav

__all__ = ["benchmark_nav"]


def benchmark_nav(result: BacktestResult, ticker: str | None, *,
                  initial: float | None = None) -> pd.Series | None:
    """用基准证券的后复权收盘价构造买入持有净值。

    Args:
        result: 回测结果(提供行情与起点净值)。
        ticker: 基准证券代码; None、或不在本次行情的证券里时返回 None。
        initial: 基准起点净值; None 时用策略的起点净值, 方便两条线直接比。

    Returns:
        与 result.nav 同时间轴的净值序列; 无法构造时返回 None。
    """
    if not ticker or ticker not in result.market.tickers:
        return None
    index = result.market.tickers.index(ticker)
    values = result.market.adj_close[:, index]
    if not np.isfinite(values).any():
        return None
    start = float(result.nav.iloc[0]) if initial is None else initial
    return buy_and_hold_nav(values, index=pd.DatetimeIndex(result.market.ts, name="ts"),
                            initial=start)
