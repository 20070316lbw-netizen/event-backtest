"""测试用的行情构造。"""
from __future__ import annotations

import numpy as np

from event_backtest.market import MarketData


def make_market(close, *, volume=None, amount=None, is_new_day=None, t0=False,
                sec_type="stock", pre_close=None, limit_up=None, limit_down=None,
                tickers=None) -> MarketData:
    """用收盘价(可 (T,N))快速造一个 MarketData; 其余字段给合理默认。"""
    close = np.asarray(close, float)
    if close.ndim == 1:
        close = close[:, None]
    T, N = close.shape

    def arr(x, default):
        if x is None:
            return np.full((T, N), default, float)
        a = np.asarray(x, float)
        return a[:, None] if a.ndim == 1 else a

    vol = np.full((T, N), 1000.0) if volume is None else arr(volume, np.nan)
    pre = (np.vstack([np.full((1, N), np.nan), close[:-1]]) if pre_close is None
           else arr(pre_close, np.nan))
    return MarketData(
        ts=np.arange(T).astype("datetime64[D]").astype("datetime64[ns]"),
        tickers=tuple(tickers) if tickers else tuple(f"T{i}" for i in range(N)),
        open=close.copy(), high=close.copy(), low=close.copy(), close=close,
        volume=vol, amount=arr(amount, np.nan), pre_close=pre,
        adj_factor=np.ones((T, N)),
        limit_up=arr(limit_up, np.inf), limit_down=arr(limit_down, -np.inf),
        tradable=~np.isnan(close) & (np.nan_to_num(vol) > 0),
        is_new_day=np.ones(T, bool) if is_new_day is None else np.asarray(is_new_day, bool),
        t0=np.full(N, t0, bool), sec_type=tuple([sec_type] * N),
    )
