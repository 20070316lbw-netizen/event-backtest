"""事件研究: 信号触发后 k 根 bar 的收益, 以及与同时段无条件基线的差。

口径(与 minievent 对齐):
    - 第 k 根累计收益 = adj_close[t+k] / adj_close[t] - 1, 起点是信号 bar 收盘;
    - 基线 = 同一证券、同一"结束时刻"(time-of-day)的全部有效窗口均值, 含事件本身;
      这是描述性的无条件基准, 不代表样本外超额;
    - 每个 horizon 输出事件数、有效样本数、均值、中位数、胜率、基线与超额。

只做描述性统计, 不提供 p 值 / bootstrap 区间(那是 minievent 的完整版; 这里是轻量版)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from event_backtest.market import MarketData

__all__ = ["EventStudy", "study_events"]

# 默认观察窗口: 信号后 1..16 根 bar(30 分钟线下正好两天)
_DEFAULT_HORIZONS = tuple(range(1, 17))


@dataclass(frozen=True)
class EventStudy:
    """事件研究结果。

    Attributes:
        events: 输入的事件表。
        paths: 逐事件逐 horizon 的路径, 列
            [strategy, event_id, ticker, bar, ts, horizon, ret, baseline_ret, excess_ret]。
        summary: 每个 horizon 一行的汇总, 列
            [horizon, n_events, n, mean_ret, median_ret, win_rate, baseline, excess]。
    """

    events: pd.DataFrame
    paths: pd.DataFrame
    summary: pd.DataFrame


def study_events(market: MarketData, events: pd.DataFrame, *,
                 horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
                 strategy_name: str = "") -> EventStudy:
    """对事件表做事件后收益研究。

    Args:
        market: 对齐后的行情(用后复权收盘价算收益)。
        events: 事件表, 至少含 [event_id, bar, ts, ticker]。
        horizons: 观察窗口(bar 数), 默认 1..16。
        strategy_name: 写进结果第一列, 便于多策略拼接。

    Returns:
        EventStudy; 事件为空时返回空 paths 与带列名的空 summary。
    """
    horizons = tuple(int(h) for h in horizons)
    adj = market.adj_close
    total = len(market.ts)
    index = {t: i for i, t in enumerate(market.tickers)}
    # 基线只需算一次, 所有事件共用
    baseline = _baseline(adj, market.ts, horizons)

    rows: list[dict[str, object]] = []
    for row in events.to_dict("records"):
        i = index.get(str(row["ticker"]))
        if i is None:
            continue
        t = int(row["bar"])
        for k in horizons:
            ret, tod = np.nan, None
            if t + k < total:
                entry, exit_ = adj[t, i], adj[t + k, i]
                # 起止价都必须是有限正数才算有效窗口
                if np.isfinite(entry) and np.isfinite(exit_) and entry > 0 and exit_ > 0:
                    ret = float(exit_ / entry - 1)
                tod = pd.Timestamp(market.ts[t + k]).time()
            # 优先用"同证券同时刻"的基线, 没有就退到该证券该 horizon 的总体均值
            base = baseline.get((i, k, tod), baseline.get((i, k, None), np.nan))
            excess = ret - base if np.isfinite(ret) and np.isfinite(base) else np.nan
            rows.append({
                "strategy": strategy_name, "event_id": row["event_id"], "ticker": row["ticker"],
                "bar": t, "ts": row["ts"], "horizon": k, "ret": ret,
                "baseline_ret": base, "excess_ret": excess,
            })

    paths = pd.DataFrame(rows, columns=["strategy", "event_id", "ticker", "bar", "ts",
                                        "horizon", "ret", "baseline_ret", "excess_ret"])
    return EventStudy(events=events, paths=paths, summary=_summary(paths))


def _baseline(adj: np.ndarray, ts: np.ndarray,
              horizons: tuple[int, ...]) -> dict[tuple[int, int, object], float]:
    """算无条件基线: 每个证券、每个 horizon、每个结束时刻的平均收益。

    对每个 horizon, 把所有 t 的前瞻收益都算出来, 再按 (证券, 结束时刻) 分组求均值;
    另外存一份 (证券, horizon, None) 的总体均值作为兜底。含事件本身, 只作描述。

    Args:
        adj: (T, N) 后复权收盘价。
        ts: (T,) 时间轴。
        horizons: 观察窗口。

    Returns:
        {(证券下标, horizon, time-of-day 或 None): 平均收益}。
    """
    total, n_assets = adj.shape
    out: dict[tuple[int, int, object], float] = {}
    for k in horizons:
        if total - k <= 0:
            continue
        entry, exit_ = adj[:total - k, :], adj[k:, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = exit_ / entry - 1
        valid = np.isfinite(rets) & (entry > 0) & (exit_ > 0)
        # 每根"结束 bar"的日内时刻, 用于同时段分组
        tods = np.array([pd.Timestamp(x).time() for x in ts[k:]], dtype=object)
        for i in range(n_assets):
            column_valid = valid[:, i]
            if not column_valid.any():
                continue
            out[(i, k, None)] = float(rets[column_valid, i].mean())
            for tod in pd.unique(tods[column_valid]):
                selected = column_valid & (tods == tod)
                out[(i, k, tod)] = float(rets[selected, i].mean())
    return out


def _summary(paths: pd.DataFrame) -> pd.DataFrame:
    """按 horizon 汇总路径表。"""
    columns = ["horizon", "n_events", "n", "mean_ret", "median_ret", "win_rate",
               "baseline", "excess"]
    if paths.empty:
        return pd.DataFrame(columns=columns)
    grouped = paths.groupby("horizon")
    summary = pd.DataFrame({
        "n_events": grouped.size(),                     # 含无效窗口的事件数
        "n": grouped["ret"].count(),                    # 只有效窗口
        "mean_ret": grouped["ret"].mean(),
        "median_ret": grouped["ret"].median(),
        "win_rate": grouped["ret"].apply(_win_rate),
        "baseline": grouped["baseline_ret"].mean(),
        "excess": grouped["excess_ret"].mean(),
    }).reset_index()
    return summary[columns]


def _win_rate(series: pd.Series) -> float:
    """收益率里大于 0 的比例; 全为 NaN 时返回 NaN。"""
    values = series.dropna()
    return float((values > 0).mean()) if len(values) else np.nan
