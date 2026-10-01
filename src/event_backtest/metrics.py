"""绩效与交易统计: 净值 -> 组合指标; 成交流水 -> 回合交易。

组合指标只看净值序列(与逐笔成交无关); 交易统计把成交流水配成"开仓到平仓"的回合。

年化口径: 分钟线的逐 bar 收益有日内自相关、还混着隔夜跳空, 直接按"一年几千根 bar"
年化会高估。这里统一先把净值聚成**日频**(每天最后一个点), 再按市场交易日数年化
(A 股 242, 美股 252), 与 minievent 的口径一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_backtest.broker import CANCELLED, FILLED, REJECTED

# 每年交易日: A 股 242, 美股 252
_TRADING_DAYS = {"cn": 242, "us": 252}


def periods_per_year(market: str) -> int:
    """该市场每年多少个交易日, 作为日频年化因子。未知市场按美股 252。"""
    return _TRADING_DAYS.get(market, 252)


def daily_nav(nav: pd.Series) -> pd.Series:
    """bar 级净值 -> 日频净值(每天取最后一个点)。

    日线输入时基本等同于原序列; 分钟线输入时把一天内多根 bar 压成一个日频点,
    之后的风险指标就在日频上算。

    Args:
        nav: 逐 bar 净值, index 是时间。

    Returns:
        日频净值序列; index 不是时间时原样返回。
    """
    if not isinstance(nav.index, pd.DatetimeIndex) or len(nav) == 0:
        return nav
    daily = nav.groupby(nav.index.normalize()).last()
    daily.index = pd.DatetimeIndex(daily.index, name=nav.index.name)
    return daily


def performance(nav: pd.Series, *, periods_per_year: int = 252) -> pd.Series:
    """净值序列的常用指标。

    Args:
        nav: 净值序列(index 是时间); 分钟线请先经 daily_nav 聚成日频。
        periods_per_year: 年化用的每年期数; 日频 A 股 242 / 美股 252。

    Returns:
        pd.Series, 含 total_return / annual_return / annual_vol / sharpe /
        max_drawdown; 样本不足 2 个点时全部为 NaN。
    """
    nav = nav.dropna()
    if len(nav) < 2:
        return pd.Series({
            "total_return": np.nan, "annual_return": np.nan, "annual_vol": np.nan,
            "sharpe": np.nan, "max_drawdown": np.nan,
        })
    returns = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years = len(nav) / periods_per_year
    annual = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    vol = std * np.sqrt(periods_per_year) if np.isfinite(std) else np.nan
    # 波动为 0 时夏普无意义, 记 NaN 而不是 inf
    sharpe = (float(returns.mean()) / std * np.sqrt(periods_per_year)
              if np.isfinite(std) and std > 0 else np.nan)
    drawdown = nav / nav.cummax() - 1
    return pd.Series({
        "total_return": total,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    })


def max_drawdown_duration(nav: pd.Series) -> int:
    """最长回撤持续时间: 连续处在"历史高点之下"的最长段, 单位是期数(日)。"""
    nav = nav.dropna()
    if len(nav) == 0:
        return 0
    underwater = (nav < nav.cummax()).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def buy_and_hold_nav(values: np.ndarray, *, index: pd.Index, initial: float = 1.0) -> pd.Series:
    """买入持有净值: 用某个价格序列(通常是基准证券的后复权收盘价)归一。

    Args:
        values: (T,) 价格序列。
        index: 与 values 等长的时间索引。
        initial: 起点净值。

    Returns:
        净值序列; values 全为 NaN 时返回全 NaN。
    """
    series = pd.Series(np.asarray(values, dtype=float), index=index).ffill()
    base = series.dropna()
    if base.empty or base.iloc[0] <= 0:
        return pd.Series(np.nan, index=index)
    return initial * series / base.iloc[0]


def summarize(nav: pd.Series, *, market: str = "us",
              benchmark_nav: pd.Series | None = None) -> pd.Series:
    """把净值序列汇总成报告要用的全部指标(日频口径)。

    Args:
        nav: 逐 bar 净值。
        market: "cn" / "us", 决定年化交易日数。
        benchmark_nav: 基准的买入持有净值(与 nav 同时间轴); None 表示不算基准对比。

    Returns:
        pd.Series, 字段: start / end / bars / days / total_return / annual_return /
        annual_vol / sharpe / sortino / calmar / max_drawdown / max_drawdown_days,
        以及基准存在时的 benchmark_return / excess_return / beta。
    """
    daily = daily_nav(nav).dropna()
    periods = periods_per_year(market)
    perf = performance(daily, periods_per_year=periods)
    out = {
        "start": daily.index[0] if len(daily) else pd.NaT,
        "end": daily.index[-1] if len(daily) else pd.NaT,
        "bars": len(nav.dropna()),
        "days": len(daily),
        **perf.to_dict(),
        "max_drawdown_days": max_drawdown_duration(daily),
    }

    returns = daily.pct_change().dropna()
    downside = returns[returns < 0]
    # Sortino: 只惩罚下行波动(分母用负收益的标准差)
    out["sortino"] = (float(returns.mean()) / float(downside.std(ddof=1)) * np.sqrt(periods)
                      if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan)
    # Calmar: 年化收益 / 最大回撤绝对值
    out["calmar"] = (perf["annual_return"] / abs(perf["max_drawdown"])
                     if perf["max_drawdown"] < 0 else np.nan)
    out["max_drawdown"] = perf["max_drawdown"]

    if benchmark_nav is not None and len(benchmark_nav):
        bench = daily_nav(benchmark_nav).reindex(daily.index).ffill()
        bench_return = float(bench.iloc[-1] / bench.iloc[0] - 1) if bench.iloc[0] else np.nan
        out["benchmark_return"] = bench_return
        out["excess_return"] = perf["total_return"] - bench_return
        # Beta: 策略日收益对基准日收益回归的斜率
        joined = pd.concat([returns.rename("s"),
                            bench.pct_change().reindex(returns.index).rename("b")],
                           axis=1).dropna()
        variance = float(joined["b"].var(ddof=1)) if len(joined) > 1 else 0.0
        out["beta"] = (float(joined["s"].cov(joined["b"])) / variance
                       if variance > 0 else np.nan)
    return pd.Series(out)


# 回合交易的列
_TRADE_COLUMNS = ["ticker", "entry_ts", "exit_ts", "qty", "entry_px", "exit_px",
                  "pnl", "ret", "fees", "bars"]


def trades(fills: pd.DataFrame) -> pd.DataFrame:
    """成交流水 -> 回合交易(按证券加权平均成本)。

    对每只证券维护持仓数量、成本总额与累计已实现盈亏; 持仓从 0 建到再次归零算一笔。
    加仓 / 减仓用加权平均成本摊销, 只有回到空仓那一刻才输出一行。

    Args:
        fills: 成交明细(BacktestResult.fills), 至少含
            [bar, ts, ticker, side, amount, price, fee]。

    Returns:
        DataFrame, 列 _TRADE_COLUMNS; 没有成交时是带列名的空表。
        ret 是相对首次建仓名义金额的净收益率(已扣双边费用)。
    """
    if fills is None or len(fills) == 0:
        return pd.DataFrame(columns=_TRADE_COLUMNS)
    state: dict[str, dict[str, float]] = {}
    out: list[dict[str, object]] = []
    for row in fills.sort_values("bar").to_dict("records"):
        ticker = str(row["ticker"])
        st = state.setdefault(ticker, {"qty": 0.0, "cost": 0.0, "fees": 0.0, "realized": 0.0,
                                       "entry_ts": None, "entry_bar": 0, "entry_px": 0.0,
                                       "bought": 0.0})
        px, amount, fee = float(row["price"]), float(row["amount"]), float(row["fee"])
        if row["side"] == "buy":
            # 从空仓开始买入时, 记下这笔回合的起点
            if st["qty"] == 0:
                st["entry_ts"], st["entry_bar"], st["entry_px"] = row["ts"], int(row["bar"]), px
                st["bought"] = 0.0
            st["qty"] += amount
            st["cost"] += amount * px + fee      # 买入费用摊进成本
            st["fees"] += fee
            st["bought"] += amount
        elif st["qty"] > 0:
            avg = st["cost"] / st["qty"]
            qty = min(amount, st["qty"])          # 卖出不超过当前持仓
            st["realized"] += qty * (px - avg) - fee
            st["cost"] -= qty * avg
            st["qty"] -= qty
            st["fees"] += fee
            if st["qty"] <= 1e-9:                 # 回到空仓 -> 结算一笔
                notional = st["bought"] * st["entry_px"]
                out.append({
                    "ticker": ticker, "entry_ts": st["entry_ts"], "exit_ts": row["ts"],
                    "qty": st["bought"], "entry_px": st["entry_px"], "exit_px": px,
                    "pnl": st["realized"],
                    "ret": st["realized"] / notional if notional else np.nan,
                    "fees": st["fees"], "bars": int(row["bar"]) - int(st["entry_bar"]),
                })
                state[ticker] = {"qty": 0.0, "cost": 0.0, "fees": 0.0, "realized": 0.0,
                                 "entry_ts": None, "entry_bar": 0, "entry_px": 0.0, "bought": 0.0}
    return pd.DataFrame(out, columns=_TRADE_COLUMNS)


def execution_stats(orders: pd.DataFrame, fills: pd.DataFrame | None = None,
                    nav: pd.Series | None = None) -> pd.Series:
    """委托执行情况: 执行率、换手与费用占比(回答"信号为什么没成交")。

    执行率是**数量口径**: 全部委托的已成交数量 / 委托数量; 部分成交按实际成交量算。
    换手是双边成交额 / 平均净值; 费用占比 = 总费用 / 平均净值。换手与费用占比需要
    fills(和 nav), 没有就记 NaN / 0。

    Args:
        orders: BacktestResult.orders(至少含 status / amount / filled)。
        fills: BacktestResult.fills; 用来算换手与总费用。
        nav: 逐 bar 净值; 用它的均值做分母。

    Returns:
        pd.Series: orders / filled_orders / rejected_orders / cancelled_orders /
        execution_rate / turnover / fees / cost_ratio; 没有委托时计数为 0,
        其余为 NaN。
    """
    out = {
        "orders": 0, "filled_orders": 0, "rejected_orders": 0, "cancelled_orders": 0,
        "execution_rate": np.nan, "turnover": np.nan, "fees": 0.0, "cost_ratio": np.nan,
    }
    if orders is None or len(orders) == 0:
        return pd.Series(out)

    status = orders["status"].astype(str)
    amount = orders["amount"].astype(float).abs()
    filled = orders["filled"].astype(float).abs()
    total = float(amount.sum())
    out["orders"] = len(orders)
    out["filled_orders"] = int((status == FILLED).sum())
    out["rejected_orders"] = int((status == REJECTED).sum())
    out["cancelled_orders"] = int((status == CANCELLED).sum())
    out["execution_rate"] = float(filled.sum() / total) if total > 0 else np.nan

    if fills is not None and len(fills):
        out["fees"] = float(fills["fee"].sum())
    if nav is not None:
        clean = nav.dropna()
        average = float(clean.mean()) if len(clean) else 0.0
        if average > 0:
            if fills is not None and len(fills):
                notional = float((fills["amount"].astype(float).abs()
                                  * fills["price"].astype(float)).sum())
                out["turnover"] = notional / average
            out["cost_ratio"] = out["fees"] / average
    return pd.Series(out)


def reject_reasons(orders: pd.DataFrame) -> pd.Series:
    """被拒订单按原因计数, 从多到少; 没有拒单时返回空 Series。

    Args:
        orders: BacktestResult.orders。

    Returns:
        pd.Series, index 是 reason, 值是笔数。
    """
    if orders is None or len(orders) == 0:
        return pd.Series(dtype=int)
    rejected = orders[orders["status"].astype(str) == REJECTED]
    if rejected.empty:
        return pd.Series(dtype=int)
    counts = rejected["reason"].fillna("").replace("", "未注明").value_counts()
    counts.name = "orders"
    return counts


def trade_stats(t: pd.DataFrame) -> pd.Series:
    """回合交易的汇总。

    Args:
        t: trades 的输出。

    Returns:
        pd.Series, 含 trades(笔数) / win_rate / profit_factor / avg_pnl /
        avg_ret / avg_bars / fees; 没有回合时笔数为 0, 其余为 NaN。
        profit_factor = 盈利总额 / 亏损总额(绝对值); 没有亏损时为 NaN。
    """
    if t is None or len(t) == 0:
        return pd.Series({"trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
                          "avg_pnl": np.nan, "avg_ret": np.nan, "avg_bars": np.nan,
                          "fees": 0.0})
    wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    gross_loss = abs(float(losses["pnl"].sum()))
    return pd.Series({
        "trades": len(t),
        "win_rate": len(wins) / len(t),
        "profit_factor": (float(wins["pnl"].sum()) / gross_loss) if gross_loss > 0 else np.nan,
        "avg_pnl": float(t["pnl"].mean()),
        "avg_ret": float(t["ret"].mean()),
        "avg_bars": float(t["bars"].mean()),
        "fees": float(t["fees"].sum()),
    })
