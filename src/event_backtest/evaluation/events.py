"""事件研究: 信号触发后 k 根 bar 的收益, 以及与同时段无条件基线的差。

口径(与 minievent 对齐):
    - 第 k 根累计收益 = adj_close[t+k] / adj_close[t] - 1, 起点是信号 bar 收盘;
    - 基线 = 同一证券、同一"结束时刻"(time-of-day)的全部有效窗口均值, 含事件本身;
      这是描述性的无条件基准, 不代表样本外超额;
    - 每个 horizon 输出事件数、有效样本数、均值、中位数、胜率、基线与超额;
    - 显著性: 每个 horizon 单独做单样本 t 检验(零假设"均值 = 0"), 并给 bootstrap
      百分位区间与零假设下的重抽 p 值。

怎么检验(默认口径, 见 tests 表的列):
    - 检验对象默认是**超额收益**(事件收益 - 同时段无条件基线), 问的是"信号相对
      同时段基准有没有增量"; test_on="ret" 则检验原始收益(多半只是在检验市场漂移)。
    - **按交易日聚类**: 同一天 (尤其同一根 bar) 触发多个事件时它们并不独立, 直接对
      事件做 t 检验会高估显著性。默认先把同一天的事件压成一个日均值, 再对日均值做
      单样本 t 检验; cluster="none" 才是事件级 iid 检验。
    - bootstrap 对日均值有放回重抽: 区间取重抽均值的百分位, p 值把样本平移到均值 0
      下再重抽。
    - 每个 horizon 各检各的, **没有做多重比较校正**: 16 个 horizon 一起看时, 单看某
      一行"显著"要打个折扣; 要更严格自己按 horizon 数做 Bonferroni / BH。
    - 有效样本太少(聚类后不足 2 个, 或标准差为 0)时统计量是 NaN, 不硬算。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from event_backtest.evaluation.stats import bootstrap_mean, mean_t_test, stars
from event_backtest.market import MarketData

__all__ = ["DEFAULT_HORIZONS", "EventStudy", "study_events"]

# tests 表的列
_TEST_COLUMNS = ["horizon", "n", "n_obs", "n_days", "mean", "t", "p",
                 "ci_low", "ci_high", "p_boot", "stars"]

# 默认观察窗口: 信号后 1..16 根 bar(30 分钟线下正好两天)
DEFAULT_HORIZONS = tuple(range(1, 17))
_DEFAULT_HORIZONS = DEFAULT_HORIZONS


@dataclass(frozen=True, eq=False)
class EventStudy:
    """事件研究结果。

    Attributes:
        events: 输入的事件表。
        paths: 逐事件逐 horizon 的路径, 列
            [strategy, event_id, ticker, bar, ts, horizon, ret, baseline_ret, excess_ret]。
        summary: 每个 horizon 一行的汇总, 列
            [horizon, n_events, n, mean_ret, median_ret, win_rate, baseline, excess]。
        tests: 每个 horizon 一行的显著性检验, 列
            [horizon, n, n_obs, n_days, mean, t, p, ci_low, ci_high, p_boot, stars]。
            n 是有效事件窗口数, n_obs 是实际进入检验的观测数(按日聚类时 = 天数),
            n_days 是涉及的不同交易日数; p 来自 t 检验, p_boot 来自 bootstrap。
    """

    events: pd.DataFrame
    paths: pd.DataFrame
    summary: pd.DataFrame
    tests: pd.DataFrame

    def tests_table(self) -> pd.DataFrame:
        """tests 的展示版: 均值 / 区间转百分数, t 与 p 保留 4 位小数。

        Returns:
            DataFrame, 列同 tests, 但数值都格式化成了字符串; 非有限的显示为 "-"。
        """
        if self.tests.empty:
            return self.tests.copy()
        out = self.tests.copy()
        for column in ("mean", "ci_low", "ci_high"):
            out[column] = out[column].map(_pct)
        for column in ("t", "p", "p_boot"):
            out[column] = out[column].map(_num4)
        return out


def study_events(market: MarketData, events: pd.DataFrame, *,
                 horizons: tuple[int, ...] = _DEFAULT_HORIZONS,
                 strategy_name: str = "",
                 test_on: str = "excess",
                 cluster: str = "day",
                 bootstrap: int = 1000,
                 alpha: float = 0.05,
                 seed: int = 0) -> EventStudy:
    """对事件表做事件后收益研究, 并给出显著性检验与 bootstrap 区间。

    Args:
        market: 对齐后的行情(用后复权收盘价算收益)。
        events: 事件表, 至少含 [event_id, bar, ts, ticker]。
        horizons: 观察窗口(bar 数), 默认 1..16。
        strategy_name: 写进结果第一列, 便于多策略拼接。
        test_on: 检验对象, "excess"(默认, 超额收益)或 "ret"(原始收益)。
        cluster: "day"(默认)按交易日聚类, 同一天多个事件先压成日均值再检验;
            "none" 用事件级 iid 检验(同日多事件时会高估显著性)。
        bootstrap: bootstrap 重抽次数; 0 表示不算(区间与 p_boot 为 NaN)。
        alpha: 置信水平, 区间是 1 - alpha 的百分位区间。
        seed: bootstrap 随机种子, 固定它结果可复现。

    Returns:
        EventStudy; 事件为空时返回空 paths 与带列名的空 summary / tests。

    Raises:
        ValueError: test_on / cluster 取值不对, 或 alpha 不在 (0, 1) 内。
    """
    if test_on not in ("excess", "ret"):
        raise ValueError(f"test_on 只能是 'excess' / 'ret', 收到 {test_on!r}")
    if cluster not in ("day", "none"):
        raise ValueError(f"cluster 只能是 'day' / 'none', 收到 {cluster!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 要在 (0, 1) 内, 收到 {alpha!r}")
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
    tests = _tests(paths, test_on=test_on, cluster=cluster, n_boot=bootstrap,
                   alpha=alpha, seed=seed)
    return EventStudy(events=events, paths=paths, summary=_summary(paths), tests=tests)


def _tests(paths: pd.DataFrame, *, test_on: str, cluster: str, n_boot: int,
           alpha: float, seed: int) -> pd.DataFrame:
    """按 horizon 做显著性检验: 单样本 t 检验 + bootstrap 区间。

    按日聚类时先把同一天的多个事件压成日均值, 再对日均值检验(见模块 docstring)。
    """
    if paths.empty:
        return pd.DataFrame(columns=_TEST_COLUMNS)
    column = "ret" if test_on == "ret" else "excess_ret"
    rows: list[dict[str, object]] = []
    for horizon, group in paths.groupby("horizon", sort=True):
        valid = group.dropna(subset=[column])
        values = valid[column].to_numpy(dtype=float)
        days = pd.DatetimeIndex(valid["ts"]).normalize()
        if cluster == "day" and len(days):
            observations = pd.Series(values, index=days).groupby(level=0).mean().to_numpy()
        else:
            observations = values
        mean, t, p, n_obs = mean_t_test(observations)
        ci_low, ci_high, p_boot = bootstrap_mean(observations, n_boot=n_boot,
                                                 alpha=alpha, seed=seed)
        rows.append({
            "horizon": int(horizon),
            "n": len(values),
            "n_obs": n_obs,
            "n_days": len(days.unique()),
            "mean": mean,
            "t": t,
            "p": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_boot": p_boot,
            "stars": stars(p),
        })
    return pd.DataFrame(rows, columns=_TEST_COLUMNS)


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


def _pct(value: float) -> str:
    """比例 -> 百分数文本; 非有限给 "-"。"""
    return f"{value:.2%}" if np.isfinite(value) else "-"


def _num4(value: float) -> str:
    """统计量 -> 4 位小数文本; 非有限给 "-"。"""
    return f"{value:.4f}" if np.isfinite(value) else "-"
