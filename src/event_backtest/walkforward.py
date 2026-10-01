"""Walk-forward: 滚动地"用过去训练 → 紧邻未来检验", 拼出一条样本外净值。

设计动机(minievent design.md §8.2 已定格式):
    在一整段历史上跑出的漂亮结果都是 in-sample。walk-forward 把历史切成若干折,
    每折在训练段上决定"这一段用什么参数", 再在紧接着的、没有参与决定的测试段上
    检验; 各折测试段拼起来, 就是一条没被参数选择污染过的样本外净值。

关于"训练步骤"的诚实说明:
    声明式策略本身没有可拟合的东西, 所以训练步骤就是**参数选择器 select**:
    输入训练段行情, 输出这一段要用的 StrategySpec。本次先把接口留出来, 默认
    None = 每折沿用同一份策略 —— 此时 walk-forward 是"分折样本外评估", 不是真正
    的参数优化; 等扫参做好, 把扫参当成 select 传进来即可。

三条正确性约束(逐条落在实现里):
    1. 参数污染: select 只拿得到本折**训练段**(train_bars)的行情, embargo 与测试段
       的数据绝不参与选参数。
    2. 样本重复: 各折测试段首尾相接、互不重叠; 训练与测试之间隔 embargo_days。
    3. 成本: 每折用同一套 fees / slippage; OOS 净值按每折收益连乘拼接(每折独立起始
       资金, 不跨折持仓)。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from event_backtest.config import MarketConfig
from event_backtest.engine import BacktestResult
from event_backtest.engine import run as run_backtest
from event_backtest.market import MarketData, slice_market
from event_backtest.strategy.declarative import (
    DeclarativeStrategy,
    StrategySpec,
    WalkForward,
)

__all__ = [
    "FoldRange",
    "FoldResult",
    "WalkForwardError",
    "WalkForwardResult",
    "check_data",
    "fold_ranges",
    "run_walk_forward",
]

# 参数选择器: (策略, 训练段行情) -> 这一段要用的策略
Select = Callable[[StrategySpec, MarketData], StrategySpec]


class WalkForwardError(ValueError):
    """walk-forward 参数或数据有问题。"""


@dataclass(frozen=True)
class FoldRange:
    """一折的时间与 bar 范围(bar 范围是全局下标 [start, end))。"""

    index: int
    train_days: tuple[pd.Timestamp, pd.Timestamp]
    test_days: tuple[pd.Timestamp, pd.Timestamp]
    train_bars: tuple[int, int]
    test_bars: tuple[int, int]


@dataclass(eq=False)
class FoldResult:
    """一折的测试结果。"""

    fold: FoldRange
    result: BacktestResult


@dataclass(eq=False)
class WalkForwardResult:
    """walk-forward 汇总: 每折结果 + 拼接后的样本外净值。"""

    name: str
    market_name: str
    initial_cash: float
    market: MarketData
    folds: list[FoldResult]
    oos_nav: pd.Series

    def fold_table(self) -> pd.DataFrame:
        """每折一行: 训练 / 测试区间、bar 数、测试段收益、成交笔数。"""
        rows = []
        for fold_result in self.folds:
            nav = fold_result.result.nav.dropna()
            total = float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) > 1 else float("nan")
            rows.append({
                "fold": fold_result.fold.index,
                "train": f"{fold_result.fold.train_days[0]:%Y-%m-%d}"
                         f"~{fold_result.fold.train_days[1]:%Y-%m-%d}",
                "test": f"{fold_result.fold.test_days[0]:%Y-%m-%d}"
                        f"~{fold_result.fold.test_days[1]:%Y-%m-%d}",
                "bars": len(nav),
                "test_return": total,
                "trades": len(fold_result.result.fills),
            })
        return pd.DataFrame(rows)


def check_data(wf: WalkForward, market: MarketData, *, name: str = "") -> None:
    """检查行情够不够长(对齐 minievent 的 check_data)。

    Args:
        wf: walk-forward 参数。
        market: 行情。
        name: 报错信息里用的策略名。

    Raises:
        WalkForwardError: 行情为空、start 早于数据起点, 或从 start 起交易日不足。
    """
    days = pd.DatetimeIndex(market.ts).normalize().unique()
    if len(days) == 0:
        raise WalkForwardError(f"{name}: 行情是空的, 做不了 walk-forward")
    # start=None 表示从数据第一个交易日开始
    if wf.start is None:
        have, since = len(days), days[0]
    else:
        start = pd.Timestamp(wf.start)
        if start < days[0]:
            raise WalkForwardError(
                f"{name}: walk_forward.start = {wf.start} 早于数据起点 {days[0].date()}")
        have, since = int((days >= start).sum()), start
    if have < wf.required_days:
        raise WalkForwardError(
            f"{name}: walk_forward 需要 {wf.required_days} 个交易日 (train {wf.train_days} "
            f"+ embargo {wf.embargo_days} + {wf.folds} x test {wf.test_days}), "
            f"从 {since.date()} 到数据终点 {days[-1].date()} 只有 {have} 个")


def fold_ranges(wf: WalkForward, market: MarketData) -> list[FoldRange]:
    """算出每一折的训练 / 测试范围。纯函数, 便于单测。

    第 k 折: 训练 [start + k*test_days, + train_days) -> 隔离 embargo_days ->
    测试 test_days; 测试段首尾相接、互不重叠。数据不够切满时少切几折。

    Args:
        wf: walk-forward 参数。
        market: 行情(用它的时间轴)。

    Returns:
        list[FoldRange], 按折序号升序。
    """
    day_of_bar = pd.DatetimeIndex(market.ts).normalize()
    days = pd.DatetimeIndex(day_of_bar.unique()).sort_values()
    i0 = (0 if wf.start is None else int(
        np.searchsorted(days.values, pd.Timestamp(wf.start).to_datetime64(), side="left")))
    out: list[FoldRange] = []
    for k in range(wf.folds):
        train_start = i0 + k * wf.test_days
        train_end = train_start + wf.train_days
        test_start = train_end + wf.embargo_days
        test_end = test_start + wf.test_days
        if test_end > len(days):
            break
        out.append(FoldRange(
            index=k,
            train_days=(days[train_start], days[train_end - 1]),
            test_days=(days[test_start], days[test_end - 1]),
            train_bars=_bar_span(day_of_bar, days, train_start, train_end),
            test_bars=_bar_span(day_of_bar, days, test_start, test_end),
        ))
    return out


def _bar_span(day_of_bar: pd.DatetimeIndex, days: pd.DatetimeIndex, day_start: int,
              day_end: int) -> tuple[int, int]:
    """交易日下标区间 [day_start, day_end) -> bar 下标区间 [a, b)。"""
    total = len(day_of_bar)
    a = int(np.searchsorted(day_of_bar.values, days[day_start].to_datetime64(), side="left"))
    if day_end >= len(days):
        return a, total
    b = int(np.searchsorted(day_of_bar.values, days[day_end].to_datetime64(), side="left"))
    return a, b


def run_walk_forward(cfg: MarketConfig, spec: StrategySpec, *, market: MarketData | None = None,
                     select: Select | None = None, start=None, end=None) -> WalkForwardResult:
    """跑一遍 walk-forward, 返回每折结果与拼接后的样本外净值。

    Args:
        cfg: 市场配置(费率 / 规则 / 滑点 / 初始资金 / 数据库都来自它)。
        spec: 策略; 必须带 walk_forward 段。
        market: 已加载好的行情; None 时用 cfg.load(start, end) 现加载。
        select: 参数选择器 (spec, train_market) -> spec; train_market 就是本折训练段
            [start + k*test_days, + train_days), 不含 embargo 与测试段。默认 None =
            每折沿用同一份策略。
        start / end: 加载行情用的日期范围(仅在 market 为 None 时生效)。

    Returns:
        WalkForwardResult。

    Raises:
        WalkForwardError: 策略没有 walk_forward 段, 或数据不够。
    """
    wf = spec.walk_forward
    if wf is None:
        raise WalkForwardError(f"{spec.name}: 策略没有 walk_forward 段")
    if market is None:
        market = cfg.load(start=start, end=end)
    check_data(wf, market, name=spec.name)
    ranges = fold_ranges(wf, market)
    if not ranges:
        raise WalkForwardError(f"{spec.name}: 按 walk_forward 切不出任何一折, 检查数据长度")

    folds: list[FoldResult] = []
    for fold in ranges:
        # 1. 选参数: 只把本折训练段交给选择器(滚动窗口, 不含 embargo / 测试段)。
        #    训练段开头因缺少 lookback 会有少量 NaN, 属于保守处理, 不构成未来函数。
        fold_spec = spec
        if select is not None:
            fold_spec = select(spec, slice_market(market, *fold.train_bars))
        # 2. 在"截至 test_end 的全部历史"上算事件(指标需要 lookback), 只保留测试段;
        #    事件用的是过去数据, 不构成未来函数 —— 参数只在第 1 步决定。
        extended = slice_market(market, 0, fold.test_bars[1])
        values = fold_spec.compute(extended)
        events = fold_spec.events(values, extended)
        offset, stop = fold.test_bars
        event_bars = _fold_event_bars(events, market.tickers, offset, stop - offset)
        # 3. 引擎只在测试段上跑(事件已注入, 不再现场重算)
        result = run_backtest(
            DeclarativeStrategy(fold_spec, event_bars=event_bars),
            slice_market(market, offset, stop),
            initial_cash=cfg.initial_cash, fees=cfg.resolved_fees(),
            fill_price=cfg.fill_price, slippage=cfg.slippage, rules=cfg.resolved_rules())
        folds.append(FoldResult(fold=fold, result=result))

    return WalkForwardResult(
        name=spec.name, market_name=cfg.market, initial_cash=cfg.initial_cash,
        market=market, folds=folds,
        oos_nav=_stitch([f.result.nav for f in folds], cfg.initial_cash))


def _fold_event_bars(events: pd.DataFrame, tickers: tuple[str, ...], offset: int,
                     length: int) -> list[set[int]]:
    """把事件表裁到测试段, 并平移到"相对本段行情"的 bar 下标。"""
    out: list[set[int]] = [set() for _ in tickers]
    index = {t: i for i, t in enumerate(tickers)}
    for row in events.to_dict("records"):
        bar = int(row["bar"]) - offset
        if 0 <= bar < length and str(row["ticker"]) in index:
            out[index[str(row["ticker"])]].add(bar)
    return out


def _stitch(navs: list[pd.Series], initial: float) -> pd.Series:
    """把各折净值的逐 bar 收益按顺序连乘, 拼成一条样本外净值。

    每折独立起始资金, 只把折内收益接起来 —— 不跨折持仓, 因此不必处理"上一折末尾
    的仓位带到下一折"; embargo 用来保证训练末尾开的仓不会影响测试段。
    """
    pieces = []
    for nav in navs:
        series = nav.dropna()
        if len(series):
            pieces.append(series.pct_change().fillna(0.0))
    if not pieces:
        return pd.Series(dtype=float)
    return initial * (1.0 + pd.concat(pieces)).cumprod()
