"""参数扫描的最小工具: 点路径打补丁 + 网格搜索 + 接进 walk-forward 的 select。

不是通用框架, 只做三件事:
    set_path / expand_grid   把"策略 yaml 的点路径"网格展开, 打到 yaml 副本上;
    search                   在一段行情上把网格跑一遍, 返回结果表 + 最优 spec;
    make_grid_select         把 search 包成 walk-forward 的 select(只在训练段扫)。

研究正确性提醒: search 是在给它的那段行情上挑最好的一组, 属于样本内; 只有在
walk-forward 里把它当 select(训练段扫、测试段用)得到的净值才是样本外的。
"""
from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from event_backtest.benchmark import benchmark_nav
from event_backtest.config import MarketConfig
from event_backtest.engine import run as run_backtest
from event_backtest.factor import FACTOR_DIR
from event_backtest.market import MarketData
from event_backtest.metrics import summarize
from event_backtest.strategy.declarative import (
    DeclarativeStrategy,
    StrategySpec,
    parse_strategy,
)

__all__ = ["SweepResult", "expand_grid", "make_grid_select", "search", "set_path"]

Select = Callable[[StrategySpec, MarketData], StrategySpec]


def set_path(node: Any, path: str, value: Any) -> None:
    """把点路径(如 factors[0].relative_volume.window)设成 value, 就地改 node。

    路径语法只有三种: key、key.subkey、list[index]; 和策略 yaml 的层级一一对应。
    """
    tokens: list[Any] = []
    for part in path.split("."):
        name = part
        while "[" in name:
            head, rest = name.split("[", 1)
            if head:
                tokens.append(head)
            index, name = rest.split("]", 1)
            tokens.append(int(index))
        if name:
            tokens.append(name)
    cursor = node
    for token in tokens[:-1]:
        cursor = cursor[token]
    cursor[tokens[-1]] = value


def expand_grid(grid: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    """网格的笛卡尔积: 每个元素是 {点路径: 取值}。空网格返回 [{}](相当于不扫)。"""
    keys = list(grid)
    return [dict(zip(keys, values, strict=True))
            for values in itertools.product(*(grid[key] for key in keys))]


@dataclass(frozen=True)
class SweepResult:
    """一次网格搜索的结果。

    Attributes:
        table: 每行一个组合(点路径取值 + 指标)。
        best: metric 最优的那个 StrategySpec。
        best_params: 最优组合对应的 {点路径: 取值}。
    """

    table: pd.DataFrame
    best: StrategySpec
    best_params: dict[str, object]


def search(base_raw: Any, grid: Mapping[str, Sequence[object]], cfg: MarketConfig,
           market: MarketData, *, factor_dir: str | Path = FACTOR_DIR,
           metric: str = "sharpe") -> SweepResult:
    """在 market 上把 grid 跑一遍, 返回结果表与最优 spec。

    Args:
        base_raw: 基础策略的 yaml 内容(yaml.safe_load 的结果)。
        grid: {点路径: 候选取值}。
        cfg: 市场配置(费率 / 规则 / 滑点 / 初始资金)。
        market: 要评估的行情(训练段或整段)。
        factor_dir: 因子 yaml 目录。
        metric: 排序 / 选优依据, 必须是结果表的列名, 如 "sharpe" / "annual_return"。

    Returns:
        SweepResult。
    """
    rows: list[dict[str, object]] = []
    best_spec: StrategySpec | None = None
    best_params: dict[str, object] = {}
    best_value = float("-inf")
    for combo in expand_grid(grid):
        patched = copy.deepcopy(base_raw)
        for path, value in combo.items():
            set_path(patched, path, value)
        spec = parse_strategy(patched, factor_dir=factor_dir, where="sweep")
        metrics = evaluate_spec(spec, cfg, market)
        rows.append({**combo, **metrics})
        value = float(metrics[metric])
        # NaN 参与 > 比较永远为 False, 所以最优一定是有限值
        if value > best_value:
            best_value, best_spec, best_params = value, spec, combo
    if best_spec is None:      # 全部组合的 metric 都是 NaN
        best_spec = parse_strategy(copy.deepcopy(base_raw), factor_dir=factor_dir,
                                   where="sweep")
    return SweepResult(table=pd.DataFrame(rows), best=best_spec, best_params=best_params)


def evaluate_spec(spec: StrategySpec, cfg: MarketConfig,
                  market: MarketData) -> dict[str, object]:
    """跑一个 spec 并返回报告要用的几个指标。"""
    result = run_backtest(DeclarativeStrategy(spec), market, initial_cash=cfg.initial_cash,
                          fees=cfg.resolved_fees(), fill_price=cfg.fill_price,
                          slippage=cfg.slippage, rules=cfg.resolved_rules())
    summary = summarize(result.nav, market=cfg.market,
                        benchmark_nav=benchmark_nav(result, cfg.benchmark))
    return {"total_return": float(summary["total_return"]),
            "annual_return": float(summary["annual_return"]),
            "sharpe": float(summary["sharpe"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "trades": len(result.fills)}


def make_grid_select(base_raw: Any, grid: Mapping[str, Sequence[object]],
                     cfg: MarketConfig, *, factor_dir: str | Path = FACTOR_DIR,
                     metric: str = "sharpe",
                     record: list[dict[str, object]] | None = None) -> Select:
    """把 search 包成 walk-forward 的 select: 只在训练段扫, 返回最优 spec。

    Args:
        base_raw / grid / cfg / factor_dir / metric: 同 search。
        record: 可选; 每折选到的 best_params 会依次 append 进去, 用来事后看参数稳不稳。

    Returns:
        select(spec, train_market) -> spec; walk-forward 每折训练段调一次。
    """
    def select(spec: StrategySpec, train_market: MarketData) -> StrategySpec:
        # 打补丁是在 yaml 原始 dict 上做的, 所以这里用构造时的 base_raw,
        # 不用传进来的 spec(它本来就是 base_raw 解析出来的同一份策略)。
        outcome = search(base_raw, grid, cfg, train_market,
                         factor_dir=factor_dir, metric=metric)
        if record is not None:
            record.append(outcome.best_params)
        return outcome.best

    return select
