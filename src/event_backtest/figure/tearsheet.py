"""图表: tearsheet(单场景)与对比图(多场景)。

直接用 matplotlib 的 Figure + Agg canvas 建图, 不走 pyplot 后端, 所以在无显示环境
(脚本 / CI)里也能 savefig; 需要交互显示时调用方自己用 pyplot 接管。

图表里的文字一律用英文: 默认字体没有中文字形, 写中文会变成方块(见 minibacktest
README "图表文字用英文" 一节), 换机器也不会因此出问题。
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from event_backtest.benchmark import benchmark_nav
from event_backtest.engine import BacktestResult
from event_backtest.walkforward import WalkForwardResult

__all__ = ["plot_comparison", "plot_tearsheet", "plot_walk_forward"]

# 场景曲线的配色(按顺序循环取)
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f"]


def plot_tearsheet(result: BacktestResult, *, market: str = "us", benchmark: str | None = None,
                   title: str | None = None,
                   figsize: tuple[float, float] = (11, 7)) -> Figure:
    """画一张两栏 tearsheet: 上=净值(可叠加基准, 归一到同一起点), 下=回撤。

    Args:
        result: engine.run 的输出。
        market: "cn" / "us"; 目前只用于图例, 具体指标由 report.print_result 打印。
        benchmark: 基准证券代码; 给了就叠加一条买入持有净值。
        title: 图标题; None 时用"证券数 · 区间"自动生成。
        figsize: 图尺寸(英寸)。

    Returns:
        matplotlib Figure; 调用方自己 savefig / show。
    """
    nav = result.nav.dropna()
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)   # 挂一个 Agg canvas, savefig 无需显示环境
    ax_nav, ax_dd = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])

    # 上图: 策略净值; 有基准就叠加一条(起点与策略对齐)
    ax_nav.plot(nav.index, nav.to_numpy(), color="#1f77b4", lw=1.4, label=f"Strategy ({market})")
    bench = benchmark_nav(result, benchmark, initial=float(nav.iloc[0]) if len(nav) else None)
    if bench is not None:
        ax_nav.plot(bench.index, bench.to_numpy(), color="#999999", lw=1.1,
                    label=f"Benchmark {benchmark}")
    ax_nav.set_ylabel("NAV")
    ax_nav.legend(loc="upper left")
    ax_nav.grid(alpha=0.3)

    # 下图: 回撤(相对历史高点的跌幅)
    drawdown = nav / nav.cummax() - 1
    ax_dd.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color="#d62728", alpha=0.35)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.3)

    if title is None:
        span = f"{nav.index[0]:%Y-%m-%d} ~ {nav.index[-1]:%Y-%m-%d}" if len(nav) else ""
        title = f"{len(result.market.tickers)} securities · {span}"
    fig.suptitle(title)
    return fig


def plot_comparison(navs: Mapping[str, pd.Series], *, normalize: bool = True,
                    title: str = "Scenario comparison",
                    figsize: tuple[float, float] = (11, 6)) -> Figure:
    """把多个场景的净值曲线画在一张图上, 用来横向比较。

    Args:
        navs: {场景名: 净值序列}, 通常来自 compare_results 对应的那些回测。
        normalize: True 时每条曲线都归一到自己的起点(=1), 方便看相对表现;
            False 时按原始净值画。
        title: 图标题。
        figsize: 图尺寸(英寸)。

    Returns:
        matplotlib Figure; 调用方自己 savefig / show。
    """
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for i, (name, nav) in enumerate(navs.items()):
        series = nav.dropna()
        if series.empty:
            continue
        values = series / float(series.iloc[0]) if normalize else series
        ax.plot(series.index, values.to_numpy(), lw=1.3, color=_COLORS[i % len(_COLORS)],
                label=str(name))
    ax.set_ylabel("Net value (start = 1)" if normalize else "NAV")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    return fig


def plot_walk_forward(result: WalkForwardResult, *, title: str | None = None,
                      figsize: tuple[float, float] = (11, 7)) -> Figure:
    """walk-forward 的样本外净值: 拼接曲线 + 折边界, 下面配回撤。

    Args:
        result: walkforward.run_walk_forward 的输出。
        title: 图标题; None 时用"策略名 · walk-forward OOS"。
        figsize: 图尺寸(英寸)。

    Returns:
        matplotlib Figure; 调用方自己 savefig / show。
    """
    oos = result.oos_nav.dropna()
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    ax_nav, ax_dd = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])

    ax_nav.plot(oos.index, oos.to_numpy(), color="#1f77b4", lw=1.4, label="Walk-forward OOS")
    # 每折测试段: 交替浅底纹 + 起点竖虚线, 方便看清折的边界
    for k, fold_result in enumerate(result.folds):
        start, end = fold_result.fold.test_days
        ax_nav.axvspan(start, end, color="#1f77b4", alpha=0.05 if k % 2 == 0 else 0.10)
        ax_nav.axvline(start, color="#999999", lw=0.8, ls="--")
    ax_nav.set_ylabel("Net value")
    ax_nav.legend(loc="upper left")
    ax_nav.grid(alpha=0.3)

    drawdown = oos / oos.cummax() - 1
    ax_dd.fill_between(drawdown.index, drawdown.to_numpy(), 0.0, color="#d62728", alpha=0.35)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.3)

    # 默认标题用英文: result.name 是中文, 直接放图上会变成方块
    fig.suptitle(title or f"walk-forward OOS · {len(result.folds)} folds")
    return fig
