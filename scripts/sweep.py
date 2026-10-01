"""参数扫描(最小版): 一份公共策略 + 一个参数网格, 出排行表 + 热力图。

网格用"策略 yaml 里的点路径"表示, 行情只加载一次。扫描/打补丁的实现在
event_backtest.sweep(只有几个函数, 不是框架), 这个脚本负责参数区和展示。

用法: uv run python scripts/sweep.py

注意: 这是在**同一段数据**上挑最好的一组, 属于样本内, 不能直接当可交易结论; 要样本外
验证, 用 scripts/run_walk_forward.py 把同一套网格当 select。
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table

from event_backtest import load_config
from event_backtest.strategy import STRATEGY_DIR
from event_backtest.sweep import search

# ---------------------------------------------------------------- 参数区(改这里)
MARKET = "us"
DB = None
FREQ = None
STRATEGY = "volume_breakout"
START = "2024-01-01"
END = "2024-12-31"
OUTPUT = "outputs/sweep"
SORT_BY = "sharpe"        # 排行 / 选优依据(结果表的列名)
TOP = 15                  # 终端只打印前几名

# 参数网格: 轴名是策略 yaml 里的点路径([i] = 列表第 i 项), 值是该轴的候选取值。
GRID: dict[str, list[object]] = {
    "factors[0].relative_volume.window": [8, 16, 32],
    "signal.all[0].value": [1.5, 2.0, 3.0],
    "exit.hold_bars": [8, 16, 32],
}
# ----------------------------------------------------------------

_PCT = {"total_return", "annual_return", "max_drawdown"}


def short_name(path: str) -> str:
    """点路径的显示名: 只取最后一段(如 exit.hold_bars -> hold_bars)。"""
    return path.split(".")[-1]


def format_value(key: str, value: object) -> str:
    """表格里的一格: 比例转百分数, 其余两位小数。"""
    if isinstance(value, float):
        return f"{value:.2%}" if key in _PCT else f"{value:,.2f}"
    return str(value)


def print_table(table: pd.DataFrame, axes: list[str]) -> None:
    """rich 打印排行表(前 TOP 名); 轴列用短名。"""
    display = table.rename(columns={path: short_name(path) for path in axes})
    console = Console()
    rich_table = Table(title=f"参数扫描 · 按 {SORT_BY} 排序 (共 {len(table)} 组, 前 {TOP})")
    for column in display.columns:
        rich_table.add_column(str(column), justify="right")
    for _, row in display.head(TOP).iterrows():
        rich_table.add_row(*[format_value(str(c), row[c]) for c in display.columns])
    console.print(rich_table)


def heatmaps(table: pd.DataFrame, axes: list[str], metric: str) -> Figure:
    """每两个轴画一张热力图(其余轴取均值): 平原 = 稳健, 孤峰 = 可能在拟合噪声。"""
    pairs = list(itertools.combinations(range(len(axes)), 2))
    fig = Figure(figsize=(5.0 * len(pairs), 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    subplots = fig.subplots(1, len(pairs))
    panels = [subplots] if len(pairs) == 1 else list(subplots)
    for ax, (i, j) in zip(panels, pairs, strict=True):
        pivot = table.pivot_table(index=axes[i], columns=axes[j], values=metric, aggfunc="mean")
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
        ax.set_xlabel(short_name(axes[j]))
        ax.set_ylabel(short_name(axes[i]))
        ax.set_title(f"{metric}: {short_name(axes[i])} x {short_name(axes[j])}")
        fig.colorbar(image, ax=ax)
    return fig


def main() -> None:
    """跑网格, 打印排行, 写 results.csv 与(至少两参数时) heatmap.png。"""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(MARKET)
    if DB is not None:
        db = Path(DB)
        cfg = replace(cfg, db_path=db if db.is_absolute() else (repo / db).resolve())
    if FREQ is not None:
        cfg = replace(cfg, freq=FREQ)

    raw = yaml.safe_load((STRATEGY_DIR / f"{STRATEGY}.yaml").read_text(encoding="utf-8"))
    market = cfg.load(start=START, end=END)      # 行情只加载一次

    outcome = search(raw, GRID, cfg, market, metric=SORT_BY)
    table = outcome.table.sort_values(SORT_BY, ascending=False).reset_index(drop=True)
    axes = list(GRID)
    print_table(table, axes)
    print(f"\n最优参数: {outcome.best_params}")

    out = repo / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)
    if len(axes) >= 2:
        heatmaps(table, axes, SORT_BY).savefig(out / "heatmap.png", dpi=120)
    print(f"结果写入: {out}")


if __name__ == "__main__":
    main()
