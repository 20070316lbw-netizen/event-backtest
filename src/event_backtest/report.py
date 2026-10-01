"""用 rich 把回测结果打印成分组中文表格, 以及多场景对比表。

对应 minibacktest 的 report.print_result: 不打印乱糟糟的 dataclass repr, 而是按
"概览 / 收益 / 风险 / 风险调整 / 交易"分组, 每组一张小表。

多场景对比是这套用法的核心: 一份公共配置派生多个变体(比如 无滑点 / 有滑点),
用 compare_results 汇总成一张表, print_comparison 打印。
"""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping

import pandas as pd
from rich.console import Console
from rich.table import Table

from event_backtest.benchmark import benchmark_nav
from event_backtest.engine import BacktestResult
from event_backtest.metrics import execution_stats, reject_reasons, summarize, trade_stats, trades

# (分组标题, [(字段名, 中文标签), ...]); 字段取不到就跳过那一行。
# 前半段来自 metrics.summarize, 后半段来自 trade_stats。
_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("概览", [("start", "起始日期"), ("end", "截止日期"), ("bars", "bar 数"),
             ("days", "交易日数")]),
    ("收益", [("total_return", "总收益率"), ("annual_return", "年化收益率"),
             ("benchmark_return", "基准收益率"), ("excess_return", "超额收益")]),
    ("风险", [("annual_vol", "年化波动率"), ("max_drawdown", "最大回撤"),
             ("max_drawdown_days", "最长回撤 [交易日]"), ("beta", "Beta")]),
    ("风险调整后收益", [("sharpe", "Sharpe"), ("sortino", "Sortino"), ("calmar", "Calmar")]),
    ("交易", [("trades", "回合数"), ("win_rate", "胜率"), ("profit_factor", "盈亏比"),
             ("avg_pnl", "平均每笔盈亏"), ("avg_ret", "平均每笔收益率"),
             ("avg_bars", "平均持有 bar"), ("fees", "总费用")]),
    ("执行", [("orders", "委托笔数"), ("filled_orders", "完全成交笔数"),
             ("rejected_orders", "拒单笔数"), ("cancelled_orders", "期末未成交"),
             ("execution_rate", "执行率 [数量]"), ("turnover", "换手 [双边/平均净值]"),
             ("cost_ratio", "费用 / 平均净值")]),
]

# 对比表展示的列(顺序即表格列顺序); 只留最关键的几列, 免得终端里被截断
_COMPARE_COLUMNS: list[tuple[str, str]] = [
    ("total_return", "总收益"), ("annual_return", "年化"), ("sharpe", "夏普"),
    ("max_drawdown", "最大回撤"), ("trades", "回合"), ("fees", "费用"),
]

# 这几类字段的显示格式
_PCT = {"total_return", "annual_return", "annual_vol", "max_drawdown",
        "benchmark_return", "excess_return", "win_rate", "avg_ret",
        "execution_rate", "cost_ratio"}
_MONEY = {"avg_pnl", "fees"}
_COUNT = {"bars", "days", "trades", "max_drawdown_days", "avg_bars",
          "orders", "filled_orders", "rejected_orders", "cancelled_orders"}


def result_values(result: BacktestResult, *, market: str = "us",
                  benchmark: str | None = None) -> dict[str, object]:
    """回测结果 -> 报告要用的全部字段(组合指标 + 交易统计 + 执行情况)。

    Args:
        result: engine.run 的输出。
        market: "cn" / "us", 决定年化交易日数(A 股 242, 美股 252)。
        benchmark: 基准证券代码; 给了且该证券在本次行情里, 就多算基准 / 超额 / Beta。

    Returns:
        {字段名: 值}; 与 result_sections 同源。
    """
    summary = summarize(result.nav, market=market,
                        benchmark_nav=benchmark_nav(result, benchmark))
    stats = trade_stats(trades(result.fills))
    execution = execution_stats(result.orders, result.fills, result.nav)
    return {**summary.to_dict(), **stats.to_dict(), **execution.to_dict()}


def result_sections(result: BacktestResult, *, market: str = "us",
                    benchmark: str | None = None) -> list[tuple[str, list[tuple[str, str]]]]:
    """回测结果 -> [(分组标题, [(中文标签, 格式化后的值)])]。

    终端(print_result)与 HTML 报告(figure.html)共用这一份, 保证两边口径一致;
    某组一个字段都取不到时该组是空列表。

    Args:
        result: engine.run 的输出。
        market: "cn" / "us", 年化口径。
        benchmark: 基准证券代码。

    Returns:
        [(标题, [(标签, 值文本)])], 顺序同报告里的分组顺序。
    """
    values = result_values(result, market=market, benchmark=benchmark)
    return [(title, [(label, format_value(key, values[key])) for key, label in items
                     if key in values and values[key] is not None])
            for title, items in _SECTIONS]


def print_result(result: BacktestResult, *, market: str = "us", benchmark: str | None = None,
                 console: Console | None = None) -> None:
    """把单个回测结果打印成分组中文表格。

    Args:
        result: engine.run 的输出。
        market: "cn" / "us", 决定年化交易日数(A 股 242, 美股 252)。
        benchmark: 基准证券代码; 给了且该证券在本次行情里, 就多打印基准 / 超额 / Beta。
        console: 复用一个已有 rich Console(比如配置过主题), 默认新建。
    """
    console = console or Console()
    for i, (title, rows) in enumerate(result_sections(result, market=market,
                                                      benchmark=benchmark)):
        if not rows:
            continue
        if i:
            console.print()   # 组间空一行
        table = Table(title=title, show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for label, text in rows:
            table.add_row(label, text)
        console.print(table)

    # 有拒单就把原因列出来, 直接回答"这个信号为什么没成交"
    reasons = reject_reasons(result.orders)
    if len(reasons):
        console.print()
        reasons_table = Table(title="拒单原因", box=None, padding=(0, 2, 0, 0))
        reasons_table.add_column("原因", style="bold cyan", no_wrap=True)
        reasons_table.add_column("笔数", justify="right")
        for reason, count in reasons.items():
            reasons_table.add_row(str(reason), str(int(count)))
        console.print(reasons_table)


def compare_results(results: Mapping[str, BacktestResult], *, market: str = "us",
                    benchmark: str | None = None) -> pd.DataFrame:
    """把多个场景的回测结果汇总成一张表(每行一个场景)。

    Args:
        results: {场景名: BacktestResult}, 通常来自同一份公共配置派生的多个变体。
        market: "cn" / "us", 年化口径。
        benchmark: 基准证券代码; 给了每个场景都会算基准收益 / 超额 / Beta。

    Returns:
        DataFrame, index 是场景名, 列是 metrics.summarize 与 trade_stats 的字段。
    """
    rows: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        summary = summarize(result.nav, market=market,
                            benchmark_nav=benchmark_nav(result, benchmark))
        stats = trade_stats(trades(result.fills))
        execution = execution_stats(result.orders, result.fills, result.nav)
        rows[str(name)] = {**summary.to_dict(), **stats.to_dict(), **execution.to_dict()}
    return pd.DataFrame.from_dict(rows, orient="index")


def print_comparison(table: pd.DataFrame, *, console: Console | None = None,
                     title: str = "场景对比") -> None:
    """把 compare_results 的表打印成 rich 表格: 行=场景, 列=关键指标。

    Args:
        table: compare_results 的输出。
        console: 复用一个已有 rich Console, 默认新建。
        title: 表格标题。
    """
    console = console or Console()
    columns = [(key, label) for key, label in _COMPARE_COLUMNS if key in table.columns]
    rich_table = Table(title=title)
    rich_table.add_column("场景", style="bold cyan", no_wrap=True)
    for _, label in columns:
        rich_table.add_column(label, justify="right")
    for name, row in table.iterrows():
        rich_table.add_row(str(name), *[format_value(key, row[key]) for key, _ in columns])
    console.print(rich_table)


def print_walk_forward(result: object, *, console: Console | None = None,
                       title: str = "walk-forward") -> None:
    """打印 walk-forward: 每折测试段一行 + 拼接后的样本外指标。

    Args:
        result: walkforward.run_walk_forward 的 WalkForwardResult。
        console: 复用一个已有 rich Console, 默认新建。
        title: 表标题。
    """
    console = console or Console()
    fold_table = result.fold_table()

    table = Table(title=f"{result.name} · {title} 各折")
    for column, justify in (("折", "right"), ("训练", "left"), ("测试", "left"),
                            ("bar", "right"), ("测试收益", "right"), ("成交", "right")):
        table.add_column(column, justify=justify)
    for _, row in fold_table.iterrows():
        table.add_row(str(row["fold"]), str(row["train"]), str(row["test"]),
                      str(row["bars"]), format_value("test_return", row["test_return"]),
                      str(row["trades"]))
    console.print(table)

    # 拼接后的样本外净值: 只挑关键指标, 复用 format_value 的格式规则
    summary = summarize(result.oos_nav, market=result.market_name)
    console.print()
    oos = Table(title=f"{result.name} · 拼接样本外", show_header=False, box=None,
                padding=(0, 2, 0, 0))
    oos.add_column(style="bold cyan", no_wrap=True)
    oos.add_column()
    for key, label in (("total_return", "总收益率"), ("annual_return", "年化收益率"),
                       ("annual_vol", "年化波动率"), ("sharpe", "Sharpe"),
                       ("max_drawdown", "最大回撤")):
        oos.add_row(label, format_value(key, summary[key]))
    console.print(oos)


def format_value(key: str, value: object) -> str:
    """按字段类型格式化: 比例 -> 百分数, 金额 -> 千分位, 计数 -> 整数, 日期 -> 年月日。

    Args:
        key: 字段名(决定格式规则)。
        value: 字段值。

    Returns:
        终端 / HTML 表格里显示的文本。
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if key in _PCT:
        return f"{float(value):.2%}"
    if key in _MONEY:
        return f"{float(value):,.2f}"
    if key in _COUNT:
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)
