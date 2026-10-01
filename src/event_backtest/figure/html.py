"""自包含的交互式 HTML 报告: tearsheet(单场景) / 场景对比 / walk-forward。

设计:
    - **自包含**: 数据以 JSON 内联, 样式与脚本内联; 不引用 CDN, 断网也能打开,
      一个文件发给别人就能看。
    - **零依赖**: 图表是原生 JS + SVG(见 html_assets.py), 不需要 plotly/echarts。
    - **口径一致**: 指标分组表直接复用 report.result_sections, 和终端里打印的一模一样;
      订单 / 拒单 / 执行率复用 metrics, 与 orders.parquet / execution.csv 同源。

用法::

    from event_backtest.figure import plot_tearsheet_html
    plot_tearsheet_html(result, market="cn", benchmark=None).save("tearsheet.html")
"""
from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from event_backtest.benchmark import benchmark_nav
from event_backtest.engine import BacktestResult
from event_backtest.figure.html_assets import CSS, JS
from event_backtest.metrics import (
    daily_nav,
    execution_stats,
    reject_reasons,
    summarize,
    trade_stats,
    trades,
)
from event_backtest.report import format_value, result_sections
from event_backtest.walkforward import WalkForwardResult

__all__ = [
    "HtmlReport",
    "plot_comparison_html",
    "plot_tearsheet_html",
    "plot_walk_forward_html",
]

# 区间按钮(月): 只在数据跨度够长时才显示
_RANGE_MONTHS = (1, 3, 6, 12, 36, 60, 120)
# 表格默认最多渲染多少行(余下的只写进 note, 免得 HTML 太大)
MAX_ROWS = 2000

_HTML = """<!doctype html>
<html lang="zh-CN" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<main id="app"></main>
<script id="report-data" type="application/json">__DATA__</script>
<script>__JS__</script>
</body>
</html>
"""


@dataclass(frozen=True)
class HtmlReport:
    """一份自包含 HTML 报告。

    Attributes:
        html: 完整页面文本。
        kind: tearsheet / comparison / walk_forward。
    """

    html: str
    kind: str

    def save(self, path: str | Path) -> Path:
        """写到文件(自动建父目录), 返回写入的路径。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.html, encoding="utf-8")
        return target

    def __len__(self) -> int:
        """页面字符数, 便于日志里看大小。"""
        return len(self.html)


# ---------------------------------------------------------------- 数据整理

def _num(value: Any) -> float | None:
    """转成 JSON 能装的 float; NaN / inf / None 一律给 None。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _values(series: pd.Series) -> list[float | None]:
    """Series -> JSON 数组(NaN 变 None, 图表里就断线而不是画到 0)。"""
    return [_num(v) for v in pd.to_numeric(series, errors="coerce").to_numpy()]


def _labels(index: pd.Index) -> list[str]:
    """时间索引 -> 字符串标签; 分钟线带上时分。"""
    if len(index) == 0:
        return []
    stamps = pd.DatetimeIndex(index)
    intraday = bool(((stamps.hour != 0) | (stamps.minute != 0)).any())
    pattern = "%Y-%m-%dT%H:%M" if intraday else "%Y-%m-%d"
    return [stamp.strftime(pattern) for stamp in stamps]


def _ts_text(value: Any) -> str:
    """单个时间戳的表格写法: 零点只写日期, 否则带上时分。"""
    stamp = pd.Timestamp(value)
    if stamp.hour or stamp.minute:
        return stamp.strftime("%Y-%m-%d %H:%M")
    return stamp.strftime("%Y-%m-%d")


def _ranges(labels: list[str]) -> list[int]:
    """按数据跨度挑可用的区间按钮(月)。"""
    if len(labels) < 2:
        return []
    try:
        first, last = pd.Timestamp(labels[0]), pd.Timestamp(labels[-1])
    except ValueError:
        return []
    months = (last - first).days / 30.44
    return [m for m in _RANGE_MONTHS if m < months * 0.98]


def _json_dumps(payload: Any) -> str:
    """内联 JSON: 不允许 NaN, 并转义 < 与行分隔符, 免得提前结束 script 标签。"""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (text.replace("<", "\\u003c")
                .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _page(payload: dict[str, Any], title: str) -> HtmlReport:
    """套页面模板: 内联 CSS / JS / 数据。"""
    page = (_HTML.replace("__TITLE__", _html.escape(title))
                 .replace("__CSS__", CSS)
                 .replace("__DATA__", _json_dumps(payload))
                 .replace("__JS__", JS))
    return HtmlReport(html=page, kind=str(payload.get("kind", "report")))


def _subtitle(index: pd.Index, market: str) -> str:
    """页头副标题: 市场 + 区间 + bar 数 / 交易日数。"""
    if len(index) == 0:
        return f"{market} · 无数据"
    stamps = pd.DatetimeIndex(index)
    days = len(stamps.normalize().unique())
    return (f"{market} · {_ts_text(stamps[0])} ~ {_ts_text(stamps[-1])} · "
            f"{len(stamps)} bar / {days} 交易日")


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]], *,
           title: str, sortable: bool = False, sort_key: str | None = None,
           note: str | None = None) -> dict[str, Any]:
    """表格 JSON: columns 是 (key, 中文标签, 格式), rows 里超量的部分截断并注明。"""
    clipped = rows[:MAX_ROWS]
    table: dict[str, Any] = {
        "title": title,
        "columns": [{"key": k, "label": label, "kind": kind} for k, label, kind in columns],
        "rows": clipped,
        "sortable": sortable,
    }
    if sort_key:
        table["sortKey"] = sort_key
    if len(rows) > len(clipped):
        table["note"] = (f"共 {len(rows)} 行, 表里只渲染前 {len(clipped)} 行"
                         f"(完整数据看 parquet / csv)。")
    elif note:
        table["note"] = note
    return table


def _kpis(values: dict[str, Any], specs: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
    """(字段, 中文标签, 格式, 颜色规则) -> KPI 卡片; 缺字段的直接跳过。"""
    cards = []
    for key, label, kind, tone in specs:
        if key not in values or values[key] is None:
            continue
        value = values[key]
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = _num(value)
            if value is None:      # NaN / inf 的指标(比如没有下行波动时的 Sortino)不摆卡片
                continue
        if not isinstance(value, (int, float, str)):
            continue
        cards.append({"label": label, "kind": kind, "tone": tone, "value": value})
    return cards


def _monthly(nav: pd.Series) -> dict[str, Any] | None:
    """月度收益柱状图(日频净值先聚到月末再算收益)。"""
    daily = daily_nav(nav).dropna()
    if len(daily) < 3:
        return None
    monthly = daily.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return None
    return {
        "title": "月度收益",
        "type": "bar",
        "colorBySign": True,
        "labels": [stamp.strftime("%Y-%m") for stamp in monthly.index],
        "series": [{"name": "月度收益", "values": [_num(v) for v in monthly.to_numpy()]}],
        "valueKind": "pct1",
        "axisKind": "pct0",
        "height": 220,
    }


def _trade_hist(trade_frame: pd.DataFrame) -> dict[str, Any] | None:
    """回合收益率分布(±20% 截尾, 20 个等宽桶)。"""
    if trade_frame is None or len(trade_frame) < 3 or "ret" not in trade_frame:
        return None
    values = pd.to_numeric(trade_frame["ret"], errors="coerce").dropna().clip(-0.2, 0.2)
    if len(values) < 3:
        return None
    counts, edges = np.histogram(values.to_numpy(), bins=np.linspace(-0.2, 0.2, 21))
    return {
        "title": "回合收益分布",
        "type": "bar",
        "colorBySign": True,
        "labels": [f"{edges[i]:.0%}~{edges[i + 1]:.0%}" for i in range(len(counts))],
        "series": [{"name": "回合数", "values": [int(c) for c in counts]}],
        "valueKind": "int",
        "axisKind": "int",
        "height": 220,
    }


def _reject_rows(result: BacktestResult) -> list[list[Any]]:
    """拒单原因 -> [[原因, 笔数], ...]。"""
    counts = reject_reasons(result.orders)
    return [[str(reason), int(count)] for reason, count in counts.items()]


# ---------------------------------------------------------------- tearsheet

_TEARSHEET_KPIS = [
    ("total_return", "总收益率", "pct2", "auto"),
    ("annual_return", "年化收益率", "pct2", "auto"),
    ("sharpe", "Sharpe", "num2", "auto"),
    ("max_drawdown", "最大回撤", "pct2", "auto"),
    ("annual_vol", "年化波动率", "pct2", "none"),
    ("calmar", "Calmar", "num2", "auto"),
    ("trades", "回合数", "int", "none"),
    ("win_rate", "胜率", "pct1", "none"),
    ("profit_factor", "盈亏比", "num2", "none"),
    ("execution_rate", "执行率 [数量]", "pct1", "none"),
]


def plot_tearsheet_html(result: BacktestResult, *, market: str = "us",
                        benchmark: str | None = None, title: str | None = None,
                        generated: datetime | None = None) -> HtmlReport:
    """单场景 HTML tearsheet: KPI 卡 + 净值 / 回撤 / 月度收益 / 回合分布 + 明细表。

    Args:
        result: engine.run 的输出。
        market: "cn" / "us", 年化口径。
        benchmark: 基准代码; 给了且本次行情里有, 就多一条基准净值与基准 KPI。
        title: 页面标题。
        generated: 生成时间(默认现在), 便于测试固定。

    Returns:
        HtmlReport(自包含 HTML, kind="tearsheet")。
    """
    nav = result.nav
    values = {**summarize(nav, market=market,
                          benchmark_nav=benchmark_nav(result, benchmark)).to_dict(),
              **trade_stats(trades(result.fills)).to_dict(),
              **execution_stats(result.orders, result.fills, nav).to_dict()}
    labels = _labels(nav.index)

    series = [{"name": "策略", "values": _values(nav)}]
    bench = benchmark_nav(result, benchmark, initial=float(nav.iloc[0])) if benchmark else None
    if bench is not None:
        series.append({"name": f"基准 {benchmark}",
                       "values": _values(bench.reindex(nav.index))})

    # 净值在历史高点附近时除法会有 1e-17 级别的噪声, 那点"回撤"不是回撤, 归零
    drawdown = (nav / nav.cummax() - 1).mask(lambda s: s > -1e-9, 0.0)
    trade_frame = trades(result.fills)

    payload: dict[str, Any] = {
        "kind": "tearsheet",
        "title": title or "回测 tearsheet",
        "subtitle": _subtitle(nav.index, market),
        "generated": (generated or datetime.now()).strftime("%Y-%m-%d %H:%M"),
        "kpis": _kpis(values, _TEARSHEET_KPIS),
        "nav": {
            "title": "净值",
            "type": "line",
            "labels": labels,
            "series": series,
            "valueKind": "money",
            "axisKind": "money",
            "height": 320,
            "ranges": _ranges(labels),
        },
        "drawdown": {
            "title": "回撤",
            "type": "area",
            "labels": labels,
            "series": [{"name": "回撤", "values": _values(drawdown)}],
            "valueKind": "pct2",
            "axisKind": "pct0",
            "height": 190,
            "zeroBase": True,
            "forceZeroMax": True,
            "ranges": _ranges(labels),
        },
        "sections": [{"title": name, "rows": rows}
                     for name, rows in result_sections(result, market=market,
                                                       benchmark=benchmark)],
        "monthly": _monthly(nav),
        "trade_hist": _trade_hist(trade_frame),
        "reject_reasons": _reject_rows(result),
    }

    if len(trade_frame):
        rows = [{
            "ticker": str(row["ticker"]),
            "entry_ts": _ts_text(row["entry_ts"]),
            "exit_ts": _ts_text(row["exit_ts"]),
            "qty": float(row["qty"]),
            "entry_px": _num(row["entry_px"]),
            "exit_px": _num(row["exit_px"]),
            "pnl": _num(row["pnl"]),
            "ret": _num(row["ret"]),
            "fees": _num(row["fees"]),
            "bars": float(row["bars"]),
        } for row in trade_frame.to_dict("records")]
        payload["trades"] = _table(rows, [
            ("ticker", "代码", "text"), ("entry_ts", "开仓", "text"),
            ("exit_ts", "平仓", "text"), ("qty", "数量", "int"),
            ("entry_px", "开仓价", "num2"), ("exit_px", "平仓价", "num2"),
            ("pnl", "盈亏", "money"), ("ret", "收益率", "pct2"),
            ("fees", "费用", "money"), ("bars", "持有 bar", "int"),
        ], title="回合交易", sortable=True)

    if len(result.orders):
        rows = [{
            "ts": _ts_text(row["ts"]),
            "ticker": str(row["ticker"]),
            "side": "买" if row["side"] == "buy" else "卖",
            "amount": float(row["amount"]),
            "filled": float(row["filled"]),
            "status": str(row["status"]),
            "reason": str(row["reason"]),
            "tag": str(row["tag"]),
        } for row in result.orders.to_dict("records")]
        payload["orders"] = _table(rows, [
            ("ts", "时间", "text"), ("ticker", "代码", "text"), ("side", "方向", "text"),
            ("amount", "委托", "int"), ("filled", "成交", "int"),
            ("status", "状态", "text"), ("reason", "原因", "text"), ("tag", "来源", "text"),
        ], title="委托明细", sortable=True)

    return _page(payload, payload["title"])


# ---------------------------------------------------------------- 场景对比

_COMPARE_COLUMNS = [
    ("name", "场景", "text"), ("total_return", "总收益", "pct2"),
    ("annual_return", "年化", "pct2"), ("sharpe", "夏普", "num2"),
    ("max_drawdown", "最大回撤", "pct2"), ("trades", "回合", "int"),
    ("fees", "费用", "money"), ("execution_rate", "执行率", "pct2"),
    ("rejected_orders", "拒单", "int"),
]


def plot_comparison_html(results: dict[str, BacktestResult], *, market: str = "us",
                         benchmark: str | None = None, title: str = "场景对比",
                         normalize: bool = True, generated: datetime | None = None) -> HtmlReport:
    """多场景对比 HTML: 叠加净值 + 可排序的指标表。

    Args:
        results: {场景名: BacktestResult}, 通常来自同一份公共配置派生的变体。
        market: 年化口径。
        benchmark: 基准代码(传给 summarize, 用来算超额 / Beta)。
        title: 页面标题。
        normalize: True 时各场景净值归一到 1, 便于比形状; False 用原始净值。
        generated: 生成时间。

    Returns:
        HtmlReport(kind="comparison")。
    """
    index = pd.DatetimeIndex(sorted(set().union(*[set(r.nav.index) for r in results.values()])))
    labels = _labels(index)
    series = []
    for name, result in results.items():
        nav = pd.to_numeric(result.nav.reindex(index).ffill(), errors="coerce")
        if normalize:
            base = nav.dropna()
            nav = nav / float(base.iloc[0]) if len(base) else nav
        series.append({"name": str(name), "values": _values(nav)})

    rows = []
    for name, result in results.items():
        frame_values = {**summarize(result.nav, market=market,
                                    benchmark_nav=benchmark_nav(result, benchmark)).to_dict(),
                        **trade_stats(trades(result.fills)).to_dict(),
                        **execution_stats(result.orders, result.fills, result.nav).to_dict()}
        rows.append({"name": str(name),
                     **{key: _num(frame_values.get(key)) if key != "name" else str(name)
                        for key, _, _ in _COMPARE_COLUMNS}})

    payload: dict[str, Any] = {
        "kind": "comparison",
        "title": title,
        "subtitle": f"{market} · {len(results)} 个场景 · "
                    + ("净值归一到 1" if normalize else "原始净值"),
        "generated": (generated or datetime.now()).strftime("%Y-%m-%d %H:%M"),
        "nav": {
            "title": "净值对比" + ("(归一到 1)" if normalize else ""),
            "type": "line",
            "labels": labels,
            "series": series,
            "valueKind": "num3" if normalize else "money",
            "axisKind": "num2" if normalize else "money",
            "height": 340,
            "ranges": _ranges(labels),
        },
        "table": _table(rows, list(_COMPARE_COLUMNS), title="指标对比",
                        sortable=True, sort_key="total_return",
                        note="点表头排序; 完整字段见 compare_results 的 DataFrame。"),
    }
    return _page(payload, title)


# ---------------------------------------------------------------- walk-forward

_OOS_ROWS = [
    ("total_return", "总收益率"), ("annual_return", "年化收益率"),
    ("annual_vol", "年化波动率"), ("sharpe", "Sharpe"), ("sortino", "Sortino"),
    ("calmar", "Calmar"), ("max_drawdown", "最大回撤"),
    ("max_drawdown_days", "最长回撤 [交易日]"),
]


def plot_walk_forward_html(result: WalkForwardResult, *, title: str | None = None,
                           generated: datetime | None = None) -> HtmlReport:
    """walk-forward HTML: 样本外净值 + 各折测试段收益 + 折明细。

    Args:
        result: walkforward.run_walk_forward 的输出。
        title: 页面标题。
        generated: 生成时间。

    Returns:
        HtmlReport(kind="walk_forward")。
    """
    oos = result.oos_nav
    summary = summarize(oos, market=result.market_name)
    fold_frame = result.fold_table()
    labels = _labels(oos.index)
    page_title = title or f"{result.name} · walk-forward"

    payload: dict[str, Any] = {
        "kind": "walk_forward",
        "title": page_title,
        "subtitle": f"{result.market_name} · {len(result.folds)} 折 · "
                    f"样本外 {_subtitle(oos.index, result.market_name).split(' · ', 1)[-1]}",
        "generated": (generated or datetime.now()).strftime("%Y-%m-%d %H:%M"),
        "kpis": _kpis({**summary.to_dict(),
                       "folds": len(result.folds)}, [
            ("total_return", "样本外总收益", "pct2", "auto"),
            ("annual_return", "样本外年化", "pct2", "auto"),
            ("sharpe", "Sharpe", "num2", "auto"),
            ("max_drawdown", "最大回撤", "pct2", "auto"),
            ("annual_vol", "年化波动率", "pct2", "none"),
            ("folds", "折数", "int", "none"),
        ]),
        "nav": {
            "title": "拼接样本外净值",
            "type": "line",
            "labels": labels,
            "series": [{"name": "样本外", "values": _values(oos)}],
            "valueKind": "money",
            "axisKind": "money",
            "height": 320,
            "ranges": _ranges(labels),
        },
        "fold_bars": {
            "title": "各折测试段收益",
            "type": "bar",
            "colorBySign": True,
            "labels": [f"折{int(f)}" for f in fold_frame["fold"]],
            "series": [{"name": "测试段收益",
                        "values": [_num(v) for v in fold_frame["test_return"]]}],
            "valueKind": "pct1",
            "axisKind": "pct0",
            "height": 220,
        },
        "sections": [{"title": "样本外指标", "rows": [
            [label, format_value(key, summary[key])] for key, label in _OOS_ROWS
        ]}],
        "folds": _table([{
            "fold": float(row["fold"]),
            "train": str(row["train"]),
            "test": str(row["test"]),
            "bars": float(row["bars"]),
            "test_return": _num(row["test_return"]),
            "trades": float(row["trades"]),
        } for row in fold_frame.to_dict("records")], [
            ("fold", "折", "int"), ("train", "训练段", "text"), ("test", "测试段", "text"),
            ("bars", "bar 数", "int"), ("test_return", "测试收益", "pct2"),
            ("trades", "成交笔数", "int"),
        ], title="各折明细", sortable=True),
        "footer": "样本外净值按折内收益连乘拼接; 参数只在各折训练段决定, 测试段不参与。",
    }
    return _page(payload, page_title)
