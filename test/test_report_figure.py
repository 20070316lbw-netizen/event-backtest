"""报告与图: print_result 输出中文标签, tearsheet / 对比图能存, benchmark 能取到。"""
from __future__ import annotations

import io

import numpy as np
from helpers import make_market
from rich.console import Console

from event_backtest.benchmark import benchmark_nav
from event_backtest.engine import run
from event_backtest.fees import FeeSchedule
from event_backtest.figure import plot_comparison, plot_tearsheet
from event_backtest.report import compare_results, print_comparison, print_result
from event_backtest.rules import MarketRules
from event_backtest.strategy.base import Strategy


class _BuyOnce(Strategy):
    def handle_data(self, ctx):
        if ctx.bar == 0:
            ctx.order(0, 3)


def _result():
    close = np.array([[10.0], [10.2], [10.5], [10.1], [10.8], [11.0]])
    market = make_market(close, tickers=["AAA"])
    return run(_BuyOnce(), market, initial_cash=1000.0,
               fees=FeeSchedule(), rules=MarketRules())


def test_print_result_contains_chinese_labels():
    buffer = io.StringIO()
    print_result(_result(), market="us", benchmark="AAA",
                 console=Console(file=buffer, width=100, no_color=True))
    text = buffer.getvalue()
    assert "总收益率" in text and "最大回撤" in text and "回合数" in text
    assert "基准收益率" in text


def test_benchmark_nav_from_result():
    result = _result()
    series = benchmark_nav(result, "AAA")
    assert series is not None
    assert len(series) == len(result.nav)
    assert benchmark_nav(result, "MISSING") is None


def test_compare_results_and_plot():
    result = _result()
    table = compare_results({"无滑点": result, "有滑点": result}, market="us", benchmark="AAA")
    assert list(table.index) == ["无滑点", "有滑点"]
    assert {"total_return", "trades", "fees"} <= set(table.columns)

    buffer = io.StringIO()
    print_comparison(table, console=Console(file=buffer, width=120, no_color=True))
    assert "场景" in buffer.getvalue() and "无滑点" in buffer.getvalue()

    fig = plot_comparison({"a": result.nav, "b": result.nav}, normalize=False)
    assert len(fig.axes) == 1


def test_plot_tearsheet_saves(tmp_path):
    fig = plot_tearsheet(_result(), market="us", benchmark="AAA")
    assert len(fig.axes) == 2
    path = tmp_path / "tearsheet.png"
    fig.savefig(path)
    assert path.exists() and path.stat().st_size > 0
