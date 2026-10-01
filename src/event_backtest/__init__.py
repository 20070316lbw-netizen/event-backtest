"""event-backtest: 轻量事件驱动回测框架, 支持 A 股与美股。

设计参考 zipline-reloaded 的事件循环与订单模型, 以及 minievent 的对齐数组数据层、
市场配置与声明式 YAML 策略。三条用法:

    # 1) 脚本式(推荐日常用): 改 scripts/ 下的参数, 编辑器里点运行
    # 2) 命令行: event-backtest run --config cn --strategy volume_breakout --freq 30
    # 3) Python
    from event_backtest import DeclarativeStrategy, load_config, load_strategy
    cfg = load_config("cn")
    result = cfg.run(DeclarativeStrategy(load_strategy("volume_breakout")),
                     start="2026-01-05", end="2026-09-24", freq="30")
"""
from __future__ import annotations

from event_backtest.benchmark import benchmark_nav
from event_backtest.broker import Fill, Order, Portfolio, apply_fill, match_order
from event_backtest.cli import main
from event_backtest.config import ConfigError, MarketConfig, default_fees, load_config
from event_backtest.engine import BacktestResult, Context, ContextStrategy, run
from event_backtest.evaluation import (
    DEFAULT_HORIZONS,
    EventStudy,
    bootstrap_mean,
    mean_t_test,
    study_events,
    two_sided_t_p,
)
from event_backtest.facade import UNSET, EventBackTest
from event_backtest.factor import FactorSpec, compute, load_specs, market_frames
from event_backtest.fees import ZERO_FEE, FeeSchedule
from event_backtest.figure import (
    HtmlReport,
    plot_comparison,
    plot_comparison_html,
    plot_tearsheet,
    plot_tearsheet_html,
    plot_walk_forward,
    plot_walk_forward_html,
)
from event_backtest.market import (
    MarketData,
    build_cn_market,
    build_us_market,
    load_market,
    slice_dates,
    slice_market,
)
from event_backtest.metrics import (
    buy_and_hold_nav,
    daily_nav,
    execution_stats,
    performance,
    periods_per_year,
    reject_reasons,
    summarize,
    trade_stats,
    trades,
)
from event_backtest.report import (
    compare_results,
    print_comparison,
    print_result,
    print_walk_forward,
)
from event_backtest.results import Run, jsonable, save_run
from event_backtest.rules import (
    CNMarketRules,
    MarketRules,
    USMarketRules,
    default_rules,
    limit_prices,
)
from event_backtest.signal import SignalSpec, generate_events, parse_signal
from event_backtest.slippage import (
    FixedSlippage,
    NoSlippage,
    SlippageModel,
    VolumeShareSlippage,
    make_slippage,
)
from event_backtest.strategy import (
    STRATEGIES,
    BuyHold,
    DeclarativeStrategy,
    ExitRules,
    FactorRef,
    Sizing,
    SmaCross,
    Strategy,
    StrategySpec,
    WalkForward,
    load_strategy,
    make_strategy,
    parse_strategy,
    resolve_strategy,
)
from event_backtest.sweep import SweepResult, expand_grid, make_grid_select, search
from event_backtest.walkforward import (
    FoldRange,
    FoldResult,
    WalkForwardError,
    WalkForwardResult,
    fold_ranges,
    run_walk_forward,
)
from event_backtest.walkforward import (
    check_data as check_walk_forward_data,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "STRATEGIES",
    "UNSET",
    "ZERO_FEE",
    "BacktestResult",
    "BuyHold",
    "CNMarketRules",
    "ConfigError",
    "Context",
    "ContextStrategy",
    "DeclarativeStrategy",
    "EventBackTest",
    "EventStudy",
    "ExitRules",
    "FactorRef",
    "FactorSpec",
    "FeeSchedule",
    "Fill",
    "FixedSlippage",
    "FoldRange",
    "FoldResult",
    "HtmlReport",
    "MarketConfig",
    "MarketData",
    "MarketRules",
    "NoSlippage",
    "Order",
    "Portfolio",
    "Run",
    "SignalSpec",
    "Sizing",
    "SlippageModel",
    "SmaCross",
    "Strategy",
    "StrategySpec",
    "SweepResult",
    "USMarketRules",
    "VolumeShareSlippage",
    "WalkForward",
    "WalkForwardError",
    "WalkForwardResult",
    "apply_fill",
    "benchmark_nav",
    "bootstrap_mean",
    "build_cn_market",
    "build_us_market",
    "buy_and_hold_nav",
    "check_walk_forward_data",
    "compare_results",
    "compute",
    "daily_nav",
    "default_fees",
    "default_rules",
    "execution_stats",
    "expand_grid",
    "fold_ranges",
    "generate_events",
    "jsonable",
    "limit_prices",
    "load_config",
    "load_market",
    "load_specs",
    "load_strategy",
    "main",
    "make_grid_select",
    "make_slippage",
    "make_strategy",
    "market_frames",
    "match_order",
    "mean_t_test",
    "parse_signal",
    "parse_strategy",
    "performance",
    "periods_per_year",
    "plot_comparison",
    "plot_comparison_html",
    "plot_tearsheet",
    "plot_tearsheet_html",
    "plot_walk_forward",
    "plot_walk_forward_html",
    "print_comparison",
    "print_result",
    "print_walk_forward",
    "reject_reasons",
    "resolve_strategy",
    "run",
    "run_walk_forward",
    "save_run",
    "search",
    "slice_dates",
    "slice_market",
    "study_events",
    "summarize",
    "trade_stats",
    "trades",
    "two_sided_t_p",
]
