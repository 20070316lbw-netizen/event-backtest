"""命令行入口。

    event-backtest list
    event-backtest run --config cn --strategy volume_breakout \
                       --start 2026-01-05 --end 2026-09-24 --freq 30 \
                       --event-study --output outputs/volume_breakout

--strategy 既可以是 strategy/ 下的 yaml 名(声明式), 也可以是内置代码策略
(buy_hold / sma_cross)。--event-study 只对声明式 YAML 策略有效。
终端里的指标表由 report.print_result 打印, 和脚本式入口同一套格式。
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import replace
from pathlib import Path

from loguru import logger

from event_backtest.benchmark import benchmark_nav
from event_backtest.config import SETTING_DIR, load_config
from event_backtest.engine import run as run_backtest
from event_backtest.evaluation import EventStudy, study_events
from event_backtest.factor import load_specs
from event_backtest.figure import plot_tearsheet_html
from event_backtest.market import MarketData
from event_backtest.metrics import execution_stats, summarize, trade_stats, trades
from event_backtest.report import print_result
from event_backtest.strategy import (
    STRATEGIES,
    STRATEGY_DIR,
    DeclarativeStrategy,
    StrategySpec,
    load_strategy,
)
from event_backtest.strategy.base import make_strategy

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """命令行入口; 返回进程退出码。

    Args:
        argv: 参数列表; None 表示用 sys.argv(交给 argparse 处理)。

    Returns:
        0 成功。
    """
    args = _parser().parse_args(argv)
    if args.command == "list":
        return _list()
    return _run(args)


def _parser() -> argparse.ArgumentParser:
    """建参数解析器: 目前有 run / list 两个子命令。"""
    parser = argparse.ArgumentParser(prog="event-backtest", description="轻量事件驱动回测")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="在某个市场配置上跑一个策略")
    run_p.add_argument("--config", default="us", help="setting/<名字>.yaml 或 yaml 路径")
    run_p.add_argument("--strategy", default="volume_breakout", help="策略 yaml 名或内置代码策略")
    run_p.add_argument("--start", default=None, help="起始交易日")
    run_p.add_argument("--end", default=None, help="结束交易日")
    run_p.add_argument("--freq", default=None, help="分钟线周期, 覆盖配置")
    run_p.add_argument("--db", default=None, help="覆盖配置里的数据库路径")
    run_p.add_argument("--output", default=None, help="结果输出目录(parquet / csv)")
    run_p.add_argument("--event-study", action="store_true", help="对声明式策略做事件研究")
    run_p.add_argument("--html", action="store_true",
                       help="额外写一份自包含的交互式 tearsheet.html(需配合 --output)")
    run_p.add_argument("--fast", type=int, default=5, help="sma_cross 快线窗口")
    run_p.add_argument("--slow", type=int, default=20, help="sma_cross 慢线窗口")

    sub.add_parser("list", help="列出配置 / 策略 / 因子")
    return parser


def _run(args: argparse.Namespace) -> int:
    """执行 run 子命令: 读配置与策略, 跑回测, 打印中文指标表, 可选写文件与事件研究。"""
    cfg = load_config(args.config)
    if args.db:
        cfg = replace(cfg, db_path=args.db)
    market = cfg.load(start=args.start, end=args.end, freq=args.freq)
    strategy, spec = _build_strategy(args)
    result = run_backtest(strategy, market, initial_cash=cfg.initial_cash,
                          fees=cfg.resolved_fees(), fill_price=cfg.fill_price,
                          slippage=cfg.slippage, rules=cfg.resolved_rules())

    summary = summarize(result.nav, market=cfg.market,
                        benchmark_nav=benchmark_nav(result, cfg.benchmark))
    order_trades = trades(result.fills)
    stats = trade_stats(order_trades)
    logger.info(f"市场={cfg.market} 策略={args.strategy} bar={len(result.nav)} "
                f"成交={len(result.fills)} 回合={len(order_trades)}")
    print_result(result, market=cfg.market, benchmark=cfg.benchmark)

    study = _maybe_event_study(args, spec, market)
    if args.output:
        _write(args.output, result, summary, stats, order_trades, study,
               market=cfg.market, benchmark=cfg.benchmark,
               title=f"{cfg.market} · {args.strategy}", html=args.html)
    elif args.html:
        logger.warning("--html 要写到目录里, 需要同时给 --output; 已跳过")
    return 0


def _build_strategy(args: argparse.Namespace) -> tuple[object, StrategySpec | None]:
    """把 --strategy 解析成 (策略对象, 声明式 spec 或 None)。

    优先级: yaml 路径 -> strategy/ 下的 yaml 名 -> 内置代码策略; 都没有则退出。
    第二个返回值给事件研究用(只有声明式策略才有 spec)。
    """
    path = Path(args.strategy)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        spec = load_strategy(path)
        return DeclarativeStrategy(spec), spec
    if (STRATEGY_DIR / f"{args.strategy}.yaml").is_file():
        spec = load_strategy(args.strategy)
        return DeclarativeStrategy(spec), spec
    if args.strategy in STRATEGIES:
        # 代码策略: 只把它构造函数认识的、命令行也有的参数传进去(如 sma_cross 的 fast/slow)
        cls = STRATEGIES[args.strategy]
        kwargs = {name: getattr(args, name)
                  for name in inspect.signature(cls.__init__).parameters
                  if name != "self" and hasattr(args, name)}
        return make_strategy(args.strategy, **kwargs), None
    raise SystemExit(f"未知策略 {args.strategy!r}: 既不是 {STRATEGY_DIR} 下的 yaml, "
                     f"也不在 {sorted(STRATEGIES)}")


def _maybe_event_study(args: argparse.Namespace, spec: StrategySpec | None,
                       market: MarketData) -> EventStudy | None:
    """按 --event-study 做事件研究并打印汇总; 代码策略(没有 spec)时跳过。"""
    if not args.event_study:
        return None
    if spec is None:
        logger.warning("--event-study 只对声明式 YAML 策略有效, 已跳过")
        return None
    values = spec.compute(market)
    events = spec.events(values, market)
    study = study_events(market, events, strategy_name=spec.name)
    logger.info(f"事件研究: 事件={len(events)}")
    print(study.summary.to_string(index=False))
    return study


def _write(output: str, result: object, summary: object, stats: object,
           order_trades: object, study: EventStudy | None, *, market: str = "us",
           benchmark: str | None = None, title: str | None = None,
           html: bool = False) -> None:
    """写回测结果: nav / fills / trades / orders 用 parquet, 指标用 csv。

    委托明细(orders)含被拒 / 期末未成交的订单与原因; 有事件研究再写两张;
    html=True 再写一份自包含的交互式 tearsheet.html。
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result.nav.rename("nav").to_frame().to_parquet(out / "nav.parquet")
    result.fills.to_parquet(out / "fills.parquet")
    result.orders.to_parquet(out / "orders.parquet")
    order_trades.to_parquet(out / "trades.parquet")
    execution_stats(result.orders, result.fills, result.nav).rename("value").to_frame().to_csv(
        out / "execution.csv")
    summary.rename("value").to_frame().to_csv(out / "performance.csv")
    stats.rename("value").to_frame().to_csv(out / "trade_stats.csv")
    if study is not None:
        study.paths.to_parquet(out / "event_paths.parquet")
        study.summary.to_csv(out / "event_summary.csv", index=False)
    if html:
        page = plot_tearsheet_html(result, market=market, benchmark=benchmark, title=title)
        logger.info(f"HTML tearsheet: {page.save(out / 'tearsheet.html')} "
                    f"({len(page) / 1024:.0f} KB)")
    logger.info(f"结果写入 {out.resolve()}")


def _list() -> int:
    """列出可用的市场配置、策略(yaml + 代码)与因子。"""
    configs = sorted(p.stem for p in SETTING_DIR.glob("*.yaml"))
    strategies = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
    factors = sorted(load_specs())
    print("配置  :", ", ".join(configs) or "(无)")
    print("策略  :", ", ".join(strategies) or "(无)", "| 代码:", ", ".join(sorted(STRATEGIES)))
    print("因子  :", ", ".join(factors) or "(无)")
    return 0
