"""命令行入口。

    event-backtest list
    event-backtest run --config cn --strategy volume_breakout \
                       --start 2026-01-05 --end 2026-09-24 --freq 30 \
                       --event-study --output outputs/volume_breakout --html

--strategy 既可以是 strategy/ 下的 yaml 名(声明式), 也可以是内置代码策略
(buy_hold / sma_cross)。--event-study 只对声明式 YAML 策略有效。
终端里的指标表由 report.print_result 打印, 和脚本式入口同一套格式。

实现上 CLI 只是 `EventBackTest` 门面的一个消费者: 解析参数 -> 构造门面 -> run -> 打印 /
落盘。--json 时改成输出机器可读的 JSON(stdout 只有 JSON, 日志走 stderr)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from event_backtest.config import SETTING_DIR
from event_backtest.evaluation import EventStudy
from event_backtest.facade import UNSET, EventBackTest
from event_backtest.factor import FACTOR_DIR, load_specs
from event_backtest.metrics import reject_reasons
from event_backtest.results import Run, jsonable
from event_backtest.strategy import STRATEGIES, STRATEGY_DIR, load_strategy

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
    if args.command == "show":
        return _show(args)
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
    run_p.add_argument("--output", default=None, help="结果输出目录(parquet / csv / run.json)")
    run_p.add_argument("--event-study", action="store_true", help="对声明式策略做事件研究")
    run_p.add_argument("--bootstrap", type=int, default=1000,
                       help="事件研究的 bootstrap 重抽次数(0 = 不算)")
    run_p.add_argument("--html", action="store_true",
                       help="额外写一份自包含的交互式 tearsheet.html(需配合 --output)")
    run_p.add_argument("--json", action="store_true",
                       help="stdout 只输出机器可读的 JSON(指标 / 上下文 / 产出)")
    run_p.add_argument("--fast", type=int, default=5, help="sma_cross 快线窗口")
    run_p.add_argument("--slow", type=int, default=20, help="sma_cross 慢线窗口")

    sub.add_parser("list", help="列出配置 / 策略 / 因子")

    show_p = sub.add_parser("show", help="打印因子 / 策略的 yaml 原文(带注释)与解析要点")
    show_p.add_argument("kind", choices=("factor", "strategy"), help="看因子还是策略")
    show_p.add_argument("name", help="因子 / 策略名(yaml 文件名)")
    return parser


def _run(args: argparse.Namespace) -> int:
    """执行 run 子命令: 用 EventBackTest 跑一遍, 打印(或 --json 输出), 可选落盘。"""
    backtest = EventBackTest(
        args.config,
        db_path=args.db,
        start=args.start,
        end=args.end,
        freq=UNSET if args.freq is None else args.freq,
        strategy=args.strategy,
    )
    run = backtest.run(fast=args.fast, slow=args.slow)
    logger.info(f"市场={run.market} 策略={run.strategy} bar={len(run.nav)} "
                f"成交={len(run.fills)} 回合={len(run.trade_frame())}")

    study = _maybe_event_study(args, run, quiet=args.json)

    if args.json:
        print(json.dumps(_report_json(run, study, args), ensure_ascii=False, indent=2))
    else:
        run.print()

    if args.output:
        out = run.save(args.output, html=args.html,
                       title=f"{run.market} · {run.strategy}")
        logger.info(f"结果写入 {out.resolve()}")
    elif args.html:
        logger.warning("--html 要写到目录里, 需要同时给 --output; 已跳过")
    return 0


def _maybe_event_study(args: argparse.Namespace, run: Run, *, quiet: bool = False
                       ) -> EventStudy | None:
    """按 --event-study 算事件研究并打印汇总; 代码策略(没有 spec)时跳过。

    Args:
        args: 命令行参数。
        run: 刚跑完的 Run(用它缓存事件研究)。
        quiet: True 时不打印表格(--json 模式下 stdout 只能有 JSON)。
    """
    if not args.event_study:
        return None
    try:
        study = run.study(bootstrap=args.bootstrap)
    except ValueError as exc:
        logger.warning(f"--event-study 跳过: {exc}")
        return None
    logger.info(f"事件研究: 事件={len(study.events)}")
    if quiet:
        return study
    print(study.summary.to_string(index=False))
    if not study.tests.empty:
        print()
        print("显著性检验(按交易日聚类):")
        print(study.tests_table().to_string(index=False))
        logger.info("p 值每个 horizon 单独检验, 未做多重比较校正; 星号 *** <0.01, ** <0.05, * <0.1")
    return study


def _report_json(run: Run, study: EventStudy | None, args: argparse.Namespace) -> dict:
    """--json 的输出: 指标 + 运行上下文 + 事件研究 + 产出路径。"""
    return {
        "market": run.market,
        "benchmark": run.benchmark,
        "strategy": run.strategy,
        "bars": len(run.nav.dropna()),
        "fills": len(run.fills),
        "orders": len(run.orders),
        "metrics": jsonable(run.values()),
        "reject_reasons": {str(key): int(value)
                           for key, value in reject_reasons(run.orders).items()},
        "event_study": None if study is None else {
            "events": len(study.events),
            "tests": jsonable(study.tests.to_dict("records")),
        },
        "output": str(Path(args.output).resolve()) if args.output else None,
    }


def _show(args: argparse.Namespace) -> int:
    """打印因子 / 策略的 yaml 原文(注释就是文档)与解析出来的要点。

    Args:
        args: 命令行参数(kind / name)。

    Returns:
        0 成功。

    Raises:
        SystemExit: 名字不存在(并列出可用的)。
    """
    if args.kind == "factor":
        catalogue = load_specs()
        if args.name not in catalogue:
            raise SystemExit(f"没有因子 {args.name!r}; 可用的有 {sorted(catalogue)}")
        spec = catalogue[args.name]
        print((FACTOR_DIR / f"{args.name}.yaml").read_text(encoding="utf-8").rstrip())
        print()
        print(f"--- 解析: 因子 {spec.name} | 参数 {list(spec.params)} | 输出 {spec.output}")
        for index, step in enumerate(spec.steps, start=1):
            operands = {key: value for key, value in step.items()
                        if key not in ("id", "operation")}
            print(f"    {index}. {step.get('id')} = {step.get('operation')} {operands}")
        return 0

    path = STRATEGY_DIR / f"{args.name}.yaml"
    if not path.is_file():
        available = sorted(item.stem for item in STRATEGY_DIR.glob("*.yaml"))
        raise SystemExit(f"没有策略 {args.name!r}; 可用的有 {available}")
    spec = load_strategy(args.name)
    print(path.read_text(encoding="utf-8").rstrip())
    print()
    print(f"--- 解析: 策略 {spec.name}")
    factors = ", ".join(
        f"{ref.alias}({', '.join(f'{key}={value}' for key, value in ref.params.items())})"
        for ref in spec.factors)
    print(f"    因子: {factors or '(无)'}")
    if spec.signal is not None:
        print(f"    信号: {spec.signal.name} | trigger={spec.signal.trigger}(条件看上面 yaml)")
    rules = spec.exit
    print(f"    出场: hold_bars={rules.hold_bars} take_profit={rules.take_profit} "
          f"stop_loss={rules.stop_loss}")
    print(f"    仓位: percent={spec.sizing.percent} max_positions={spec.sizing.max_positions}")
    print(f"    walk-forward: {spec.walk_forward}")
    return 0


def _list() -> int:
    """列出可用的市场配置、策略(yaml + 代码)与因子。"""
    configs = sorted(p.stem for p in SETTING_DIR.glob("*.yaml"))
    strategies = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
    factors = sorted(load_specs())
    print("配置  :", ", ".join(configs) or "(无)")
    print("策略  :", ", ".join(strategies) or "(无)", "| 代码:", ", ".join(sorted(STRATEGIES)))
    print("因子  :", ", ".join(factors) or "(无)")
    return 0
