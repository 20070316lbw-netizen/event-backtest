"""event-backtest 的场景对比脚本: 一份公共配置, 派生出多个变体, 横向比。

对应 minibacktest 的用法: common_kwargs 定义基准配置, 再用 Backtester(**common_kwargs,
...) 加一个变量(比如 commission_bps/slippage_bps), 对比这个变量带来的差异。

这里每个场景 = 一个策略名 + 若干对 MarketConfig 的覆盖。空覆盖就是"基准场景"。

用法: 改下面"参数区", 在编辑器里点运行; 或 `uv run python scripts/run_backtest.py`。
注意: 图里的文字一律用英文(默认字体没有中文字形), 所以场景名用英文; 终端表格是中文。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from event_backtest import (
    DeclarativeStrategy,
    FixedSlippage,
    load_config,
    load_strategy,
)
from event_backtest.figure import plot_comparison, plot_tearsheet
from event_backtest.report import compare_results, print_comparison, print_result

# ---------------------------------------------------------------- 参数区(改这里)
MARKET = "us"                      # "us" / "cn", 对应 setting/<MARKET>.yaml
DB = None                          # 覆盖配置里的数据库; None = 用配置文件里的
FREQ = None                        # 覆盖配置里的频率; None = 用配置文件里的
START = "2024-01-01"
END = "2024-12-31"
OUTPUT = "outputs/compare"         # 图输出目录

# 场景: 场景名(英文, 图上用) -> {"strategy": 策略名, ...对 MarketConfig 的覆盖}
# 空覆盖 {} 就是"基准场景"; 其余场景在公共配置上只加一个变量, 看它的影响。
SCENARIOS: dict[str, dict[str, object]] = {
    "volume_breakout": {"strategy": "volume_breakout"},
    "volume_breakout + slip0.05": {"strategy": "volume_breakout",
                                   "slippage": FixedSlippage(spread=0.05)},
    "momentum_only": {"strategy": "momentum_only"},
}
# ----------------------------------------------------------------


def main() -> None:
    """逐个跑场景, 打印各自报告 + 对比表, 并保存叠加图与各场景 tearsheet。"""
    repo = Path(__file__).resolve().parents[1]
    base = load_config(MARKET)

    # 参数区里显式写了才覆盖配置(默认全用 setting/<MARKET>.yaml)
    overrides: dict[str, object] = {}
    if DB is not None:
        db = Path(DB)
        overrides["db_path"] = db if db.is_absolute() else (repo / db).resolve()
    if FREQ is not None:
        overrides["freq"] = FREQ
    if overrides:
        base = replace(base, **overrides)

    results = {}
    for name, scenario in SCENARIOS.items():
        scenario = dict(scenario)
        strategy_name = str(scenario.pop("strategy"))
        cfg = replace(base, **scenario) if scenario else base
        result = cfg.run(DeclarativeStrategy(load_strategy(strategy_name)),
                         start=START, end=END)
        results[name] = result
        print(f"\n===== {name} =====")
        print_result(result, market=cfg.market, benchmark=cfg.benchmark)

    print("\n===== 场景对比 =====")
    table = compare_results(results, market=base.market, benchmark=base.benchmark)
    print_comparison(table, title="场景对比")

    out = repo / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    plot_comparison({name: r.nav for name, r in results.items()}).savefig(
        out / "comparison.png", dpi=120)
    for name, result in results.items():
        plot_tearsheet(result, market=base.market, benchmark=base.benchmark,
                       title=name).savefig(out / f"{name}.png", dpi=120)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
