"""event-backtest 的场景对比脚本: 一份公共配置, 派生出多个变体, 横向比。

对应 minibacktest 的用法: common_kwargs 定义基准配置, 再用 Backtester(**common_kwargs,
...) 加一个变量(比如 commission_bps/slippage_bps), 对比这个变量带来的差异。

这里每个场景 = 一个策略名 + 若干对 MarketConfig 的覆盖。空覆盖就是"基准场景"。
底层就是 EventBackTest.compare + results.save_comparison, 脚本只负责参数区与打印。

用法: 改下面"参数区", 在编辑器里点运行; 或 `uv run python scripts/run_backtest.py`。
注意: 图里的文字一律用英文(默认字体没有中文字形), 所以场景名用英文; 终端表格是中文。
"""
from __future__ import annotations

from pathlib import Path

from event_backtest import UNSET, EventBackTest, FixedSlippage
from event_backtest.report import compare_results, print_comparison
from event_backtest.results import save_comparison

# ---------------------------------------------------------------- 参数区(改这里)
MARKET = "us"                      # "us" / "cn" / "cn_stock", 对应 setting/<MARKET>.yaml
DB = None                          # 覆盖配置里的数据库; None = 用配置文件里的
FREQ = None                        # 覆盖配置里的频率; None = 用配置文件里的
START = "2024-01-01"
END = "2024-12-31"
OUTPUT = "outputs/compare"         # 输出目录(图 / HTML / csv)

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
    """逐个跑场景, 打印各自报告 + 对比表, 并保存 PNG / HTML / csv。"""
    repo = Path(__file__).resolve().parents[1]
    backtest = EventBackTest(
        MARKET,
        db_path=DB,
        freq=UNSET if FREQ is None else FREQ,
        start=START,
        end=END,
    )

    runs = backtest.compare(SCENARIOS)          # {场景名: Run}; 行情只加载一次
    for name, run in runs.items():
        print(f"\n===== {name} =====")
        run.print()

    print("\n===== 场景对比 =====")
    table = compare_results({name: run.result for name, run in runs.items()},
                            market=backtest.name, benchmark=backtest.benchmark)
    print_comparison(table, title="场景对比")

    out = save_comparison(repo / OUTPUT, runs, market=backtest.name,
                          benchmark=backtest.benchmark)
    print(f"\n结果写入(csv + png + html): {out}")


if __name__ == "__main__":
    main()
