"""event-backtest 的 walk-forward 脚本: 改下面参数区, 在编辑器里点运行。

折数 / 训练 / 测试 / 隔离都从策略 yaml 的 walk_forward 段读。
GRID 非空时, 每折会在**训练段**上扫一遍网格、挑最优参数, 再拿去测**测试段** ——
这样得到的是样本外净值(参数没有偷看测试段)。

命令行等价写法: uv run python scripts/run_walk_forward.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from event_backtest import (
    load_config,
    make_grid_select,
    parse_strategy,
    print_walk_forward,
    run_walk_forward,
)
from event_backtest.figure import plot_walk_forward, plot_walk_forward_html
from event_backtest.strategy import STRATEGY_DIR

# ---------------------------------------------------------------- 参数区(改这里)
MARKET = "cn"
DB = "/Users/liu/quant/minievent/data/ashare.db"   # None = 用配置文件里的
FREQ = "30"                                       # None = 用配置文件里的
STRATEGY = "volume_breakout"
START = None                                      # 喂给 cfg.load 的区间; None 不限
END = None
OUTPUT = "outputs/walk_forward"

# 每折在训练段上扫的参数网格; 空 dict = 不选参数(分折样本外评估)。
GRID: dict[str, list[object]] = {
    "signal.all[0].value": [1.5, 2.0, 3.0],
}
SELECT_BY = "sharpe"                              # 选优依据
# ----------------------------------------------------------------


def main() -> None:
    """跑 walk-forward(可选每折选参), 打印每折表 + 样本外指标, 保存结果与图。"""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(MARKET)
    if DB is not None:
        db = Path(DB)
        cfg = replace(cfg, db_path=db if db.is_absolute() else (repo / db).resolve())
    if FREQ is not None:
        cfg = replace(cfg, freq=FREQ)

    raw = yaml.safe_load((STRATEGY_DIR / f"{STRATEGY}.yaml").read_text(encoding="utf-8"))
    spec = parse_strategy(raw, where=STRATEGY)

    chosen: list[dict[str, object]] = []
    select = (make_grid_select(raw, GRID, cfg, metric=SELECT_BY, record=chosen)
              if GRID else None)
    result = run_walk_forward(cfg, spec, select=select, start=START, end=END)
    print_walk_forward(result)
    if chosen:
        print("\n每折在训练段选到的参数:")
        for index, params in enumerate(chosen):
            print(f"  折 {index}: {params}")

    out = repo / OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    result.fold_table().to_csv(out / "fold_table.csv", index=False)
    result.oos_nav.rename("nav").to_frame().to_parquet(out / "oos_nav.parquet")
    plot_walk_forward(result).savefig(out / "oos.png", dpi=120)
    plot_walk_forward_html(result).save(out / "oos.html")
    print(f"\n结果写入(含 oos.html): {out}")


if __name__ == "__main__":
    main()
