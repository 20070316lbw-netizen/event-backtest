"""一次回测的"成品": Run 容器 + 公开的落盘入口。

分两层, 各取所需:
    - **开发者**: engine.run / metrics / report / figure 这些组件照旧可单独组合;
    - **调用者**: `Run` 把"这次跑的配置 + 结果 + 口径"绑在一起, 于是
      summary() / trades() / execution() / html() / save() 都不用再重复传
      market / benchmark —— 那套咒语原本在 4 个文件里抄了 6 遍。

    cfg = load_config("us")
    run = Run(cfg.run(strategy), config=cfg, strategy="volume_breakout")
    run.summary()["sharpe"]
    run.save("outputs/demo", html=True)
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from event_backtest.benchmark import benchmark_nav
from event_backtest.engine import BacktestResult
from event_backtest.evaluation.events import DEFAULT_HORIZONS, EventStudy, study_events
from event_backtest.metrics import execution_stats, last_marks, summarize, trade_stats, trades
from event_backtest.report import print_result, result_sections, result_values

if TYPE_CHECKING:  # 只给类型用, 避免 import 期就把 matplotlib 拖进来
    from rich.console import Console

    from event_backtest.config import MarketConfig
    from event_backtest.figure import HtmlReport
    from event_backtest.strategy import StrategySpec

__all__ = ["Metrics", "Run", "jsonable", "save_comparison", "save_run"]


@dataclass(frozen=True)
class Metrics:
    """类型化的指标视图: 有属性名, 编辑器能补全; 缺的字段是 None。

    值就是 report.result_values 的那一份(组合 + 交易 + 执行), 只是换成了具名属性,
    所以 run.metrics().sharpe 与 run.summary()["sharpe"] 永远一致。
    """

    # 概览
    bars: float | None = None
    days: float | None = None
    # 收益 / 风险
    total_return: float | None = None
    annual_return: float | None = None
    annual_vol: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    max_drawdown: float | None = None
    max_drawdown_days: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    beta: float | None = None
    # 回合交易
    trades: float | None = None
    open_trades: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_pnl: float | None = None
    avg_ret: float | None = None
    avg_bars: float | None = None
    fees: float | None = None
    # 执行
    orders: float | None = None
    filled_orders: float | None = None
    rejected_orders: float | None = None
    cancelled_orders: float | None = None
    execution_rate: float | None = None
    turnover: float | None = None
    cost_ratio: float | None = None

    @classmethod
    def from_values(cls, values: Mapping[str, object]) -> Metrics:
        """从 result_values 的字典里挑出已知字段; NaN / 缺字段都留 None。

        Args:
            values: report.result_values 的输出(或同结构的字典)。

        Returns:
            Metrics。
        """
        known = {item.name for item in fields(cls)}
        clean: dict[str, float] = {}
        for name in known:
            value = values.get(name)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
                    float(value)):
                clean[name] = float(value)
        return cls(**clean)


@dataclass(eq=False)
class Run:
    """一次回测的完整上下文。

    Attributes:
        result: engine.run 的输出(净值 / 成交 / 委托 / 行情)。
        config: 这次用的市场配置; None 表示调用者直接调了 engine.run。
        strategy: 策略名(写进报告与 run.json)。
        strategy_spec: 声明式策略的 spec; 代码策略为 None(事件研究需要它)。
        event_study: 调过 study() 之后缓存的事件研究结果。
    """

    result: BacktestResult
    config: MarketConfig | None = None
    strategy: str = ""
    strategy_spec: StrategySpec | None = None
    event_study: EventStudy | None = None

    # ---------------------------------------------------------------- 基本属性
    @property
    def nav(self) -> pd.Series:
        """逐 bar 净值。"""
        return self.result.nav

    @property
    def fills(self) -> pd.DataFrame:
        """成交明细。"""
        return self.result.fills

    @property
    def orders(self) -> pd.DataFrame:
        """委托明细(含拒单原因)。"""
        return self.result.orders

    @property
    def market(self) -> str:
        """市场名 "cn" / "us" (年化口径按它取)。"""
        if self.config is not None:
            return self.config.market
        return str(self.result.meta.get("market") or "us")

    @property
    def benchmark(self) -> str | None:
        """基准代码; 没配就是 None。"""
        if self.config is not None:
            return self.config.benchmark
        value = self.result.meta.get("benchmark")
        return str(value) if value else None

    # ---------------------------------------------------------------- 指标
    def benchmark_curve(self) -> pd.Series | None:
        """基准的买入持有净值; 没配基准或基准不在行情里时 None。"""
        if not self.benchmark:
            return None
        return benchmark_nav(self.result, self.benchmark)

    def summary(self) -> pd.Series:
        """组合指标(总收益 / 年化 / Sharpe / 最大回撤 ...), 与终端表同源。"""
        return summarize(self.nav, market=self.market,
                         benchmark_nav=self.benchmark_curve())

    def trade_frame(self) -> pd.DataFrame:
        """回合交易(开仓到平仓); 期末未平仓的用最后收盘价盯市, open=True。"""
        return trades(self.fills, marks=last_marks(self.result.market))

    def trade_stats(self) -> pd.Series:
        """回合交易的汇总(胜率 / 盈亏比 / 平均持有 ...)。"""
        return trade_stats(self.trade_frame())

    def execution(self) -> pd.Series:
        """执行情况(委托 / 成交 / 拒单 / 执行率 / 换手 / 费用占比)。"""
        return execution_stats(self.orders, self.fills, self.nav)

    def values(self) -> dict[str, object]:
        """上面所有指标合成一个字典(HTML / 落盘 / 自定义报告用)。"""
        return result_values(self.result, market=self.market, benchmark=self.benchmark)

    def metrics(self) -> Metrics:
        """类型化指标(属性名可补全): run.metrics().sharpe / .execution_rate / ...。"""
        return Metrics.from_values(self.values())

    def sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """[(分组标题, [(中文标签, 值)])], 与 print_result 打印的一致。"""
        return result_sections(self.result, market=self.market, benchmark=self.benchmark)

    def print(self, *, console: Console | None = None) -> None:
        """把结果打印成分组中文表(等价 print_result, 但不用再传 market / benchmark)。"""
        print_result(self.result, market=self.market, benchmark=self.benchmark,
                     console=console)

    # ---------------------------------------------------------------- 事件研究 / 出图 / 落盘
    def study(self, *, horizons: tuple[int, ...] | None = None, **kwargs: Any) -> EventStudy:
        """算事件研究并缓存到 self.event_study。

        Args:
            horizons: 观察窗口(bar 数); None 用默认 1..16。
            **kwargs: 透传给 study_events(如 bootstrap / alpha / seed / cluster)。

        Returns:
            EventStudy。

        Raises:
            ValueError: 这次跑的是代码策略(没有声明式 spec), 做不了事件研究。
        """
        if self.strategy_spec is None:
            raise ValueError(
                "代码策略没有声明式 spec, 做不了事件研究; 请用 YAML 策略, "
                "或自己调 study_events(market, events)")
        market = self.result.market
        events = self.strategy_spec.events(self.strategy_spec.compute(market), market)
        self.event_study = study_events(
            market, events, horizons=horizons or DEFAULT_HORIZONS,
            strategy_name=self.strategy_spec.name, **kwargs)
        return self.event_study

    def html(self, *, title: str | None = None) -> HtmlReport:
        """自包含交互式 tearsheet。"""
        from event_backtest.figure import plot_tearsheet_html

        return plot_tearsheet_html(self.result, market=self.market,
                                   benchmark=self.benchmark,
                                   title=title or f"{self.market} · {self.strategy}".strip(" ·"))

    def save(self, directory: str | Path, *, html: bool = False,
             title: str | None = None) -> Path:
        """把这次结果写成一套文件(parquet / csv / run.json, 可选 tearsheet.html)。"""
        return save_run(directory, self.result, market=self.market,
                        benchmark=self.benchmark, title=title, html=html,
                        study=self.event_study, strategy=self.strategy)


# ---------------------------------------------------------------- 落盘

def save_run(directory: str | Path, result: BacktestResult, *, market: str | None = None,
             benchmark: str | None = None, title: str | None = None, html: bool = False,
             study: EventStudy | None = None, strategy: str = "") -> Path:
    """把一次回测写成一套文件, 并落一份 run.json 清单。

    产出: `nav.parquet` / `fills.parquet` / `orders.parquet` / `trades.parquet` /
    `performance.csv` / `trade_stats.csv` / `execution.csv` / `run.json`;
    `study` 非空时再写 `event_paths.parquet` / `event_summary.csv` /
    `event_tests.csv`; `html=True` 再写 `tearsheet.html`。

    Args:
        directory: 输出目录(自动创建)。
        result: engine.run 的输出。
        market: 年化口径; None 时取 `result.meta["market"]`, 再退到 "us"。
        benchmark: 基准代码; None 时取 `result.meta["benchmark"]`。
        title: HTML / 清单里的标题。
        html: 是否额外写自包含 tearsheet。
        study: 事件研究结果; 给了就把三张事件表一起写。
        strategy: 策略名, 写进清单。

    Returns:
        输出目录的 Path。
    """
    market = market or str(result.meta.get("market") or "us")
    if benchmark is None:
        benchmark = result.meta.get("benchmark") or None
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    result.nav.rename("nav").to_frame().to_parquet(out / "nav.parquet")
    result.fills.to_parquet(out / "fills.parquet")
    result.orders.to_parquet(out / "orders.parquet")
    order_trades = trades(result.fills, marks=last_marks(result.market))
    order_trades.to_parquet(out / "trades.parquet")
    summarize(result.nav, market=market,
              benchmark_nav=benchmark_nav(result, benchmark)).rename("value").to_frame().to_csv(
                  out / "performance.csv")
    trade_stats(order_trades).rename("value").to_frame().to_csv(out / "trade_stats.csv")
    execution_stats(result.orders, result.fills, result.nav).rename("value").to_frame().to_csv(
        out / "execution.csv")

    if study is not None:
        study.paths.to_parquet(out / "event_paths.parquet")
        study.summary.to_csv(out / "event_summary.csv", index=False)
        study.tests.to_csv(out / "event_tests.csv", index=False)

    written = sorted(path.name for path in out.iterdir() if path.is_file())
    if html:
        from event_backtest.figure import plot_tearsheet_html

        page = plot_tearsheet_html(result, market=market, benchmark=benchmark, title=title)
        page.save(out / "tearsheet.html")
        written = sorted(path.name for path in out.iterdir() if path.is_file())

    manifest = _manifest(result, market=market, benchmark=benchmark, files=written,
                         strategy=strategy, study=study)
    (out / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return out


def save_comparison(directory: str | Path,
                     runs: Mapping[str, Run | BacktestResult], *,
                     market: str | None = None, benchmark: str | None = None,
                     title: str = "场景对比", png: bool = True,
                     html: bool = True) -> Path:
    """把多场景结果写成一套文件(替掉脚本里那段样板)。

    产出: `comparison.csv`(每行一个场景) / `comparison.png` / `comparison.html` /
    `<场景名>.html`(每个场景一份 tearsheet)。

    Args:
        directory: 输出目录(自动创建)。
        runs: {场景名: Run 或 BacktestResult}。
        market: 年化口径; None 时从第一个 Run 里取, 再退到 "us"。
        benchmark: 基准代码; None 时从第一个 Run 里取。
        title: 图 / 页标题。
        png: 是否出 matplotlib 版对比图。
        html: 是否出 HTML 版对比页与各场景 tearsheet。

    Returns:
        输出目录的 Path。
    """
    from event_backtest.figure import plot_comparison, plot_comparison_html, plot_tearsheet_html
    from event_backtest.report import compare_results

    models = {str(name): (item.result if isinstance(item, Run) else item)
              for name, item in runs.items()}
    first = next(iter(runs.values()), None)
    if market is None:
        market = first.market if isinstance(first, Run) else str(
            first.meta.get("market") or "us") if first is not None else "us"
    if benchmark is None and isinstance(first, Run):
        benchmark = first.benchmark

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    compare_results(models, market=market, benchmark=benchmark).to_csv(out / "comparison.csv")
    if png:
        plot_comparison({name: item.nav for name, item in models.items()}).savefig(
            out / "comparison.png", dpi=120)
    if html:
        plot_comparison_html(models, market=market, benchmark=benchmark,
                             title=title).save(out / "comparison.html")
        for name, item in models.items():
            safe = name.replace("/", "_")
            plot_tearsheet_html(item, market=market, benchmark=benchmark,
                                title=name).save(out / f"{safe}.html")
    return out


def _manifest(result: BacktestResult, *, market: str, benchmark: str | None,
              files: list[str], strategy: str, study: EventStudy | None) -> dict[str, Any]:
    """run.json 的内容: 这次跑的是什么、用什么参数、结果如何。"""
    nav = result.nav.dropna()
    metrics = result_values(result, market=market, benchmark=benchmark)
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "package_version": _package_version(),
        "git_commit": _git_commit(),
        "market": market,
        "benchmark": benchmark,
        "strategy": strategy,
        "start": nav.index[0].isoformat() if len(nav) else None,
        "end": nav.index[-1].isoformat() if len(nav) else None,
        "bars": len(nav),
        "event_study": None if study is None else {"events": len(study.events)},
        "meta": jsonable(result.meta),
        "metrics": {key: jsonable(value) for key, value in metrics.items()},
        "files": files,
    }


def jsonable(value: Any) -> Any:
    """转成能进 JSON 的东西: numpy / NaN / Path / dataclass / 嵌套容器。

    Args:
        value: 任意对象。

    Returns:
        JSON 可序列化的等价物; 不认识的类型退化成 str; 非有限的 float 变成 None。
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def _package_version() -> str | None:
    """已安装的 event-backtest 版本; 拿不到就 None。"""
    try:
        return metadata.version("event-backtest")
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """当前 git commit(短哈希); 不在仓库里 / 读不到就 None。"""
    try:
        root = Path(__file__).resolve().parents[2]
        head_file = root / ".git" / "HEAD"
        if not head_file.is_file():
            return None
        head = head_file.read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(" ", 1)[1].strip()
        ref_file = root / ".git" / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()[:12]
        packed = root / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:12]
    except OSError:
        return None
    return None
