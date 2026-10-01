"""EventBackTest: 超参数式的一次回测入口(给调用者的便利层)。

设计原则: **先有组件, 才有这个类**。
    - 组件: `load_config` / `MarketConfig.load` / `engine.run` / `resolve_strategy` /
      `study_events` / `run_walk_forward` / `search` / `Run` / `save_run` —— 开发者
      要细粒度控制时照旧单独用;
    - 门面: 这个类只负责"把组件按一次完整运行组装起来", 所以调用者写
      `EventBackTest(market="us", start=..., strategy="volume_breakout").run()`
      就有补全、少犯错, 而不用记住组件之间的顺序与参数。

用法::

    backtest = EventBackTest(
        market="us", start="2016-10-01", end="2026-09-30",
        strategy="volume_breakout", initial_cash=1_000_000,
    )
    run = backtest.run()               # -> Run(净值 / 成交 / 委托 / 上下文)
    run.summary()["sharpe"]
    run.save("outputs/us_full", html=True)

    backtest.study(bootstrap=2000)                 # 事件研究(带显著性)
    backtest.walk_forward()                        # 样本外
    backtest.sweep({"exit.hold_bars": [8, 16, 32]})  # 扫参
    backtest.compare({"基准": {}, "滑点0.05": {"slippage": FixedSlippage(0.05)}})
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from event_backtest.config import MarketConfig, load_config
from event_backtest.engine import run as run_backtest
from event_backtest.evaluation.events import DEFAULT_HORIZONS, EventStudy, study_events
from event_backtest.fees import FeeSchedule
from event_backtest.market import DateLike, MarketData
from event_backtest.results import Run
from event_backtest.rules import MarketRules
from event_backtest.slippage import SlippageModel
from event_backtest.strategy import (
    STRATEGY_DIR,
    StrategyError,
    StrategySpec,
    resolve_strategy,
)
from event_backtest.sweep import SweepResult, search
from event_backtest.walkforward import WalkForwardResult, run_walk_forward

__all__ = ["UNSET", "EventBackTest"]

# 参数默认值用 UNSET 表示"没传, 用配置里的"; 显式传 None 才表示"清空"
# (例如 universe=None = 库里全部证券, benchmark=None = 不要基准)
UNSET: Any = object()


def _strategy_name(which: Any) -> str:
    """给 Run / 报告用的策略名。"""
    if isinstance(which, StrategySpec):
        return which.name
    if isinstance(which, (str, Path)):
        return Path(str(which)).stem
    return type(which).__name__


class EventBackTest:
    """一次回测的超参数容器 + 入口。

    Args:
        market: 市场名("cn" / "us", 对应 setting/<market>.yaml), 或直接给一个
            MarketConfig; 给 None 表示"不要市场配置"(配合 market_data 用, 此时
            扫参 / walk-forward 这类需要配置的能力会明确报错)。
        config: 市场配置或配置路径; 给了就忽略 market 的查找。
        market_data: 已经加载好的行情; 给了就不碰数据库(测试 / 复用同一段数据)。
        db_path: 覆盖配置里的数据库路径。
        universe: 证券池; 不传用配置里的, 显式 None = 库里的全部证券。
        benchmark: 基准代码; 不传用配置里的, 显式 None = 不要基准。
        start / end: 回测区间的默认值(交易日, 两端含); run() 时还可以再覆盖。
        freq: 分钟线周期; 不传用配置里的, 显式 None = 日线。
        initial_cash / fill_price / slippage / fees / rules: 覆盖配置里的交易口径。
        strategy: 默认策略(yaml 名 / yaml 路径 / 内置代码策略名 / 策略对象)。
        seed: 预留给随机性(bootstrap 等)的默认种子。
    """

    def __init__(self, market: str | MarketConfig | Path | None = "us", *,
                 config: MarketConfig | str | Path | None = None,
                 market_data: MarketData | None = None,
                 db_path: str | Path | None = None,
                 universe: Sequence[str] | None = UNSET,
                 benchmark: str | None = UNSET,
                 start: DateLike | None = None,
                 end: DateLike | None = None,
                 freq: str | int | None = UNSET,
                 initial_cash: float | None = None,
                 fill_price: str | None = None,
                 slippage: SlippageModel | None = None,
                 fees: Mapping[str, FeeSchedule] | None = None,
                 rules: MarketRules | None = None,
                 strategy: Any = None,
                 seed: int = 0) -> None:
        base = config if config is not None else market
        if isinstance(base, MarketConfig):
            resolved: MarketConfig | None = base
        elif base is None:
            resolved = None
        else:
            resolved = load_config(base)

        if resolved is not None:
            overrides: dict[str, Any] = {}
            if db_path is not None:
                overrides["db_path"] = db_path
            if universe is not UNSET:
                overrides["tickers"] = tuple(universe) if universe else None
            if benchmark is not UNSET:
                overrides["benchmark"] = benchmark
            if freq is not UNSET:
                overrides["freq"] = str(freq) if freq is not None else None
            if initial_cash is not None:
                overrides["initial_cash"] = float(initial_cash)
            if fill_price is not None:
                overrides["fill_price"] = fill_price
            if slippage is not None:
                overrides["slippage"] = slippage
            if fees is not None:
                overrides["fees"] = dict(fees)
            if rules is not None:
                overrides["rules"] = rules
            if overrides:
                resolved = replace(resolved, **overrides)

        self.config = resolved
        self.market_data = market_data
        self.start = start
        self.end = end
        self.freq = None if freq is UNSET else freq
        self.strategy = strategy
        self.seed = seed
        self.initial_cash = initial_cash
        self.fill_price = fill_price
        self.slippage = slippage
        self.fees = dict(fees) if fees is not None else None
        self.rules = rules
        self._name = resolved.market if resolved is not None else "us"
        self._benchmark: str | None = resolved.benchmark if resolved is not None else None
        self._cache: dict[tuple[Any, Any, Any], MarketData] = {}

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_market(cls, market: MarketData, *, market_name: str = "us",
                    benchmark: str | None = None, initial_cash: float = 1_000_000.0,
                    fill_price: str = "open", slippage: SlippageModel | None = None,
                    fees: Mapping[str, FeeSchedule] | None = None,
                    rules: MarketRules | None = None, strategy: Any = None,
                    seed: int = 0) -> EventBackTest:
        """用一份已加载的行情建实例(不碰数据库)。

        Args:
            market: 对齐后的行情。
            market_name: "cn" / "us", 决定年化口径。
            benchmark: 基准代码(可选)。
            initial_cash / fill_price / slippage / fees / rules: 交易口径。
            strategy: 默认策略。
            seed: 随机种子。

        Returns:
            EventBackTest。
        """
        # market=None: 不带市场配置, 只用这份行情与显式传进来的交易口径
        instance = cls(None, market_data=market, initial_cash=initial_cash,
                       fill_price=fill_price, slippage=slippage, fees=fees, rules=rules,
                       strategy=strategy, seed=seed)
        instance._name = market_name
        instance._benchmark = benchmark
        return instance

    # ---------------------------------------------------------------- 行情
    @property
    def name(self) -> str:
        """市场名("cn" / "us")。"""
        return self._name

    @property
    def benchmark(self) -> str | None:
        """基准代码。"""
        return self.config.benchmark if self.config is not None else self._benchmark

    def load_market(self, *, start: DateLike | None = None, end: DateLike | None = None,
                    freq: str | int | None = None) -> MarketData:
        """读出行情(同一组参数只读一次)。

        Args:
            start / end: 覆盖构造时的区间。
            freq: 覆盖构造时的周期。

        Returns:
            MarketData。

        Raises:
            ValueError: 既没有配置也没有已加载的行情。
        """
        if self.market_data is not None and start is None and end is None and freq is None:
            return self.market_data
        if self.config is None:
            raise ValueError("没有市场配置也没有 market_data, 加载不了行情")
        key = (start or self.start, end or self.end,
               self.freq if freq is None else freq)
        if key not in self._cache:
            self._cache[key] = self.config.load(start=key[0], end=key[1], freq=key[2])
        return self._cache[key]

    # ---------------------------------------------------------------- 跑一遍
    def run(self, strategy: Any = UNSET, *, market: MarketData | None = None,
            start: DateLike | None = None, end: DateLike | None = None,
            freq: str | int | None = None, **strategy_params: Any) -> Run:
        """跑一遍回测, 返回带上下文的 Run。

        Args:
            strategy: 覆盖构造时的默认策略。
            market: 直接用这份行情(给了就不读数据库)。
            start / end / freq: 覆盖区间与周期。
            **strategy_params: 透传给内置代码策略(如 sma_cross 的 fast / slow)。

        Returns:
            Run。
        """
        target = self.strategy if strategy is UNSET else strategy
        if target is None:
            raise ValueError("没有指定策略: 构造时给 strategy=, 或 run(strategy=...)")
        strategy_object, spec = resolve_strategy(target, **strategy_params)
        name = _strategy_name(target)
        data = (market if market is not None
                else self.load_market(start=start, end=end, freq=freq))

        result = run_backtest(
            strategy_object, data,
            initial_cash=(self.initial_cash if self.initial_cash is not None
                          else (self.config.initial_cash if self.config else 1_000_000.0)),
            fees=(self.fees if self.fees is not None
                  else (self.config.resolved_fees() if self.config else None)),
            fill_price=(self.fill_price if self.fill_price is not None
                        else (self.config.fill_price if self.config else "open")),
            slippage=self.slippage if self.slippage is not None
            else (self.config.slippage if self.config else None),
            rules=(self.rules if self.rules is not None
                   else (self.config.resolved_rules() if self.config else None)),
        )
        if self.config is not None:
            result.meta.update(self.config.describe())
        result.meta.setdefault("market", self.name)
        result.meta.setdefault("benchmark", self.benchmark)
        result.meta["strategy"] = name
        return Run(result=result, config=self.config, strategy=name, strategy_spec=spec)

    # ---------------------------------------------------------------- 事件研究 / 样本外 / 扫参
    def study(self, strategy: Any = UNSET, *, horizons: tuple[int, ...] | None = None,
              market: MarketData | None = None, **kwargs: Any) -> EventStudy:
        """跑事件研究(带显著性检验), 不跑回测。

        Args:
            strategy: 覆盖默认策略; 必须是声明式 YAML 策略。
            horizons: 观察窗口; None 用默认 1..16。
            market: 直接用这份行情。
            **kwargs: 透传给 study_events(如 bootstrap / alpha / seed / cluster)。

        Returns:
            EventStudy。

        Raises:
            ValueError: 策略不是声明式的(没有 spec), 做不了事件研究。
        """
        target = self.strategy if strategy is UNSET else strategy
        if target is None:
            raise ValueError("没有指定策略")
        _, spec = resolve_strategy(target)
        if spec is None:
            raise ValueError(f"{_strategy_name(target)!r} 不是声明式 YAML 策略, 做不了事件研究")
        data = market if market is not None else self.load_market()
        events = spec.events(spec.compute(data), data)
        return study_events(data, events, horizons=horizons or DEFAULT_HORIZONS,
                            strategy_name=spec.name, **kwargs)

    def walk_forward(self, strategy: Any = UNSET, *, select: Any = None,
                     market: MarketData | None = None, start: DateLike | None = None,
                     end: DateLike | None = None) -> WalkForwardResult:
        """跑 walk-forward(策略 yaml 里要有 walk_forward 段)。

        Args:
            strategy: 覆盖默认策略; 必须是带 walk_forward 段的声明式策略。
            select: 参数选择器 (spec, 训练段行情) -> spec; None = 每折沿用同一份。
            market: 直接用这份行情。
            start / end: 覆盖区间。

        Returns:
            WalkForwardResult。

        Raises:
            ValueError: 没有市场配置, 或策略没有 walk_forward 段。
        """
        if self.config is None:
            raise ValueError("walk-forward 需要市场配置(费率 / 规则 / 数据来源)")
        target = self.strategy if strategy is UNSET else strategy
        if target is None:
            raise ValueError("没有指定策略")
        _, spec = resolve_strategy(target)
        if spec is None:
            raise ValueError(f"{_strategy_name(target)!r} 不是声明式 YAML 策略")
        data = market if market is not None else self.load_market(start=start, end=end)
        return run_walk_forward(self.config, spec, market=data, select=select)

    def sweep(self, grid: Mapping[str, Sequence[object]], *, strategy: Any = UNSET,
              metric: str = "sharpe", market: MarketData | None = None) -> SweepResult:
        """在同一段行情上扫参数网格(样本内; 要样本外请用 walk_forward + select)。

        Args:
            grid: {策略 yaml 的点路径: 候选取值}, 如 {"exit.hold_bars": [8, 16, 32]}。
            strategy: 覆盖默认策略; 必须来自 yaml(点路径是打在 yaml 上的)。
            metric: 选优 / 排序依据, 可选 evaluate_spec 里的列。
            market: 直接用这份行情。

        Returns:
            SweepResult。

        Raises:
            ValueError: 没有市场配置, 或策略不是从 yaml 来的。
        """
        if self.config is None:
            raise ValueError("扫参需要市场配置")
        target = self.strategy if strategy is UNSET else strategy
        if target is None:
            raise ValueError("没有指定策略")
        raw = _strategy_raw(target)
        data = market if market is not None else self.load_market()
        return search(raw, grid, self.config, data, metric=metric)

    def compare(self, scenarios: Mapping[str, Mapping[str, Any]], *,
                market: MarketData | None = None, start: DateLike | None = None,
                end: DateLike | None = None) -> dict[str, Run]:
        """一份公共配置派生多个场景, 各跑一遍; 结果可直接喂 compare_results / HTML。

        场景里的覆盖有两种键:
            - `strategy`: 这个场景用哪个策略;
            - 其余键: 覆盖 MarketConfig 的字段(如 `slippage` / `initial_cash` / `fees`)。

        Args:
            scenarios: {场景名: {覆盖参数}}; 空 dict = 基准场景。
            market: 直接用这份行情(所有场景共用)。
            start / end: 覆盖区间。

        Returns:
            {场景名: Run}; 交给 figure.plot_comparison_html 或 report.compare_results。

        Raises:
            ValueError: 没有市场配置。
        """
        if self.config is None:
            raise ValueError("场景对比需要市场配置")
        data = market if market is not None else self.load_market(start=start, end=end)
        runs: dict[str, Run] = {}
        for name, scenario in scenarios.items():
            overrides = dict(scenario)
            target = overrides.pop("strategy", self.strategy)
            if target is None:
                raise ValueError(
                    f"场景 {name!r} 没有策略: 给 scenario['strategy'] 或构造时的 strategy")
            cfg = replace(self.config, **overrides) if overrides else self.config
            strategy_object, spec = resolve_strategy(target)
            result = run_backtest(
                strategy_object, data, initial_cash=cfg.initial_cash,
                fees=cfg.resolved_fees(), fill_price=cfg.fill_price,
                slippage=cfg.slippage, rules=cfg.resolved_rules())
            result.meta.update(cfg.describe())
            result.meta["strategy"] = _strategy_name(target)
            runs[str(name)] = Run(result=result, config=cfg,
                                  strategy=_strategy_name(target), strategy_spec=spec)
        return runs

    def __repr__(self) -> str:
        """调试用的简短描述。"""
        return (f"EventBackTest(market={self.name!r}, strategy={self.strategy!r}, "
                f"start={self.start!r}, end={self.end!r})")


def _strategy_raw(which: Any) -> Any:
    """取出策略的 yaml 原始内容(扫参的点路径是打在它上面的)。

    Raises:
        StrategyError: 不是 yaml 文件也不是 strategy/ 下的 yaml 名。
    """
    if isinstance(which, (str, Path)):
        path = Path(str(which))
        if path.suffix in (".yaml", ".yml") and path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        candidate = STRATEGY_DIR / f"{which}.yaml"
        if candidate.is_file():
            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raise StrategyError(
        f"扫参需要策略 yaml(点路径打在 yaml 上), 收到 {which!r}; "
        f"可用的有 {sorted(p.stem for p in STRATEGY_DIR.glob('*.yaml'))}")
