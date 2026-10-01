"""声明式策略: 从 strategy yaml(factors + signal + exit + sizing)构建可回测策略。

yaml 示例(与 minievent 的 factors / signal 格式一致)::

    name: 放量突破
    factors:
      - relative_volume: {window: 16}
      - momentum: {window: 8}
    signal:
      name: volume_breakout
      trigger: enter
      all:
        - {factor: relative_volume, operation: greater_than, value: 2}
    exit:
      hold_bars: 16
      take_profit: 0.06
      stop_loss: 0.04
    sizing:
      max_positions: 3

加载时做静态校验(因子存在、参数齐全、signal 只引用已启用因子); 事件在第 t 根收盘后
产生, 引擎在第 t+1 根撮合, 所以没有未来函数。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from event_backtest.broker import Fill
from event_backtest.engine import Context
from event_backtest.factor import FACTOR_DIR, FactorSpec, compute, load_specs, market_frames
from event_backtest.market import MarketData
from event_backtest.signal import SignalSpec, generate_events, parse_signal
from event_backtest.strategy.base import Strategy

__all__ = [
    "STRATEGY_DIR",
    "DeclarativeStrategy",
    "ExitRules",
    "FactorRef",
    "Sizing",
    "StrategyError",
    "StrategySpec",
    "WalkForward",
    "load_strategy",
    "parse_strategy",
]

STRATEGY_DIR = Path(__file__).parent
# 策略 yaml 允许的顶层字段
_TOP_KEYS = ("name", "walk_forward", "factors", "signal", "exit", "sizing")


class StrategyError(ValueError):
    """策略 yaml 写错(字段、因子、参数、signal 引用等)。"""


@dataclass(frozen=True)
class WalkForward:
    """walk-forward 切分参数(minievent design.md §8.2 已定的格式), 单位都是交易日。

    Attributes:
        folds: 切几折; 0 表示不做(解析时直接返回 None)。
        start: 第一折训练期的起点; None 表示用数据第一个交易日。
        train_days: 训练段长度(交易日)。
        test_days: 每折测试段长度(交易日)。
        embargo_days: 训练与测试之间的隔离期; 应 >= 最长持有期, 防止训练末尾开的仓
            带进测试段。
    """

    folds: int
    start: date | None
    train_days: int
    test_days: int
    embargo_days: int

    @property
    def required_days(self) -> int:
        """从 start 起至少需要的交易日数。"""
        return self.train_days + self.embargo_days + self.folds * self.test_days


_WF_KEYS = ("folds", "start", "train_days", "test_days", "embargo_days")


def parse_walk_forward(raw: Any, *, where: str) -> WalkForward | None:
    """解析 walk_forward 段; 不写或 folds: 0 返回 None(不做)。"""
    if raw is None:
        return None
    loc = f"{where}: walk_forward"
    if not isinstance(raw, dict):
        raise StrategyError(f"{loc} 必须是映射")
    _check_keys(raw, _WF_KEYS, loc)
    folds = _int(raw.get("folds"), f"{loc}.folds", minimum=0)
    if folds == 0:
        return None
    # start 可选: 不写就用数据第一个交易日, 一份策略可以在不同数据起点的市场上跑
    required = ("train_days", "test_days", "embargo_days")
    if missing := [k for k in required if k not in raw]:
        raise StrategyError(f"{loc} 缺字段 {missing}")
    start = raw.get("start")
    return WalkForward(
        folds=folds,
        start=_date(start, f"{loc}.start") if start is not None else None,
        train_days=_int(raw["train_days"], f"{loc}.train_days", minimum=1),
        test_days=_int(raw["test_days"], f"{loc}.test_days", minimum=1),
        embargo_days=_int(raw["embargo_days"], f"{loc}.embargo_days", minimum=0),
    )


def _int(value: Any, loc: str, *, minimum: int) -> int:
    """>= minimum 的整数; bool 和小数都不行。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrategyError(f"{loc} 必须是 >= {minimum} 的整数, 实际是 {value!r}")
    return value


def _date(value: Any, loc: str) -> date:
    """yaml 读出的日期(或 ISO 字符串)转成 date。"""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise StrategyError(f"{loc}: {value!r} 不是日期(如 2026-01-05)") from None


@dataclass(frozen=True)
class FactorRef:
    """策略里启用的一路因子。

    Attributes:
        name: 因子名(对应 factor/<name>.yaml)。
        params: 传给该因子的参数, 如 {"window": 16}。
        alias: 输出名; 写了 as 用别名, 否则就是因子名。signal 引用的是 alias。
    """

    name: str
    params: Mapping[str, Any]
    alias: str


@dataclass(frozen=True)
class ExitRules:
    """出场规则, 都为可选; 都不给表示一直持有到回测结束。

    Attributes:
        hold_bars: 开仓后最多持有多少根 bar。
        take_profit: 浮盈达到该比例就止盈(相对开仓价, 用后复权价比较)。
        stop_loss: 浮亏达到该比例就止损。
    """

    hold_bars: int | None = None
    take_profit: float | None = None
    stop_loss: float | None = None


@dataclass(frozen=True)
class Sizing:
    """仓位分配。

    Attributes:
        percent: 每个仓位占净值的比例; None 表示等权 1/N(N = 证券数)。
        max_positions: 最多同时持有几个仓位; None 表示不限。
    """

    percent: float | None = None
    max_positions: int | None = None


@dataclass(frozen=True)
class StrategySpec:
    """一个已校验的声明式策略。

    Attributes:
        name: 策略名(报告 / event_id 里用)。
        factors: 启用的因子列表, 按书写顺序。
        signal: 信号定义; None 表示没有信号(策略不下单)。
        exit: 出场规则。
        sizing: 仓位分配。
        walk_forward: walk-forward 切分参数; None 表示不做。
        factor_specs: {因子名: FactorSpec}, 加载时一次性读好并复用。
        path: 来源 yaml 路径。
    """

    name: str
    factors: tuple[FactorRef, ...]
    signal: SignalSpec | None
    exit: ExitRules
    sizing: Sizing
    walk_forward: WalkForward | None = None
    factor_specs: Mapping[str, FactorSpec] = field(repr=False, default_factory=dict)
    path: Path | None = None

    def compute(self, market: MarketData) -> dict[str, np.ndarray]:
        """算本策略启用的全部因子。

        Args:
            market: 对齐后的行情。

        Returns:
            {输出名(别名): (T, N) float64}; 一份行情里多个因子共用同一套宽表。
        """
        frames = market_frames(market)
        return {ref.alias: compute(self.factor_specs[ref.name], frames, ref.params)
                for ref in self.factors}

    def events(self, values: Mapping[str, np.ndarray], market: MarketData) -> pd.DataFrame:
        """把所有因子值变成事件表; 没有 signal 时返回带列名的空表。"""
        if self.signal is None:
            return pd.DataFrame(columns=["event_id", "bar", "ts", "ticker", "signal"])
        return generate_events(self.signal, values, market, strategy_name=self.name)


def load_strategy(which: str | Path, *, strategy_dir: str | Path = STRATEGY_DIR,
                  factor_dir: str | Path = FACTOR_DIR) -> StrategySpec:
    """读策略 yaml 并校验。

    校验内容: 顶层字段名、name 存在、factors 指向的因子存在且参数与因子声明一致、
    输出名不重复、signal 只引用已启用的输出名。全部在加载时报错, 不拖到回测中途。

    Args:
        which: 策略名(在 strategy_dir 找 <which>.yaml)或一个 yaml 路径。
        strategy_dir: 按名字查找时的目录。
        factor_dir: 因子 yaml 所在目录。

    Returns:
        StrategySpec。

    Raises:
        StrategyError: 文件不存在或任何一项校验不通过。
    """
    path = Path(which)
    if path.suffix not in (".yaml", ".yml"):
        path = Path(strategy_dir) / f"{which}.yaml"
    if not path.is_file():
        raise StrategyError(f"找不到策略 {str(which)!r}({path})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_strategy(raw, factor_dir=factor_dir, where=path.name, path=path)


def parse_strategy(raw: Any, *, factor_dir: str | Path = FACTOR_DIR,
                   where: str = "策略", path: Path | None = None) -> StrategySpec:
    """从已经读进来的 yaml 内容构建 StrategySpec。

    load_strategy 读完文件后调它; 参数扫描脚本可以先把 yaml 打补丁再调它, 不用写临时文件。

    Args:
        raw: yaml.safe_load 出来的内容。
        factor_dir: 因子 yaml 所在目录。
        where: 报错信息里的来源名(通常是文件名)。
        path: 来源路径; 只放进 StrategySpec 里备查。

    Returns:
        StrategySpec。

    Raises:
        StrategyError: 任何一项校验不通过。
    """
    if not isinstance(raw, dict):
        raise StrategyError(f"{where}: 顶层必须是一个映射")
    unknown = [k for k in raw if k not in _TOP_KEYS]
    if unknown:
        raise StrategyError(f"{where}: 不认识的字段 {unknown}, 可用的有 {list(_TOP_KEYS)}")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise StrategyError(f"{where}: 缺 name")

    # 先加载全部因子定义, 再解析本策略启用的那几路
    specs = load_specs(factor_dir)
    factors = _parse_factors(raw.get("factors"), specs, where)
    aliases = {ref.alias for ref in factors}
    if len(aliases) != len(factors):
        raise StrategyError(f"{where}: 因子输出名重复, 用 as 区分: {sorted(aliases)}")

    signal = parse_signal(raw.get("signal"), default_name=name)
    if signal is not None:
        unknown_factors = [f for f in signal.required_factors() if f not in aliases]
        if unknown_factors:
            raise StrategyError(f"{where}: signal 引用了未启用的因子 {unknown_factors}, "
                                f"已启用的是 {sorted(aliases)}")

    return StrategySpec(
        name=name, factors=factors, signal=signal,
        exit=_parse_exit(raw.get("exit"), where),
        sizing=_parse_sizing(raw.get("sizing"), where),
        walk_forward=parse_walk_forward(raw.get("walk_forward"), where=where),
        factor_specs=specs, path=path,
    )


def _parse_factors(raw: Any, specs: Mapping[str, FactorSpec],
                   where: str) -> tuple[FactorRef, ...]:
    """解析 factors 段: 支持 "- 因子名" 和 "- 因子名: {参数, as: 别名}" 两种写法。"""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise StrategyError(f"{where}: factors 必须是非空列表")
    refs: list[FactorRef] = []
    for number, item in enumerate(raw, start=1):
        loc = f"{where}: factors 第 {number} 项"
        if isinstance(item, str):
            factor_name, params = item, {}
        elif isinstance(item, dict) and len(item) == 1:
            (factor_name, body), = item.items()
            if body is not None and not isinstance(body, dict):
                raise StrategyError(f"{loc} 的参数必须是映射, 实际是 {body!r}")
            params = dict(body or {})
        else:
            raise StrategyError(f"{loc} 写法不对: {item!r}; 应为因子名, 或 {{因子名: {{参数}}}}")
        if factor_name not in specs:
            raise StrategyError(f"{loc}: 因子 {factor_name!r} 不存在, 可用的有 {sorted(specs)}")
        # as 是输出别名, 不传给因子; 其余参数的取名必须和因子声明完全一致
        alias = params.pop("as", factor_name)
        spec = specs[factor_name]
        missing = [p for p in spec.params if p not in params]
        extra = [p for p in params if p not in spec.params]
        if missing or extra:
            raise StrategyError(f"{loc}: 因子 {factor_name!r} 参数不对, 缺 {missing}, 多 {extra}, "
                                f"声明的是 {list(spec.params)}")
        refs.append(FactorRef(name=factor_name, params=params, alias=str(alias)))
    return tuple(refs)


def _parse_exit(raw: Any, where: str) -> ExitRules:
    """解析 exit 段(hold_bars / take_profit / stop_loss, 都可选)。"""
    if raw is None:
        return ExitRules()
    if not isinstance(raw, dict):
        raise StrategyError(f"{where}: exit 必须是映射")
    allowed = ("hold_bars", "take_profit", "stop_loss")
    _check_keys(raw, allowed, f"{where}: exit")
    hold = raw.get("hold_bars")
    if hold is not None and (isinstance(hold, bool) or not isinstance(hold, int) or hold <= 0):
        raise StrategyError(f"{where}: exit.hold_bars 必须是正整数, 实际是 {hold!r}")
    return ExitRules(
        hold_bars=hold,
        take_profit=_optional_pct(raw.get("take_profit"), f"{where}: exit.take_profit"),
        stop_loss=_optional_pct(raw.get("stop_loss"), f"{where}: exit.stop_loss"),
    )


def _parse_sizing(raw: Any, where: str) -> Sizing:
    """解析 sizing 段(percent / max_positions, 都可选)。"""
    if raw is None:
        return Sizing()
    if not isinstance(raw, dict):
        raise StrategyError(f"{where}: sizing 必须是映射")
    _check_keys(raw, ("percent", "max_positions"), f"{where}: sizing")
    percent = raw.get("percent")
    if percent is not None:
        # percent 是比例: 0.2 表示单个仓位占净值 20%; 写成 20 会报错
        if (isinstance(percent, bool) or not isinstance(percent, int | float)
                or not 0 < percent <= 1):
            raise StrategyError(f"{where}: sizing.percent 必须在 (0, 1], 实际是 {percent!r}")
        percent = float(percent)
    max_positions = raw.get("max_positions")
    if max_positions is not None and (isinstance(max_positions, bool)
                                      or not isinstance(max_positions, int) or max_positions <= 0):
        raise StrategyError(f"{where}: sizing.max_positions 必须是正整数, 实际是 {max_positions!r}")
    return Sizing(percent=percent, max_positions=max_positions)


def _optional_pct(value: Any, loc: str) -> float | None:
    """可选的比例字段: None 直接返回 None, 否则必须在 (0, 1) 之间。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise StrategyError(f"{loc} 必须在 (0, 1) 之间, 实际是 {value!r}")
    return float(value)


def _check_keys(raw: Mapping[str, Any], allowed: Sequence[str], loc: str) -> None:
    """有不认识的字段就报错并列出可用字段。"""
    if unknown := [k for k in raw if k not in allowed]:
        raise StrategyError(f"{loc} 有不认识的字段 {unknown}, 可用的有 {list(allowed)}")


class DeclarativeStrategy(Strategy):
    """把 StrategySpec 接到引擎上的适配器: 事件入场 + exit 规则出场。

    初始化时一次性算好全部因子与事件, 回测中每根 bar 只做两件事: 检查出场、检查入场。
    """

    def __init__(self, spec: StrategySpec, *,
                 event_bars: list[set[int]] | None = None) -> None:
        """

        Args:
            spec: 策略定义。
            event_bars: 可选; 每只证券有事件的 bar 下标集合(相对本次要跑的行情)。
                给了就直接用, 不再从 ctx.market 现场算 —— walk-forward 用它把
                "在更长历史上算好的事件"注入到只含测试段的行情上。
        """
        self.spec = spec
        self._event_bars = event_bars

    def initialize(self, ctx: Context) -> None:
        """预计算因子与事件, 并把"每只证券在第几根 bar 有事件"整理成集合。"""
        self.n = len(ctx.tickers)
        if self._event_bars is not None:
            self.event_bars = self._event_bars
        else:
            values = self.spec.compute(ctx.market)
            events = self.spec.events(values, ctx.market)
            index = {t: i for i, t in enumerate(ctx.tickers)}
            self.event_bars = [set() for _ in range(self.n)]
            for row in events.to_dict("records"):
                self.event_bars[index[str(row["ticker"])]].add(int(row["bar"]))
        # 记录每只证券当前这笔持仓的开仓 bar 与开仓价(后复权), 供 exit 规则判断
        self.entry_bar = np.full(self.n, -1, dtype=int)
        self.entry_px = np.full(self.n, np.nan)

    def handle_data(self, ctx: Context) -> None:
        """先处理出场(腾出仓位), 再处理入场。"""
        self._handle_exits(ctx)
        self._handle_entries(ctx)

    def on_fill(self, fill: Fill, ctx: Context) -> None:
        """成交回报: 买入时记开仓信息, 卖到空仓时清掉。"""
        i = fill.index
        if fill.amount > 0:
            # 只在建仓的第一笔记开仓价(加仓不覆盖)
            if self.entry_bar[i] < 0:
                self.entry_bar[i] = fill.bar
                self.entry_px[i] = float(ctx.market.adj_close[fill.bar, i])
        elif ctx.portfolio.total[i] <= 0:
            self.entry_bar[i] = -1
            self.entry_px[i] = np.nan

    def _handle_exits(self, ctx: Context) -> None:
        """对每个持仓检查 hold_bars / take_profit / stop_loss, 满足就市价平掉。"""
        rules = self.spec.exit
        for i in range(self.n):
            if ctx.position(i) <= 0:
                continue
            price = ctx.price(i, "adj_close")
            entry = self.entry_px[i]
            reason = None
            # 用后复权收盘价判断(避免除权造成假信号); 优先级: 止损 > 止盈 > 到期。
            # TP 与 SL 不可能同时成立: 一个要求 price >= entry*(1+tp)、另一个要求
            # price <= entry*(1-sl), 而 tp/sl 都为正。hold_bars 与它们同时成立时,
            # 让位于 TP/SL(先按风险/收益处理, 时间到不是最紧迫的)。
            if (rules.stop_loss is not None and np.isfinite(entry)
                    and price <= entry * (1 - rules.stop_loss)):
                reason = "stop_loss"
            elif (rules.take_profit is not None and np.isfinite(entry)
                    and price >= entry * (1 + rules.take_profit)):
                reason = "take_profit"
            elif (rules.hold_bars is not None and self.entry_bar[i] >= 0
                    and ctx.bar - self.entry_bar[i] >= rules.hold_bars):
                reason = "hold_bars"
            if reason is not None:
                # 卖出数量 = 当前全部持仓; 下一根撮合, 没成交(如跌停)会在下一根重试。
                # reason 记进委托明细, 期末复盘能看到"这笔平仓是止损还是到期"。
                ctx.order(i, -int(ctx.position(i)), tag=f"exit:{reason}")

    def _handle_entries(self, ctx: Context) -> None:
        """对每只有事件的证券, 在空仓且没超过持仓上限时按 sizing 建仓。"""
        percent = self.spec.sizing.percent
        max_positions = self.spec.sizing.max_positions
        # 只有设了上限才需要数当前持仓数
        held = int((ctx.portfolio.total > 0).sum()) if max_positions is not None else 0
        for i in range(self.n):
            if ctx.bar not in self.event_bars[i] or ctx.position(i) > 0:
                continue
            if max_positions is not None and held >= max_positions:
                continue
            ctx.order_target_percent(i, percent if percent is not None else 1.0 / self.n,
                                     tag="enter")
            held += 1
