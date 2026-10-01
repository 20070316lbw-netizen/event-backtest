"""信号: 把因子值按条件变成事件。

signal yaml(写在策略文件里)字段:
    name      信号名, 用于拼 event_id
    trigger   enter(默认, 条件由不满足变满足时触发一次) / every_bar(每根满足都触发)
    all / any 条件列表, 二选一; 每条 {factor, operation, value}

任一条件因子为 NaN / inf 即视为不满足(而不是"恰好命中")。条件用第 t 根收盘信息,
不额外 shift; 第 t 根出事件, 第 t+1 根才成交(引擎负责), 所以没有未来函数。

Example:
    >>> from event_backtest.signal import generate_events, parse_signal
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from event_backtest.market import MarketData

__all__ = ["Condition", "SignalError", "SignalSpec", "generate_events", "parse_signal"]

# 比较算子 -> numpy 函数; 条件里 operation 写这些名字
_OPS = {
    "greater_than": np.greater,
    "greater_than_or_equal": np.greater_equal,
    "less_than": np.less,
    "less_than_or_equal": np.less_equal,
    "equal": np.equal,
    "not_equal": np.not_equal,
}
_TRIGGERS = ("enter", "every_bar")
# 事件表的列
_EVENT_COLUMNS = ["event_id", "bar", "ts", "ticker", "signal"]


class SignalError(ValueError):
    """signal 段写错, 或引用了不存在的因子。"""


@dataclass(frozen=True)
class Condition:
    """一条信号条件: 某因子的值 与 阈值 做一次比较。

    Attributes:
        factor: 因子输出名(策略里启用因子的别名)。
        operation: 比较算子, 见 _OPS。
        value: 阈值, 必须是有限数值。
    """

    factor: str
    operation: str
    value: float


@dataclass(frozen=True)
class SignalSpec:
    """一个已校验的信号定义。

    Attributes:
        name: 信号名。
        trigger: "enter" 或 "every_bar"。
        logic: "all"(全部条件满足) 或 "any"(任一满足)。
        conditions: 条件列表, 至少一条。
    """

    name: str
    trigger: str
    logic: str
    conditions: tuple[Condition, ...]

    def required_factors(self) -> tuple[str, ...]:
        """本信号引用到的因子名(去重, 按出现顺序); 策略加载器用它校验因子是否启用。"""
        return tuple(dict.fromkeys(c.factor for c in self.conditions))


def parse_signal(raw: Any, *, default_name: str = "signal") -> SignalSpec | None:
    """校验 signal 段。

    Args:
        raw: 策略 yaml 里 signal 段的内容; None 表示这个策略没有信号。
        default_name: 没写 name 时用的名字。

    Returns:
        SignalSpec; raw 为 None 时返回 None。

    Raises:
        SignalError: 不是映射、字段不认识、all / any 不是恰好一个、条件为空或写错。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SignalError("signal 必须是一个映射")
    unknown = [k for k in raw if k not in ("name", "trigger", "all", "any")]
    if unknown:
        raise SignalError(f"signal 有不认识的字段 {unknown}, 可用 name / trigger / all / any")

    # all / any 必须二选一, 不允许同时写或都不写
    logic_keys = [k for k in ("all", "any") if k in raw]
    if len(logic_keys) != 1:
        raise SignalError(f"signal 必须恰好有 all / any 之一, 实际有 {logic_keys}")
    logic = logic_keys[0]
    body = raw[logic]
    if not isinstance(body, list) or not body:
        raise SignalError(f"signal.{logic} 必须是非空列表")

    trigger = str(raw.get("trigger", "enter"))
    if trigger not in _TRIGGERS:
        raise SignalError(f"signal.trigger 只能是 {list(_TRIGGERS)}, 实际是 {trigger!r}")

    return SignalSpec(
        name=str(raw.get("name") or default_name),
        trigger=trigger,
        logic=logic,
        conditions=tuple(_parse_condition(c, i, logic) for i, c in enumerate(body, start=1)),
    )


def _parse_condition(raw: Any, number: int, logic: str) -> Condition:
    """校验 all / any 里的一条条件。"""
    loc = f"signal.{logic} 第 {number} 条"
    if not isinstance(raw, dict):
        raise SignalError(f"{loc} 必须是映射")
    unknown = [k for k in raw if k not in ("factor", "operation", "value")]
    if unknown:
        raise SignalError(f"{loc} 有不认识的字段 {unknown}, 可用 factor / operation / value")
    factor = raw.get("factor")
    if not isinstance(factor, str) or not factor:
        raise SignalError(f"{loc} 缺 factor")
    operation = raw.get("operation")
    if operation not in _OPS:
        raise SignalError(f"{loc} 的 operation {operation!r} 不认识, 可用的有 {sorted(_OPS)}")
    value = raw.get("value")
    # bool 是 int 的子类, 单独排除; NaN / inf 也不能当阈值
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise SignalError(f"{loc} 的 value 必须是有限数值, 实际是 {value!r}")
    return Condition(factor=factor, operation=operation, value=float(value))


def generate_events(signal: SignalSpec, values: Mapping[str, np.ndarray],
                    market: MarketData, *, strategy_name: str = "") -> pd.DataFrame:
    """因子值 -> 事件表。

    Args:
        signal: parse_signal 的结果。
        values: {因子输出名: (T, N) 数组}, 必须含信号引用的全部因子。
        market: 对齐后的行情(取时间轴与证券代码, 并校验形状一致)。
        strategy_name: 用于拼 event_id; 策略名 + 信号名 + 证券 + 时间唯一确定一条事件。

    Returns:
        DataFrame, 列 [event_id, bar, ts, ticker, signal], 按 (bar, 证券) 升序;
        没有事件时是带列名的空表。

    Raises:
        SignalError: 引用的因子没有算, 或因子形状和行情不一致。
    """
    combined = _combined_mask(signal, values, market)
    # enter: 只保留"上一根不满足、这一根满足"的上升沿; every_bar: 满足就记
    active = combined & ~_previous(combined) if signal.trigger == "enter" else combined

    bars, columns = np.nonzero(active)   # 行 = bar 下标, 列 = 证券下标
    rows = [
        {
            "event_id": f"{strategy_name}|{signal.name}|{market.tickers[c]}|"
                        f"{pd.Timestamp(market.ts[b]).isoformat()}",
            "bar": int(b), "ts": pd.Timestamp(market.ts[b]),
            "ticker": market.tickers[c], "signal": signal.name,
        }
        for b, c in zip(bars, columns, strict=True)
    ]
    return pd.DataFrame(rows, columns=_EVENT_COLUMNS)


def _combined_mask(signal: SignalSpec, values: Mapping[str, np.ndarray],
                   market: MarketData) -> np.ndarray:
    """把各条条件算成布尔矩阵, 再按 all / any 合并。

    NaN / inf 先被 valid 掩掉, 所以 not_equal 之类也不会把"缺数据"当成满足。
    """
    mask: np.ndarray | None = None
    for condition in signal.conditions:
        if condition.factor not in values:
            raise SignalError(f"signal {signal.name!r} 引用的因子 {condition.factor!r} 没有算; "
                              f"可用的是 {sorted(values)}")
        arr = np.asarray(values[condition.factor], dtype=float)
        if arr.shape != market.shape:
            raise SignalError(
                f"因子 {condition.factor!r} 形状 {arr.shape} 与行情 {market.shape} 不一致")
        valid = np.isfinite(arr)
        cond = valid & _OPS[condition.operation](arr, condition.value)
        mask = cond if mask is None else (mask & cond if signal.logic == "all" else mask | cond)
    if mask is None:
        raise SignalError(f"signal {signal.name!r} 没有条件")
    return mask


def _previous(mask: np.ndarray) -> np.ndarray:
    """上一根 bar 的掩码; 第一根视为 False(所以第一根满足也会触发 enter)。"""
    prev = np.zeros_like(mask)
    prev[1:] = mask[:-1]
    return prev
