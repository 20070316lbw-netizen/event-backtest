"""策略: 回调基类、内置代码策略与声明式 YAML 策略。"""
from __future__ import annotations

from collections.abc import Mapping
from inspect import signature
from pathlib import Path

from event_backtest.engine import ContextStrategy
from event_backtest.strategy.base import (
    STRATEGIES,
    BuyHold,
    SmaCross,
    Strategy,
    make_strategy,
)
from event_backtest.strategy.declarative import (
    STRATEGY_DIR,
    DeclarativeStrategy,
    ExitRules,
    FactorRef,
    Sizing,
    StrategyError,
    StrategySpec,
    WalkForward,
    load_strategy,
    parse_strategy,
)


def _init_params(cls: type) -> set[str]:
    """策略类构造函数显式声明的参数名(忽略 *args / **kwargs 这类兜底形参)。"""
    return {name for name, parameter in signature(cls.__init__).parameters.items()
            if name != "self"
            and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)}


def _accepted_params(name: str, params: Mapping[str, object]) -> dict[str, object]:
    """挑出某个内置策略构造函数认识的参数。

    调用者往往把所有策略的参数一起传(比如 CLI 的 --fast/--slow), 所以这里按签名过滤;
    但**谁都不认识**的参数名仍然报错, 免得把拼写错误吞掉。
    """
    known = set().union(*(_init_params(cls) for cls in STRATEGIES.values()))
    if unknown := sorted(set(params) - known):
        raise StrategyError(
            f"策略参数 {unknown} 不是任何内置策略认识的; 可用的是 {sorted(known)}")
    accepted = _init_params(STRATEGIES[name])
    return {key: value for key, value in params.items() if key in accepted}


def resolve_strategy(which: str | Path | StrategySpec | ContextStrategy,
                     **params: object) -> tuple[ContextStrategy, StrategySpec | None]:
    """把"策略怎么写"统一解析成 (可跑的策略对象, 声明式 spec 或 None)。

    这是给调用者的一个便利入口(CLI 与 EventBackTest 都用它), 优先级:
    yaml 路径 -> strategy/ 下的 yaml 名 -> 内置代码策略名 -> 原样当作策略对象。

    Args:
        which: yaml 路径 / yaml 名 / 内置策略名(buy_hold / sma_cross) / 策略对象 /
            StrategySpec。
        **params: 只透传给内置代码策略的构造函数(如 sma_cross 的 fast / slow)。

    Returns:
        (策略对象, 声明式 spec 或 None); spec 供事件研究用, 代码策略为 None。

    Raises:
        StrategyError: 字符串既不是 yaml, 也不在 STRATEGIES 里。
    """
    if isinstance(which, StrategySpec):
        return DeclarativeStrategy(which), which
    if isinstance(which, (str, Path)):
        text = str(which)
        path = Path(text)
        if path.suffix in (".yaml", ".yml") and path.is_file():
            spec = load_strategy(path)
            return DeclarativeStrategy(spec), spec
        if (STRATEGY_DIR / f"{text}.yaml").is_file():
            spec = load_strategy(text)
            return DeclarativeStrategy(spec), spec
        if text in STRATEGIES:
            return make_strategy(text, **_accepted_params(text, params)), None
        raise StrategyError(
            f"未知策略 {text!r}: 既不是 {STRATEGY_DIR} 下的 yaml, 也不在 {sorted(STRATEGIES)}")
    return which, None


__all__ = [
    "STRATEGIES",
    "STRATEGY_DIR",
    "BuyHold",
    "DeclarativeStrategy",
    "ExitRules",
    "FactorRef",
    "Sizing",
    "SmaCross",
    "Strategy",
    "StrategyError",
    "StrategySpec",
    "WalkForward",
    "load_strategy",
    "make_strategy",
    "parse_strategy",
    "resolve_strategy",
]
