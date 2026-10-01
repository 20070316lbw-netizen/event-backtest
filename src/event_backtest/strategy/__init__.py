"""策略: 回调基类、内置代码策略与声明式 YAML 策略。"""
from __future__ import annotations

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
]
