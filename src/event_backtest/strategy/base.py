"""策略基类与内置代码策略(声明式 YAML 见 declarative.py)。

公开接口是声明式 YAML(不写代码); 这里的 Strategy 是引擎的回调协议, 声明式适配器与
内置示例策略都实现它, 也方便测试时直接写个临时策略。
"""
from __future__ import annotations

import numpy as np

from event_backtest.broker import Fill
from event_backtest.engine import Context


class Strategy:
    """策略协议: 实现 handle_data 即可, initialize / on_fill 可选。"""

    def initialize(self, ctx: Context) -> None:
        """回测开始前调用一次, 用来建状态(如均线窗口、事件集合)。

        Args:
            ctx: 策略上下文, 此时 bar 为 0、尚未成交任何订单。
        """

    def handle_data(self, ctx: Context) -> None:
        """每根 bar 调用一次; 用 ctx.order / ctx.order_target_percent 下单。

        Args:
            ctx: 策略上下文, ctx.bar 是当前 bar 下标。
        """
        raise NotImplementedError

    def on_fill(self, fill: Fill, ctx: Context) -> None:
        """成交回报, 每次成交调用一次; 声明式策略用它记录开仓价与开仓 bar。

        Args:
            fill: 刚发生的成交。
            ctx: 策略上下文。
        """


class BuyHold(Strategy):
    """等权买入并持有: 第一根 bar 就把净值平均分配到全部证券, 之后不再调仓。"""

    def initialize(self, ctx: Context) -> None:
        self.target = 1.0 / len(ctx.tickers)

    def handle_data(self, ctx: Context) -> None:
        # 每根 bar 都调一次目标仓位; 已经到位时 order_target_percent 算出来是 0, 不会下单
        for i in range(len(ctx.tickers)):
            ctx.order_target_percent(i, self.target)


class SmaCross(Strategy):
    """双均线交叉: 快线在慢线上方则等权持有, 否则清仓。信号用后复权收盘价。

    Args:
        fast: 快线窗口(bar 数), 必须为正且小于 slow。
        slow: 慢线窗口(bar 数)。

    Raises:
        ValueError: fast / slow 不为正, 或 fast >= slow。
    """

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError(f"需要 0 < fast < slow, 收到 fast={fast}, slow={slow}")
        self.fast = fast
        self.slow = slow

    def initialize(self, ctx: Context) -> None:
        self.n = len(ctx.tickers)

    def handle_data(self, ctx: Context) -> None:
        target = 1.0 / self.n
        for i in range(self.n):
            # 取够 slow 根才开始判断; 不足时跳过(不产生信号)
            close = ctx.history(i, "adj_close", self.slow)
            close = close[np.isfinite(close)]
            if len(close) < self.slow:
                continue
            fast = float(close[-self.fast:].mean())
            slow = float(close.mean())
            ctx.order_target_percent(i, target if fast > slow else 0.0)


# 内置代码策略, 供测试 / 命令行 --strategy 用
STRATEGIES: dict[str, type[Strategy]] = {
    "buy_hold": BuyHold,
    "sma_cross": SmaCross,
}


def make_strategy(name: str, **params: object) -> Strategy:
    """按名字建代码策略。

    Args:
        name: 注册名(buy_hold / sma_cross)。
        **params: 透传给策略构造函数的关键字参数。

    Returns:
        策略实例。

    Raises:
        ValueError: name 不在 STRATEGIES 里。
    """
    if name not in STRATEGIES:
        raise ValueError(f"未知策略 {name!r}, 可用的有 {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
