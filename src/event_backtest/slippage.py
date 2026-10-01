"""滑点模型: 固定价差与成交量比例(借 zipline 的 FixedSlippage / VolumeShareSlippage)。

模型只回答两个问题: 这笔成交价是多少、本 bar 最多能成交多少股。撮合层调用 apply,
成交价与数量都在 broker 里落账。成交量比例模型按 zipline 的公式:
    买入价 = price * (1 + price_impact * share^2), 卖出价同式取负,
    其中 share = min(成交量 / bar 成交量, volume_limit), 单 bar 最多成交
    volume_limit x bar 成交量。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SlippageModel(ABC):
    """滑点接口: 把"理论成交价 + 想成交的数量"换成"实际成交价 + 实际可成交数量"。"""

    @abstractmethod
    def apply(self, price: float, *, is_buy: bool, bar_volume: float,
              qty: int) -> tuple[float, int]:
        """算一笔成交的最终价与量。

        Args:
            price: 撮合层算出的理论成交价(不含滑点)。
            is_buy: 买入还是卖出; 滑点对买方向不利、卖方向不利(一增一减)。
            bar_volume: 当前 bar 的历史成交量, 用于限制成交比例。
            qty: 撮合层本打算成交的数量。

        Returns:
            (调整后成交价, 本 bar 允许的最大成交量); 数量上限小于 qty 时按上限成交,
            相当于部分成交, 剩余部分留到下根 bar。
        """


@dataclass(frozen=True)
class NoSlippage(SlippageModel):
    """不调价、不限量。"""

    def apply(self, price: float, *, is_buy: bool, bar_volume: float,
              qty: int) -> tuple[float, int]:
        return price, qty


@dataclass(frozen=True)
class FixedSlippage(SlippageModel):
    """固定价差: 买入价 + spread/2, 卖出价 - spread/2; 不限量。

    Attributes:
        spread: 买卖价差(绝对价格单位), 比如 0.01 表示一分钱。
    """

    spread: float = 0.0

    def apply(self, price: float, *, is_buy: bool, bar_volume: float,
              qty: int) -> tuple[float, int]:
        # 一半价差加到买入价上, 一半从卖出价里扣, 价格总是往不利方向偏
        half = self.spread / 2.0
        return (price + half if is_buy else price - half), qty


@dataclass(frozen=True)
class VolumeShareSlippage(SlippageModel):
    """成交量比例: 每 bar 最多成交 volume_limit x bar 成交量, 冲击按占比平方。

    Attributes:
        volume_limit: 单 bar 最多吃掉历史成交量的比例(0, 1]。
        price_impact: 冲击系数; 占比越高成交价越差, 按 share^2 放大。
    """

    volume_limit: float = 0.025
    price_impact: float = 0.1

    def apply(self, price: float, *, is_buy: bool, bar_volume: float,
              qty: int) -> tuple[float, int]:
        # 没有历史成交量就买不到 / 卖不掉
        if bar_volume <= 0 or qty <= 0:
            return price, 0
        cap = int(self.volume_limit * bar_volume)
        if cap < 1:
            return price, 0
        fill = min(qty, cap)
        # 实际成交占比(不会超过 volume_limit), 用它算冲击
        share = min(fill / bar_volume, self.volume_limit)
        impact = share ** 2 * self.price_impact * price
        return (price + impact if is_buy else price - impact), fill


# 配置里的 type -> 模型类
_SLIPPAGE = {"none": NoSlippage, "fixed": FixedSlippage, "volume_share": VolumeShareSlippage}


def make_slippage(kind: str, **params: float) -> SlippageModel:
    """按名字建滑点模型(配置解析用)。

    Args:
        kind: "none" / "fixed" / "volume_share"。
        **params: 透传给对应模型, 如 fixed 的 spread、volume_share 的
            volume_limit / price_impact。

    Returns:
        对应的 SlippageModel。

    Raises:
        ValueError: kind 不认识。
    """
    if kind not in _SLIPPAGE:
        raise ValueError(f"未知滑点类型 {kind!r}, 可用的有 {sorted(_SLIPPAGE)}")
    return _SLIPPAGE[kind](**params)
