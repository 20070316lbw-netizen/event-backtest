"""费用模型: 按成交金额比例计费, A 股与美股共用一套结构。

A 股: 佣金(双向, 有最低值) + 印花税(仅卖出) + 过户费(双向)。
美股: 通常只填 commission, 其余留 0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeeSchedule:
    """一类证券的费率, 都是成交金额的比例(min_commission 是货币单位)。

    Attributes:
        commission: 佣金率, 买卖双向都收, 按成交金额计。
        min_commission: 每笔最低佣金; 成交金额不为 0 时佣金至少是这个数。
        stamp_duty: 印花税率, 仅卖出收(A 股股票 0.05%, ETF 为 0)。
        transfer_fee: 过户费率, 买卖双向都收(A 股股票 0.001%, ETF 为 0)。

    Example:
        >>> etf = FeeSchedule(commission=0.00025, min_commission=5)
        >>> etf.fee(1000, sell=False), etf.fee(100_000, sell=True)
        (5.0, 25.0)
    """

    commission: float = 0.0
    min_commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0

    def fee(self, amount: float | np.ndarray, *, sell: bool) -> float | np.ndarray:
        """成交金额 -> 这笔成交的总费用。

        计算式::

            max(成交金额 x commission, min_commission)
            + 成交金额 x transfer_fee
            + (卖出 ? 成交金额 x stamp_duty : 0)

        Args:
            amount: 成交金额(货币单位), 标量或数组; 内部取绝对值, 传正负都可以。
            sell: 是否卖出 —— 决定收不收印花税。

        Returns:
            与 amount 同形状的费用; 标量输入返回 float。amount 为 0 的位置费用
            为 0(连最低佣金也不收), 用来表示"这笔没成交"。
        """
        # 取绝对值: 卖出时调用方可能传负数金额, 费率都按成交金额的比例算
        amt = np.abs(np.asarray(amount, dtype=float))
        # 佣金有每笔下限; 过户费双向; 印花税只有卖出才收
        commission = np.maximum(amt * self.commission, self.min_commission)
        total = commission + amt * self.transfer_fee + (amt * self.stamp_duty if sell else 0.0)
        # 没成交(金额为 0)时连最低佣金都不收
        out = np.where(amt > 0, total, 0.0)
        return float(out) if out.ndim == 0 else out


# 零费率, 不指定费用时的默认值
ZERO_FEE = FeeSchedule()
