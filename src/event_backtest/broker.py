"""交易核心: 订单、组合状态与撮合。

撮合遵循 zipline 的顺序: 每个 bar 先用**当前 bar**撮合**上一根**挂出的订单, 再让
策略下新单, 所以天然没有未来函数。A 股的约束都落在这一层:
    - tradable 为 False(无行情 / 零成交量 / 停牌)不成交;
    - 一字涨停买不进(low >= limit_up), 一字跌停卖不出(high <= limit_down);
    - 买入按整手向下取整; 卖出受 T+1 可卖数量限制, 非清仓也按整手;
    - 现金不足按可负担手数成交。

撮合价: 市价单用 fill_price 指定的开盘价或 VWAP; 限价单在价格触及时成交; 止损单在
价格突破时成交。都是对当前 bar 的 OHLC 做判断, 不使用未来数据。滑点模型只在最后调整
一次成交价与可成交量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from event_backtest.fees import FeeSchedule
from event_backtest.market import MarketData
from event_backtest.slippage import SlippageModel

OPEN = "open"
FILLED = "filled"
REJECTED = "rejected"
CANCELLED = "cancelled"

FillPrice = Literal["open", "vwap"]
# 允许的市价单成交价口径; engine.run 用它校验, 写错的值不会再被静默当成 vwap
FILL_PRICES: tuple[str, ...] = ("open", "vwap")

# 委托明细(BacktestResult.orders)的列, 顺序即列顺序
ORDER_COLUMNS = ("ts", "bar", "ticker", "side", "amount", "filled", "status", "reason",
                 "tag", "limit", "stop")


@dataclass
class Order:
    """一张订单。amount > 0 买, < 0 卖; 卖单数量是负数。

    Attributes:
        index: 证券在 MarketData.tickers 里的下标。
        amount: 委托数量(带方向, 正买负卖)。
        created: 下单时的 bar 下标; 用于回报里回溯"第几根挂的"。
        limit: 限价, None 表示市价。
        stop: 止损 / 触发价, None 表示不设。
        tag: 下单来源(如 "enter" / "exit:stop_loss"), 事后归因用。
        filled: 已成交数量(带方向)。
        status: open / filled / rejected / cancelled。
        reason: 被拒或状态变化的原因, 便于排查"为什么没成交"。
    """

    index: int
    amount: int
    created: int
    limit: float | None = None
    stop: float | None = None
    tag: str = ""
    filled: int = 0
    status: str = OPEN
    reason: str = ""

    @property
    def open_amount(self) -> int:
        """还没成交的数量(带方向): amount - filled。"""
        return self.amount - self.filled

    @property
    def is_buy(self) -> bool:
        """是否买单。"""
        return self.amount > 0


@dataclass
class Fill:
    """一笔成交。amount > 0 买, < 0 卖。

    Attributes:
        bar: 成交所在 bar 下标。
        ts: 成交时间(取 market.ts[bar])。
        index: 证券下标。
        amount: 成交数量(带方向)。
        price: 成交价(已含滑点, 未复权)。
        fee: 这笔成交的费用。
        order_created: 下单时的 bar 下标, 可算"挂了几根才成交"。
    """

    bar: int
    ts: np.datetime64
    index: int
    amount: int
    price: float
    fee: float
    order_created: int


@dataclass
class Portfolio:
    """组合状态: 现金、持仓数量、可卖数量、持仓成本均价。

    Attributes:
        cash: 可用现金。
        total: (N,) 持仓总数(含当天买入未解禁的)。
        available: (N,) 可卖数量; T+1 品种当天买入的部分不计入, 次日 unlock 时补上。
        cost: (N,) 持仓成本均价(含买入费用)。
    """

    cash: float
    total: np.ndarray
    available: np.ndarray
    cost: np.ndarray

    @classmethod
    def initial(cls, cash: float, n: int) -> Portfolio:
        """建一个空仓组合。

        Args:
            cash: 初始现金。
            n: 证券数。

        Returns:
            全零持仓、available / cost 都是长度 n 的数组。
        """
        return cls(cash=float(cash), total=np.zeros(n), available=np.zeros(n), cost=np.zeros(n))

    def market_value(self, close: np.ndarray) -> float:
        """持仓市值合计; 缺价按 0 计(停牌 / 无行情的证券不参与盯市)。"""
        return float(np.dot(self.total, np.nan_to_num(close)))

    def nav(self, close: np.ndarray) -> float:
        """净值 = 现金 + 持仓市值。"""
        return self.cash + self.market_value(close)

    def unlock(self) -> None:
        """新交易日开始: 把 T+1 买入的持仓解禁为可卖。"""
        # 每个交易日开盘前调用一次; 简化处理为"可卖 = 当前持仓"
        self.available = self.total.copy()


def apply_fill(fill: Fill, portfolio: Portfolio, t0: np.ndarray) -> None:
    """把一笔成交落到组合上: 更新现金、持仓、成本与可卖数量。

    买入把费用摊进成本价, 卖出不改变成本价(只减数量)。

    Args:
        fill: 成交。
        portfolio: 就地修改的组合。
        t0: (N,) bool, 哪些是 T+0 品种(买入即可卖)。
    """
    i = fill.index
    if fill.amount > 0:  # 买入
        portfolio.cash -= fill.amount * fill.price + fill.fee
        new_total = portfolio.total[i] + fill.amount
        # 加权平均成本: (原持仓成本 + 本次买入含费金额) / 新持仓数量
        prev_value = portfolio.cost[i] * portfolio.total[i]
        portfolio.cost[i] = (prev_value + fill.amount * fill.price + fill.fee) / new_total
        portfolio.total[i] = new_total
        if t0[i]:  # T+0 品种买入即可卖, 直接进可卖
            portfolio.available[i] += fill.amount
    else:  # 卖出
        qty = -fill.amount
        portfolio.cash += qty * fill.price - fill.fee
        portfolio.total[i] -= qty
        portfolio.available[i] -= qty


def match_order(order: Order, bar: int, market: MarketData, portfolio: Portfolio, *,
                fee: FeeSchedule, fill_price: FillPrice = "open", lot_size: int = 1,
                slippage: SlippageModel | None = None) -> Fill | None:
    """尝试在 bar 撮合一笔订单。

    处理顺序(任一不通过就不成交):
        1. 不可交易(无行情 / 停牌 / 零成交量) -> 拒绝;
        2. 按市价 / 限价 / 止损算出成交价, 价格没触及 -> 继续挂着;
        3. 一字涨停买不进 / 一字跌停卖不出 -> 拒绝;
        4. 数量: 买入按可买数量, 卖出受 T+1 可卖数量限制, 按整手;
        5. 滑点: 调整成交价并限制本 bar 成交量;
        6. 买入再看现金够不够;
        7. 落账: 更新 order.filled / status, 返回 Fill。

    Args:
        order: 要撮合的订单(就地更新)。
        bar: 当前 bar 下标。
        market: 行情。
        portfolio: 组合(用来查可卖数量与现金)。
        fee: 这笔订单对应的费率。
        fill_price: 市价单成交价来源, "open" 或 "vwap"。
        lot_size: 整手大小(A 股 100, 美股 1)。
        slippage: 滑点模型; None 表示不调价不限量。

    Returns:
        成交返回 Fill; 价格未触及返回 None(订单继续挂着); 被拒时把 order 标成
        REJECTED 并返回 None。
    """
    i = order.index
    # 1. 这根 bar 能不能交易
    if not market.tradable[bar, i]:
        order.status, order.reason = REJECTED, "不可交易"
        return None

    o, h, low = market.open[bar, i], market.high[bar, i], market.low[bar, i]
    base = o if fill_price == "open" else market.vwap[bar, i]
    if not np.isfinite(base):
        order.status, order.reason = REJECTED, "无成交价"
        return None

    # 2. 限价 / 止损触发判断; None 表示价格没到, 订单留着下根再试
    price = _trigger_price(order, base, o, h, low)
    if price is None:
        return None

    # 3. 涨跌停封死: 整根 bar 都在涨停价(跌停价)上, 买(卖)不到
    buy = order.is_buy
    limit_up, limit_down = market.limit_up[bar, i], market.limit_down[bar, i]
    if buy and np.isfinite(limit_up) and low >= limit_up:
        order.status, order.reason = REJECTED, "涨停买不进"
        return None
    if not buy and np.isfinite(limit_down) and h <= limit_down:
        order.status, order.reason = REJECTED, "跌停卖不出"
        return None

    # 4. 数量: 买入想买多少, 卖出最多能卖多少(受 T+1 限制)
    want = (order.open_amount if buy
            else min(-order.open_amount, int(portfolio.available[i])))
    # 清仓时允许零股, 其余按整手向下取整
    allow_odd = not buy and want == int(portfolio.available[i])
    qty = _round_qty(want, lot_size, allow_odd=allow_odd)
    if qty <= 0:
        order.status, order.reason = REJECTED, "无可用持仓" if not buy else "数量不足一手"
        return None

    # 5. 滑点: 调整成交价, 并把数量压到本 bar 允许的成交量以内
    if slippage is not None:
        price, cap = slippage.apply(price, is_buy=buy,
                                    bar_volume=float(market.volume[bar, i]), qty=qty)
        # 成交量上限是任意整数(如 37), 对整手品种要再截一次, 否则 A 股会成交零股
        qty = _round_qty(min(qty, cap), lot_size, allow_odd=allow_odd)
        if qty <= 0:
            order.status, order.reason = REJECTED, "成交量不足"
            return None
        # 滑点后的价格若比限价更差, 不成交(订单继续挂着, 不视为拒绝)
        if order.limit is not None and ((buy and price > order.limit)
                                        or (not buy and price < order.limit)):
            return None

    # 6. 买入再看现金; 手续费也要算进去, 所以用 _affordable 而不是直接除价格
    if buy:
        affordable = _affordable(portfolio.cash, price, fee, lot_size)
        qty = min(qty, affordable)
        if qty <= 0:
            order.status, order.reason = REJECTED, "现金不足"
            return None

    # 7. 落账
    amount = qty if buy else -qty
    fill_fee = float(fee.fee(qty * price, sell=not buy))
    order.filled += amount
    if order.filled == order.amount:
        order.status = FILLED
    return Fill(bar=bar, ts=market.ts[bar], index=i, amount=amount, price=float(price),
                fee=fill_fee, order_created=order.created)


def _trigger_price(order: Order, base: float, o: float, h: float,
                   low: float) -> float | None:
    """按限价 / 止损条件算成交价; 条件没满足返回 None。

    限价单: 开盘价已经优于限价就按开盘价成交, 否则价格触及限价就按限价成交;
    止损单: 价格突破止损价后, 按更差的一侧成交(买 max(价, 止损), 卖 min(价, 止损))。

    Args:
        order: 订单(提供 limit / stop 与买卖方向)。
        base: 未触发前的基准价(开盘价或 VWAP)。
        o / h / low: 当前 bar 的开盘 / 最高 / 最低价。

    Returns:
        成交价; 未触发返回 None。
    """
    price = base
    if order.limit is not None:
        if order.is_buy:
            if base <= order.limit:          # 开盘就比限价便宜, 直接成交
                price = base
            elif low <= order.limit:         # 盘中触及限价
                price = order.limit
            else:
                return None
        else:
            if base >= order.limit:          # 开盘就比限价高, 直接成交
                price = base
            elif h >= order.limit:
                price = order.limit
            else:
                return None
    if order.stop is not None:
        if order.is_buy:
            if h < order.stop:               # 没涨到止损价
                return None
            price = max(price, order.stop)
        else:
            if low > order.stop:             # 没跌到止损价
                return None
            price = min(price, order.stop)
    return price


def _round_qty(want: int, lot_size: int, *, allow_odd: bool) -> int:
    """按整手取整。

    Args:
        want: 期望数量(正数)。
        lot_size: 一手多少股。
        allow_odd: True 时允许零股(用于清仓, 把不足一手的尾仓一次卖完)。

    Returns:
        取整后的数量; want <= 0 返回 0。
    """
    if want <= 0:
        return 0
    if lot_size <= 1 or allow_odd:
        return want
    return (want // lot_size) * lot_size


def _affordable(cash: float, price: float, fee: FeeSchedule, lot_size: int) -> int:
    """用现有现金最多能买多少股(含手续费), 按整手。

    先按"不含费"估算一个上界, 再逐手往下退, 直到含费金额不超过现金 —— 因为最低佣金
    的存在, 含费可买量可能比 cash / price 少几手, 不能用一次除法算准。

    Args:
        cash: 可用现金。
        price: 成交价。
        fee: 费率。
        lot_size: 一手多少股。

    Returns:
        可买数量(0 表示买不起一手)。
    """
    lot = max(lot_size, 1)
    if price <= 0 or cash <= 0:
        return 0
    qty = int(cash // (price * lot)) * lot
    while qty > 0 and qty * price + float(fee.fee(qty * price, sell=False)) > cash:
        qty -= lot
    return max(qty, 0)
