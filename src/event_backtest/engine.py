"""事件驱动引擎: 主循环、策略上下文与回测入口。

每个 bar 的顺序(与 zipline 一致):
    1. 新交易日先给 T+1 持仓解禁;
    2. 用当前 bar 撮合上一根挂出的订单, 落到组合上;
    3. 用当前 bar 收盘盯市, 记净值;
    4. 调 strategy.handle_data(ctx), 收集新订单留到下一根撮合。

第 1 步在第 2 步之前, 所以昨天(T+1)买入的持仓今天可以卖; 第 3 步在第 4 步之前, 所以
策略看到的是"已经成交后"的净值。这两点保证了撮合和下单都不会用到未来数据。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from event_backtest.broker import (
    CANCELLED,
    FILL_PRICES,
    FILLED,
    ORDER_COLUMNS,
    REJECTED,
    Fill,
    FillPrice,
    Order,
    Portfolio,
    apply_fill,
    match_order,
)
from event_backtest.fees import ZERO_FEE, FeeSchedule
from event_backtest.market import MarketData
from event_backtest.rules import MarketRules
from event_backtest.slippage import SlippageModel


class Context:
    """策略看到的世界: 行情访问 + 下单 API。

    一个 Context 贯穿整场回测, bar 属性指向当前进度; 策略每次 handle_data 拿到的都是
    同一个对象, 所以自定义状态放在策略对象上, 不要放在 Context 上。
    """

    def __init__(self, market: MarketData, portfolio: Portfolio, rules: MarketRules,
                 fees: Mapping[str, FeeSchedule]) -> None:
        self.market = market
        self.portfolio = portfolio
        self.rules = rules
        self.fees = fees
        self.bar = 0
        self._pending: list[Order] = []
        # 盯市价: 每根 bar 更新为最新有效收盘价, 缺价时沿用上一个(见 run 主循环)
        self.mark = np.full(len(market.tickers), np.nan)
        # 代码 -> 列下标, 避免每根 bar 都线性查找
        self._index = {t: i for i, t in enumerate(market.tickers)}

    # ---------------------------------------------------------- 查询
    def _resolve(self, ticker: str | int) -> int:
        """把证券代码或下标统一成列下标。"""
        if isinstance(ticker, int) and not isinstance(ticker, bool):
            return ticker
        return self._index[str(ticker)]

    @property
    def tickers(self) -> tuple[str, ...]:
        """当前回测的全部证券代码(顺序即矩阵列顺序)。"""
        return self.market.tickers

    def price(self, ticker: str | int, field: str = "close") -> float:
        """当前 bar 的某个价格字段; 缺价为 NaN。

        Args:
            ticker: 证券代码或下标。
            field: 字段名, 如 "close" / "adj_close" / "open" / "high" / "low"。

        Returns:
            当前 bar 的标量值。
        """
        return float(getattr(self.market, field)[self.bar, self._resolve(ticker)])

    def history(self, ticker: str | int, field: str = "close", window: int = 1) -> np.ndarray:
        """截至当前 bar(含)的最近 window 个值。

        数据不足 window 时返回已有的那部分(从头开始), 不会补 NaN; 策略自己判断长度。

        Args:
            ticker: 证券代码或下标。
            field: 字段名。
            window: 往回取多少根 bar。

        Returns:
            一维数组, 长度 <= window。
        """
        i = self._resolve(ticker)
        start = max(0, self.bar - window + 1)
        return getattr(self.market, field)[start:self.bar + 1, i]

    def position(self, ticker: str | int) -> float:
        """当前持仓数量(含当天买入未解禁的部分)。"""
        return float(self.portfolio.total[self._resolve(ticker)])

    def nav(self) -> float:
        """当前净值 = 现金 + 持仓市值(用最近的有效收盘价盯市)。"""
        return self.portfolio.nav(self.mark)

    @property
    def cash(self) -> float:
        """可用现金。"""
        return self.portfolio.cash

    # ---------------------------------------------------------- 下单
    def order(self, ticker: str | int, amount: int, *, limit: float | None = None,
              stop: float | None = None, tag: str = "") -> Order | None:
        """下市价 / 限价 / 止损单; amount > 0 买, < 0 卖。

        Args:
            ticker: 证券代码或下标。
            amount: 数量(带方向); 0 会被忽略并返回 None。
            limit: 限价。
            stop: 止损 / 触发价。
            tag: 下单来源(如 "enter" / "exit:stop_loss"); 会原样记进委托明细,
                用来回答"这笔单是哪个信号 / 哪条出场规则下的"。

        Returns:
            新建的 Order(下一根 bar 才撮合); amount 为 0 时返回 None。
        """
        amount = int(amount)
        if amount == 0:
            return None
        order = Order(index=self._resolve(ticker), amount=amount, created=self.bar,
                      limit=limit, stop=stop, tag=str(tag))
        self._pending.append(order)
        return order

    def order_target_percent(self, ticker: str | int, percent: float,
                             field: str = "close", *, tag: str = "") -> Order | None:
        """调到目标净值占比。

        用当前 bar 的 field 价把"目标市值"换算成股数, 再减去现有持仓, 得到要买 / 卖的
        数量; 实际成交数量由 broker 取整手并受现金 / 可卖限制。

        Args:
            ticker: 证券代码或下标。
            percent: 目标市值占净值的比例(1.0 = 满仓该证券, 0 = 清仓)。
            field: 换算用的价格字段。
            tag: 下单来源; 同 order 的 tag。

        Returns:
            新建的 Order; 当前缺价或价格非正时返回 None(不下单)。
        """
        i = self._resolve(ticker)
        px = getattr(self.market, field)[self.bar, i]
        if not np.isfinite(px) or px <= 0:
            return None
        delta = (percent * self.nav() - self.portfolio.total[i] * px) / px
        return self.order(i, int(delta), tag=tag)

    def take_orders(self) -> list[Order]:
        """取走本次 handle_data 产生的全部订单, 并清空暂存。"""
        out, self._pending = self._pending, []
        return out


class ContextStrategy(Protocol):
    """引擎接受的策略协议: 只要有 handle_data(ctx)。

    initialize(ctx) 与 on_fill(fill, ctx) 都是可选的; 直接给一个
    `Callable[[Context], None]` 也可以(见 run 的实现)。
    """

    def handle_data(self, ctx: Context) -> None:
        """每根 bar 调一次。"""
        ...


def run(strategy: ContextStrategy | Callable[[Context], None], market: MarketData, *,
        initial_cash: float,
        fees: Mapping[str, FeeSchedule] | FeeSchedule | None = None,
        fill_price: FillPrice = "open", slippage: SlippageModel | None = None,
        rules: MarketRules | None = None) -> BacktestResult:
    """跑一遍回测。

    Args:
        strategy: 实现了 initialize(ctx) / handle_data(ctx) 的对象; 只要 handle_data
            也行。有 on_fill(fill, ctx) 时每次成交回调一次。
        market: 对齐后的行情。
        initial_cash: 初始现金。
        fees: 按 sec_type 给的费率表; 也可以直接给一个 FeeSchedule 应用到所有证券。
        fill_price: 市价单成交价, "open"(默认) 或 "vwap"。写别的值直接报错, 不再
            静默当成 vwap。
        slippage: 滑点模型, 默认 None(不调价)。
        rules: 交易规则(整手等), 默认美股口径。

    Returns:
        BacktestResult, 含逐 bar 净值、成交明细、委托明细、行情与期末组合。

    Raises:
        ValueError: fill_price 不是 "open" / "vwap"; 或要按 vwap 成交但行情里
            没有成交额(amount)数据(此时所有单都会被拒, 不如直接报错)。
    """
    if fill_price not in FILL_PRICES:
        raise ValueError(f"fill_price 只能是 {list(FILL_PRICES)}, 收到 {fill_price!r}")
    if fill_price == "vwap" and not np.isfinite(market.amount).any():
        raise ValueError(
            "行情没有成交额(amount)数据, 不能用 fill_price='vwap'(每笔都会拒单); "
            "该数据源请改用 fill_price='open'")
    rules = rules or MarketRules()
    n = len(market.tickers)
    # 每只证券对应的费率: 统一成"下标 -> FeeSchedule", 撮合时按 order.index 取
    fee_for = _fee_lookup(fees, market, n)
    lot = int(rules.lot_size)

    portfolio = Portfolio.initial(initial_cash, n)
    ctx = Context(market, portfolio, rules, fees if isinstance(fees, Mapping) else {})
    if hasattr(strategy, "initialize"):
        strategy.initialize(ctx)

    T = len(market.ts)
    nav = np.full(T, np.nan)
    pending: list[Order] = []      # 已挂出、等下一根撮合的订单
    orders: list[Order] = []       # 全部委托(含已成交 / 被拒 / 期末未成交), 供事后复盘
    fills: list[Fill] = []

    for t in range(T):
        ctx.bar = t

        # 0. 更新盯市价: 有收盘价就更新; 缺失(停牌 / 零成交量 bar)沿用上一个有效价,
        #    否则持仓会被按 0 计, 净值出现假的跳水
        valid = np.isfinite(market.close[t])
        ctx.mark[valid] = market.close[t, valid]

        # 1. 新交易日: T+1 持仓解禁
        if market.is_new_day[t]:
            portfolio.unlock()

        # 2. 撮合上一根挂出的订单, 并把成交回报给策略
        for order in list(pending):
            fill = match_order(order, t, market, portfolio, fee=fee_for[order.index],
                               fill_price=fill_price, lot_size=lot, slippage=slippage)
            if fill is not None:
                apply_fill(fill, portfolio, market.t0)
                fills.append(fill)
                hook = getattr(strategy, "on_fill", None)
                if hook is not None:
                    hook(fill, ctx)
            if order.status in (FILLED, REJECTED):
                pending.remove(order)

        # 3. 用最近有效收盘价盯市(在用户下单之前, 记录的是当前真实净值)
        nav[t] = portfolio.nav(ctx.mark)

        # 4. 策略决策: 新订单留到下一根撮合
        handle = getattr(strategy, "handle_data", strategy)
        handle(ctx)
        new_orders = ctx.take_orders()
        orders.extend(new_orders)
        pending.extend(new_orders)

    # 收尾: 最后一根挂出、还没成交的订单作废, 原因写清楚供复盘
    for order in pending:
        order.status = CANCELLED
        order.reason = "回测结束未成交"

    return BacktestResult(
        nav=pd.Series(nav, index=pd.DatetimeIndex(market.ts, name="ts"), name="nav"),
        fills=_fills_frame(fills, market),
        market=market,
        portfolio=portfolio,
        orders=_orders_frame(orders, market),
    )


@dataclass(eq=False)
class BacktestResult:
    """回测输出。

    Attributes:
        nav: 逐 bar 净值序列(index 是 bar 时间)。
        fills: 成交明细 DataFrame; 没成交时是带列名的空表。
        market: 本次回测用的行情(便于事后算事件研究等)。
        portfolio: 期末组合状态。
        orders: 委托明细 DataFrame(全部订单, 含被拒 / 期末未成交 / 部分成交),
            列见 broker.ORDER_COLUMNS; status 是 open / filled / rejected / cancelled,
            reason 是没成交的原因, tag 是下单来源。
        meta: 这次运行的上下文(市场 / 基准 / 库路径 / 初始资金 / 费率口径等);
            engine.run 只保证建一个空 dict, 由上层(如 MarketConfig.run)填。
            有了它, summarize / benchmark_nav 这些就不用调用者再重复传参数。
    """

    nav: pd.Series
    fills: pd.DataFrame
    market: MarketData
    portfolio: Portfolio
    orders: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(ORDER_COLUMNS)))
    meta: dict[str, Any] = field(default_factory=dict)


def _fee_lookup(fees: Mapping[str, FeeSchedule] | FeeSchedule | None, market: MarketData,
                n: int) -> list[FeeSchedule]:
    """把不同写法的 fees 统一成"每个证券下标一个费率"的列表。

    Args:
        fees: None(全零) / 单个 FeeSchedule(所有证券一样) / {sec_type: FeeSchedule}。
        market: 行情(用 sec_type 字段)。
        n: 证券数。

    Returns:
        长度 n 的列表; 某只证券的 sec_type 不在 fees 里时用 ZERO_FEE。
    """
    if fees is None:
        return [ZERO_FEE] * n
    if isinstance(fees, FeeSchedule):
        return [fees] * n
    return [fees.get(t, ZERO_FEE) for t in market.sec_type]


def _orders_frame(orders: list[Order], market: MarketData) -> pd.DataFrame:
    """委托列表 -> DataFrame, 把下标翻译成代码, 保留状态 / 原因 / 来源 tag。

    amount 与 filled 取绝对值, 方向看 side; limit / stop 没设时是 NaN。
    """
    if not orders:
        return pd.DataFrame(columns=list(ORDER_COLUMNS))
    return pd.DataFrame([
        {
            "ts": pd.Timestamp(market.ts[o.created]), "bar": o.created,
            "ticker": market.tickers[o.index],
            "side": "buy" if o.is_buy else "sell",
            "amount": abs(o.amount), "filled": abs(o.filled),
            "status": o.status, "reason": o.reason, "tag": o.tag,
            "limit": o.limit if o.limit is not None else np.nan,
            "stop": o.stop if o.stop is not None else np.nan,
        }
        for o in orders
    ], columns=list(ORDER_COLUMNS))


def _fills_frame(fills: list[Fill], market: MarketData) -> pd.DataFrame:
    """成交列表 -> DataFrame, 把下标翻译成代码, 方向翻译成 buy / sell。"""
    columns = ["ts", "bar", "ticker", "side", "amount", "price", "fee", "order_created"]
    if not fills:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([
        {
            "ts": pd.Timestamp(f.ts), "bar": f.bar, "ticker": market.tickers[f.index],
            "side": "buy" if f.amount > 0 else "sell", "amount": abs(f.amount),
            "price": f.price, "fee": f.fee, "order_created": f.order_created,
        }
        for f in fills
    ], columns=columns)
