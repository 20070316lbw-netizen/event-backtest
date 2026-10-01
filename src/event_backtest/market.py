"""给回测引擎的数组接口: 把 liudb 的长表整理成对齐的 (T, N) numpy 数组。

T 是 bar 数(所有证券时间戳的并集, 升序), N 是证券数。引擎只和 MarketData 打交道,
不碰 DuckDB / pandas; 换数据源只要换这里的 loader。

两个市场的口径:
    美股(us): prices 自带 adj_close, 复权因子 = adj_close / close; 无涨跌停、T+0;
        没有独立的前收表, pre_close 由上一交易日收盘推出。
    A 股(cn): 不信任 BaoStock 的 adj_close(对 ETF 无效), 用官方前收自行推算后复权
        因子; 日线 / 分钟线共用 build_cn_market。涨跌停价由 rules + 官方前收算。

统一约定(与 minievent 一致):
    成交量为 0 的 bar 视为"有 bar 但不可成交", OHLC 置 NaN, volume 保留 0;
    tradable = 有行情 & 成交量 > 0 & 未停牌; is_new_day 标记每个交易日的首根 bar,
    供 T+1 在解禁时使用。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import liudb
import liudb.ashare
import liudb.sp500
import numpy as np
import pandas as pd
from loguru import logger

from event_backtest.rules import MarketRules, default_rules, limit_prices

DateLike = str | date | datetime
# 行情长表里会被透视成 (T, N) 的字段
_BAR_FIELDS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class MarketData:
    """对齐后的行情。除注明外都是 (T, N) float64 数组, 缺失为 NaN。

    这是引擎与数据源之间唯一的契约: 引擎不关心数据来自哪个库、哪个市场, 只按下面的
    字段读。价格字段是不复权价(撮合 / 涨跌停用), 算收益和因子请用 adj_close。

    Attributes:
        ts: (T,) datetime64, bar 结束时间; 日线是一天的收盘时刻。
        tickers: (N,) 证券代码, 决定所有矩阵的列顺序。
        open / high / low / close: 不复权 OHLC。
        volume: 成交量; amount: 成交额(没有时全为 NaN)。
        pre_close: 当天的官方前收(日线是一日一个值); 没有时由上一根收盘推出。
        adj_factor: 后复权因子, 同一天每根 bar 相同; adj_close = close * adj_factor。
        limit_up / limit_down: 涨跌停价; 不设涨跌停的市场(美股)为 NaN。
        tradable: (T, N) bool, 这根 bar 能不能成交。
        is_new_day: (T,) bool, 每个交易日的第一根 bar, 供 T+1 解禁用。
        t0: (N,) bool, 是否 T+0 品种。
        sec_type: (N,) 证券类型("stock" / "etf"), 用来选费率。
    """

    ts: np.ndarray            # (T,) datetime64, bar 结束时间(日线为一日的收盘时刻)
    tickers: tuple[str, ...]  # (N,)
    open: np.ndarray          # 以下 OHLC 为不复权价
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    pre_close: np.ndarray     # 当天的官方前收, 日线是一日一个值
    adj_factor: np.ndarray    # 后复权因子, 当天每根 bar 相同
    limit_up: np.ndarray
    limit_down: np.ndarray
    tradable: np.ndarray      # (T, N) bool
    is_new_day: np.ndarray    # (T,) bool, 每个交易日的第一根 bar(T+1 解锁用)
    t0: np.ndarray            # (N,) bool
    sec_type: tuple[str | None, ...]  # (N,) 证券类型("stock" / "etf"), 用于选费率

    @property
    def shape(self) -> tuple[int, int]:
        """(T, N): bar 数, 证券数。"""
        return self.close.shape

    @property
    def adj_close(self) -> np.ndarray:
        """后复权收盘价 close * adj_factor。

        算收益 / 因子用它(价格连续); 撮合与涨跌停用不复权的 close / 官方前收。
        """
        return self.close * self.adj_factor

    @property
    def vwap(self) -> np.ndarray:
        """每根 bar 的成交均价 amount / volume; 成交量为 0 时为 NaN。"""
        out = np.full(self.shape, np.nan)
        # where 只对 volume > 0 的位置做除法, 其余保持 NaN, 不产生 inf
        np.divide(self.amount, self.volume, out=out, where=self.volume > 0)
        return out


def load_market(
    db_path: str | Path,
    market: str = "us",
    tickers: Sequence[str] | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    freq: str | int | None = None,
    rules: MarketRules | None = None,
) -> MarketData:
    """从 liudb 读出 [start, end] 的行情, 整理成 MarketData。

    Args:
        db_path: 数据库文件路径(美股 sp500.db, A 股 ashare.db)。
        market: "cn" 或 "us"。
        tickers: 证券列表(决定列顺序); 默认库里出现的全部代码。
        start / end: 交易日, 两端都含; 默认不限。
        freq: A 股分钟线周期("5" / "15" / "30" / "60"); None 表示日线。美股只有日线。
        rules: 交易规则, 默认按 market 取。

    Returns:
        MarketData, 列顺序同 tickers(或默认顺序)。

    Raises:
        FileNotFoundError: 数据库文件不存在(先报清楚, 不然会退化成一句"没有行情")。
        ValueError: market 未知, 或美股传了 freq。
    """
    rules = rules or default_rules(market)
    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(
            f"数据库不存在: {db_file} (market={market}); 检查 setting/<市场>.yaml 的 "
            f"data.db_path, 或用 --db 指定, 并先用 liudb 写入行情")
    path = str(db_file)
    if market == "cn":
        return _load_cn(path, tickers, start, end, freq, rules)
    if market == "us":
        if freq is not None:
            raise ValueError("美股数据当前只有日线, freq 只能为 None")
        return _load_us(path, tickers, start, end, rules)
    raise ValueError(f"未知市场 {market!r}, 只支持 'cn' / 'us'")


# ---------------------------------------------------------------- A 股

def _load_cn(
    path: str,
    tickers: Sequence[str] | None,
    start: DateLike | None,
    end: DateLike | None,
    freq: str | int | None,
    rules: MarketRules,
) -> MarketData:
    """A 股 loader: 先探一下区间内有没有数据, 再按日线 / 分钟线组装。

    日线部分(前收 / 复权因子 / 停牌 / ST)总是读到 end 为止的全部历史, 这样复权因子
    的起点不随 start 变化(否则换个回测区间, 同一根 bar 的复权价会变)。

    Args:
        path: ashare.db 路径。
        tickers: 证券列表; None 表示用区间内有行情的全部代码。
        start / end: 交易日, 两端含。
        freq: 分钟线周期; None 表示日线。
        rules: 交易规则。

    Returns:
        MarketData。

    Raises:
        ValueError: 区间内没有任何可用行情。
    """
    # 先只查一小段用于确定证券列表; 日线下面会另查全历史
    if freq is None:
        probe = liudb.ashare.load_prices(tickers, start, end, path=path)
    else:
        probe = liudb.ashare.load_intraday_bars(tickers, start, end, freq=freq, path=path)
    if probe.empty:
        logger.warning(f"{path}: {start} ~ {end} 没有行情数据")
    selected = list(tickers) if tickers is not None else sorted(probe["ticker"].unique())

    # 日线读到 end 为止(不限 start), 保证复权因子的起点稳定
    prices = liudb.ashare.load_prices(selected, None, end, path=path)
    status = liudb.ashare.load_daily_status(selected, None, end, path=path)
    daily = _cn_daily(prices, status)
    basic = liudb.ashare.load_stock_basic(selected, path=path)

    if freq is None:
        # 日线时把行情本身当作 bar; 成交额在 status 里, 合并进来
        bars = probe.merge(
            status[["date", "ticker", "amount"]], on=["date", "ticker"], how="left",
        ).rename(columns={"date": "ts"})
    else:
        bars = probe
    if bars.empty:
        raise ValueError(f"{path}: 没有可用的 A 股行情, 检查 tickers / 日期范围 / freq")
    return build_cn_market(bars, daily, basic, tickers=selected, rules=rules)


def _cn_daily(prices: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """日线行情 + 交易状态 -> build_cn_market 需要的日频表。

    Args:
        prices: prices 表, 至少 [date, ticker, close]。
        status: daily_status 表, 至少 [date, ticker, pre_close, is_suspended, is_st]。

    Returns:
        日频长表, 列 [date, ticker, close, pre_close, is_suspended, is_st]。
    """
    return prices[["date", "ticker", "close"]].merge(
        status[["date", "ticker", "pre_close", "is_suspended", "is_st"]],
        on=["date", "ticker"], how="left",
    )


def build_cn_market(
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    *,
    tickers: Sequence[str] | None = None,
    rules: MarketRules | None = None,
) -> MarketData:
    """A 股长表 -> MarketData。纯函数, 不访问数据库, 便于测试。

    处理三件 A 股特有的事: 用官方前收推算复权因子、按 rules 算涨跌停价、把零成交量的
    bar 标成不可交易。

    Args:
        bars: 行情长表, 列 [ts, ticker, open, high, low, close, volume, amount]。
        daily: 日频长表, 列 [date, ticker, close, pre_close, is_suspended, is_st];
            close 是官方收盘价。复权因子从这部分的第一天起算, 所以 daily 可以(也建议)
            比 bars 覆盖更长的历史。
        basic: 证券资料, 列 [ticker, name, sec_type]。
        tickers: 列顺序; 默认取 bars 里出现的全部代码并排序。
        rules: 交易规则, 默认 CNMarketRules()。

    Returns:
        MarketData, 列顺序同 cols。
    """
    rules = rules or default_rules("cn")
    cols = tuple(dict.fromkeys(tickers)) if tickers is not None \
        else tuple(sorted(bars["ticker"].unique()))
    bars = bars[bars["ticker"].isin(cols)].copy()
    # 统一成 datetime, 否则字符串时间戳和下面的 DatetimeIndex 对不齐(整列变 NaN)
    bars["ts"] = pd.to_datetime(bars["ts"])
    if missing := [t for t in cols if t not in set(bars["ticker"])]:
        logger.warning(f"这些证券没有行情数据, 整列为 NaN: {missing}")

    # 1. 时间轴 = 所有证券 ts 的并集; days 用来把日频字段广播到每根 bar
    ts = pd.DatetimeIndex(sorted(bars["ts"].unique()), name="ts")
    fields = _pivot_fields(bars, "ts", ts, cols, _BAR_FIELDS)
    days = ts.normalize()
    is_new_day = np.ones(len(ts), dtype=bool)
    is_new_day[1:] = days[1:] != days[:-1]

    # 2. 日频字段(前收 / 复权因子 / 停牌 / ST)按天对齐到每根 bar
    per_day = {k: v.reindex(days) for k, v in _daily_matrices(daily, cols).items()}
    pre_close = per_day["pre_close"].to_numpy(dtype=float)
    # 当天日线还没入库时(比如盘中抓的数据)沿用上一日因子, 起点兜底为 1
    adj_factor = per_day["adj_factor"].ffill().fillna(1.0).to_numpy(dtype=float)
    suspended = per_day["is_suspended"].fillna(False).to_numpy(dtype=bool)
    is_st = per_day["is_st"].fillna(False).to_numpy(dtype=bool)

    # 3. 每只证券的涨跌停幅度: 先按"是 ST"和"不是 ST"各算一列, 主板 ST 再逐日查表
    info = basic.drop_duplicates("ticker").set_index("ticker").reindex(list(cols))
    sec_type = [_str_or_none(v) for v in info["sec_type"]]
    name = [_str_or_none(v) for v in info["name"]]
    pct_st = np.array([rules.limit_pct(t, s, n, True) for t, s, n in zip(cols, sec_type, name,
                                                                        strict=True)])
    dated = np.array([rules.st_follows_schedule(t, s) for t, s in zip(cols, sec_type,
                                                                     strict=True)])
    pct_st = np.where(dated, rules.st_limit_series(days.to_numpy())[:, None], pct_st)
    pct_ok = np.array([rules.limit_pct(t, s, n, False) for t, s, n in zip(cols, sec_type, name,
                                                                         strict=True)])
    tick = np.array([rules.tick(s) for s in sec_type])
    limit_up, limit_down = limit_prices(pre_close, np.where(is_st, pct_st, pct_ok), tick)

    # 4. 没有成交就没有价格: BaoStock 给的 0 不是真实价, 留着会让因子除出 inf
    volume = fields["volume"]
    no_trade = np.nan_to_num(volume) <= 0
    for field in ("open", "high", "low", "close"):
        fields[field] = np.where(no_trade, np.nan, fields[field])
    close = fields["close"]
    # 可交易 = 有收盘价 & 有成交量 & 当天没停牌(涨跌停封死由撮合层再判)
    tradable = ~np.isnan(close) & ~no_trade & ~suspended

    return MarketData(
        ts=ts.to_numpy(), tickers=cols,
        open=fields["open"], high=fields["high"], low=fields["low"], close=close,
        volume=volume, amount=fields["amount"],
        pre_close=pre_close, adj_factor=adj_factor,
        limit_up=limit_up, limit_down=limit_down,
        tradable=tradable, is_new_day=is_new_day,
        t0=np.array([rules.is_t0(t, s, n) for t, s, n in zip(cols, sec_type, name,
                                                             strict=True)], dtype=bool),
        sec_type=tuple(sec_type),
    )


def _daily_matrices(daily: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """日频字段 -> 以日期为索引、以证券为列的矩阵, 并推算后复权因子。

    复权因子: 每日因子 = 上一日官方收盘 / 当日前收, 逐日连乘, 每只证券第一天为 1。
    价格因此连续: pre_close[t] * adj_factor[t] == close[t-1] * adj_factor[t-1]。
    前收缺失或为 0 的那天因子不变(比值记 1)。

    Args:
        daily: 日频长表, 列 [date, ticker, close, pre_close, is_suspended, is_st]。
        cols: 证券顺序(矩阵的列)。

    Returns:
        {"pre_close" / "adj_factor" / "is_suspended" / "is_st": 日期 x 证券 的 DataFrame};
        没有日线的证券整列为 NaN。
    """
    d = daily[daily["ticker"].isin(cols)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["ticker", "date"])
    prev_close = d.groupby("ticker")["close"].shift(1)
    ratio = (prev_close / d["pre_close"]).where(prev_close.notna() & (d["pre_close"] > 0), 1.0)
    d["adj_factor"] = ratio.groupby(d["ticker"]).cumprod()
    d["is_suspended"] = d["is_suspended"].astype("boolean").fillna(False).astype(bool)
    d["is_st"] = d["is_st"].astype("boolean").fillna(False).astype(bool)

    out = {}
    for field in ("pre_close", "adj_factor", "is_suspended", "is_st"):
        out[field] = d.pivot(index="date", columns="ticker", values=field).reindex(
            columns=list(cols))
    return out


# ---------------------------------------------------------------- 美股

def _load_us(path: str, tickers: Sequence[str] | None, start: DateLike | None,
             end: DateLike | None, rules: MarketRules) -> MarketData:
    """美股 loader: 直接从 sp500.db 的 prices 读区间, 交给 build_us_market。"""
    prices = liudb.sp500.load_prices(tickers, start, end, path=path)
    if prices.empty:
        raise ValueError(f"{path}: {start} ~ {end} 没有行情数据, 检查 tickers / 日期范围")
    selected = list(tickers) if tickers is not None else sorted(prices["ticker"].unique())
    return build_us_market(prices, tickers=selected, rules=rules)


def build_us_market(prices: pd.DataFrame, *, tickers: Sequence[str] | None = None,
                    rules: MarketRules | None = None) -> MarketData:
    """美股日线长表 -> MarketData。纯函数, 不访问数据库。

    复权因子直接取 adj_close / close(yfinance 已算好); 无涨跌停、T+0、1 股整手;
    前收由上一交易日收盘推出(prices 表没有独立的前收字段)。

    Args:
        prices: 日线长表, 列 [date, ticker, open, high, low, close, adj_close, volume]。
        tickers: 列顺序; 默认取 prices 里出现的全部代码并排序。
        rules: 交易规则, 默认 USMarketRules()。

    Returns:
        MarketData; limit_up / limit_down 全为 NaN, amount 全为 NaN, t0 全为 True。
    """
    rules = rules or default_rules("us")
    cols = tuple(dict.fromkeys(tickers)) if tickers is not None \
        else tuple(sorted(prices["ticker"].unique()))
    p = prices[prices["ticker"].isin(cols)].copy()
    if missing := [t for t in cols if t not in set(p["ticker"])]:
        logger.warning(f"这些证券没有行情数据, 整列为 NaN: {missing}")
    p["date"] = pd.to_datetime(p["date"])

    ts = pd.DatetimeIndex(sorted(p["date"].unique()), name="ts")
    fields = _pivot_fields(p, "date", ts, cols, ("open", "high", "low", "close", "volume"))
    adj = _pivot_fields(p, "date", ts, cols, ("adj_close",))["adj_close"]

    # 复权因子 = adj_close / close; close 缺失或为 0 时兜底为 1(该 bar 本来也不可交易)
    close = fields["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = adj / close
    factor = np.where(np.isfinite(factor) & (np.asarray(factor) > 0), factor, 1.0)

    # 前收 = 上一交易日的收盘(未复权); 第一根为 NaN
    pre_close = np.vstack([np.full((1, len(cols)), np.nan), close[:-1]])
    volume = fields["volume"]
    tradable = ~np.isnan(close) & (np.nan_to_num(volume) > 0)
    nan = np.full(close.shape, np.nan)

    return MarketData(
        ts=ts.to_numpy(), tickers=cols,
        open=fields["open"], high=fields["high"], low=fields["low"], close=close,
        volume=volume, amount=np.full(close.shape, np.nan),
        pre_close=pre_close, adj_factor=factor,
        limit_up=nan.copy(), limit_down=nan.copy(),
        tradable=tradable, is_new_day=np.ones(len(ts), dtype=bool),
        t0=np.ones(len(cols), dtype=bool),
        sec_type=tuple("stock" for _ in cols),
    )


# ---------------------------------------------------------------- helpers

def _pivot_fields(df: pd.DataFrame, index: str, ts: pd.DatetimeIndex,
                  cols: tuple[str, ...], fields: Sequence[str]) -> dict[str, np.ndarray]:
    """把长表按 [index, ticker] 透视成对齐的 (T, N) float 数组。

    Args:
        df: 长表。
        index: 时间列名(如 "ts" / "date")。
        ts: 目标时间轴; 透视结果会 reindex 到它, 缺的时点补 NaN。
        cols: 目标证券顺序。
        fields: 要透视的列名。

    Returns:
        {列名: (T, N) float 数组}, 行列顺序分别是 ts / cols。
    """
    return {
        field: df.pivot(index=index, columns="ticker", values=field)
        .reindex(index=ts, columns=list(cols)).to_numpy(dtype=float)
        for field in fields
    }


def _str_or_none(value: object) -> str | None:
    """把 stock_basic 里的字段转成 str; 缺失(None / NaN)转成 None。"""
    return None if value is None or (isinstance(value, float) and np.isnan(value)) \
        else str(value)


def slice_dates(market: MarketData, start: DateLike | None = None,
                end: DateLike | None = None) -> MarketData:
    """按日期切一段 [start, end](两端都含), 复制出一份新行情。

    这是"切片器": 拿到一份(比如 10 年的)行情后, 只保留设计好的时间范围内的 bar,
    用来单独测某个区间(例如疫情那段), 或者喂给 walk-forward。原始 market 不会被改动。

    Args:
        market: 原始行情。
        start / end: 日期(含); None 表示不限这一端。

    Returns:
        新的 MarketData, 数组是拷贝(不与原行情共享内存)。

    Raises:
        ValueError: 范围内一根 bar 都没有。
    """
    dates = pd.DatetimeIndex(market.ts).normalize()
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    inside = np.flatnonzero(mask)
    if len(inside) == 0:
        raise ValueError(f"{start} ~ {end} 范围内没有行情 bar")
    # slice_market 取的是视图, 这里再 copy 一份, 满足"复制原始数据"的语义
    part = slice_market(market, int(inside[0]), int(inside[-1]) + 1)
    return replace(
        part,
        ts=part.ts.copy(), open=part.open.copy(), high=part.high.copy(),
        low=part.low.copy(), close=part.close.copy(), volume=part.volume.copy(),
        amount=part.amount.copy(), pre_close=part.pre_close.copy(),
        adj_factor=part.adj_factor.copy(), limit_up=part.limit_up.copy(),
        limit_down=part.limit_down.copy(), tradable=part.tradable.copy(),
        is_new_day=part.is_new_day.copy(),
    )


def slice_market(market: MarketData, start: int, end: int) -> MarketData:
    """按 bar 下标切一段 [start, end)。

    walk-forward 用它在训练 / 测试段之间切行情。tickers / sec_type / t0 这类
    "按证券"的字段原样保留, 其余按 bar 切的字段都跟着切。

    Args:
        market: 原行情。
        start: 起始 bar 下标(含)。
        end: 结束 bar 下标(不含)。

    Returns:
        新的 MarketData。
    """
    return replace(
        market,
        ts=market.ts[start:end],
        open=market.open[start:end], high=market.high[start:end],
        low=market.low[start:end], close=market.close[start:end],
        volume=market.volume[start:end], amount=market.amount[start:end],
        pre_close=market.pre_close[start:end], adj_factor=market.adj_factor[start:end],
        limit_up=market.limit_up[start:end], limit_down=market.limit_down[start:end],
        tradable=market.tradable[start:end], is_new_day=market.is_new_day[start:end],
    )
