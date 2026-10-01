"""市场规则: A 股与美股。

`MarketRules` 是默认口径(美股): 无涨跌停、T+0、1 股整手、最小价位 0.01。
`CNMarketRules` 是 A 股口径(逻辑沿用 minievent, 已按其测试验证过):

    涨跌停: 北交所 30%; 创业板 300/301、科创板 688/689 20%; 主板 10%,
            主板 ST 按日期查 st_limits(2026-07-06 起 5% -> 10%); ETF 名称含
            "创业板" / "科创" 的 20%, 其余 10%; ETF 的 is_st 一律忽略。
    T+0:   只有 ETF 可能 T+0, 名称含跨境 / 黄金 / 债券 / 货币类关键字的判为 T+0。
    最小价位: ETF 0.001, 其余 0.01; 涨跌停价按最小价位四舍五入(非银行家舍入)。

引擎只调用 limit_pct / is_t0 / tick / st_limit_series, 不关心是哪个市场; 要加新市场
再写一个子类即可(美股直接继承基类)。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np

# 创业板 / 科创板代码前缀, 这些板块涨跌幅 20%
_GROWTH_PREFIXES = ("300", "301", "688", "689")
# 北交所代码后缀, 涨跌幅 30%
_BSE_SUFFIX = ".BJ"


@dataclass(frozen=True)
class MarketRules:
    """默认(美股)规则: 无涨跌停、T+0、1 股整手、tick 0.01。"""

    lot_size: int = 1
    tick_size: float = 0.01

    def limit_pct(self, ticker: str, sec_type: str | None, name: str | None,
                  is_st: bool | None, day: date | None = None) -> float:
        """这只证券当天的涨跌停幅度。美股不设涨跌停, 恒为 0。

        Args:
            ticker / sec_type / name / is_st: 证券标识与状态; 基类用不到, 由 A 股
                子类按板块 / 名称 / ST 判定时使用。
            day: 交易日; 基类用不到(A 股的 ST 幅度随日期变化时才用)。

        Returns:
            幅度(小数), 0.10 表示 10%。
        """
        return 0.0

    def st_follows_schedule(self, ticker: str, sec_type: str | None) -> bool:
        """该证券的 ST 幅度是否随日期变化。

        build_cn_market 用它决定要不要对这类证券逐日查表算 ST 幅度。
        """
        return False

    def st_limit_series(self, days: np.ndarray) -> np.ndarray:
        """向量化的 ST 幅度查询: 一次算出多个交易日的 ST 涨跌幅。

        Args:
            days: datetime64 数组。

        Returns:
            与 days 同形状的幅度数组; 基类恒为 0(A 股子类按生效日分段)。
        """
        return np.zeros(np.asarray(days).shape, dtype=float)

    def is_t0(self, ticker: str, sec_type: str | None, name: str | None) -> bool:
        """是否当天买入当天可卖; 美股 T+0。"""
        return True

    def tick(self, sec_type: str | None) -> float:
        """最小价位(元 / 美元); 基类统一用 tick_size。"""
        return self.tick_size


@dataclass(frozen=True)
class USMarketRules(MarketRules):
    """美股: 与默认口径一致, 单独命名便于在配置里明确区分市场。"""


@dataclass(frozen=True)
class CNMarketRules(MarketRules):
    """A 股规则, 字段含义见模块说明; 所有字段都可以在市场配置里覆盖。"""

    lot_size: int = 100
    tick_size: float = 0.01
    main_limit: float = 0.10
    # 主板 ST 的涨跌幅: (生效日, 幅度), 按日期升序; 早于第一项的日期用第一项。
    # 2026-07-06 起沪深主板 ST 由 5% 调为 10%(已在 minievent 核实)。
    st_limits: tuple[tuple[date, float], ...] = (
        (date(1998, 4, 22), 0.05),
        (date(2026, 7, 6), 0.10),
    )
    growth_limit: float = 0.20
    bse_limit: float = 0.30
    fund_limit: float = 0.10
    # ETF 名称含这些词的按 growth_limit(跟踪创业板 / 科创板指数)
    growth_fund_keywords: tuple[str, ...] = ("创业板", "科创")
    # ETF 名称含这些词的判为 T+0(跨境、黄金、债券、货币类)
    t0_fund_keywords: tuple[str, ...] = (
        "QDII", "黄金", "债", "货币", "恒生", "港股", "H股", "中概", "纳斯达克", "标普", "日经",
    )
    # 关键字判断错了在这里按代码纠正, 覆盖优先于一切其他规则
    limit_overrides: Mapping[str, float] = field(default_factory=dict)
    t0_overrides: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 st_limits 的格式, 写错了在构造时就报, 不拖到算涨跌停价的时候。"""
        days = [d for d, _ in self.st_limits]
        if not days or days != sorted(set(days)):
            raise ValueError(f"st_limits 必须非空、按日期升序且不重复: {self.st_limits}")

    def limit_pct(self, ticker: str, sec_type: str | None, name: str | None,
                  is_st: bool | None, day: date | None = None) -> float:
        """这只证券当天的涨跌停幅度, 优先级见模块说明。

        Args:
            ticker: 标准代码, 如 "600519.SH"。
            sec_type: 证券类型("stock" / "etf" / ...), 来自 stock_basic。
            name: 证券简称, 只用于 ETF 的关键字判断。
            is_st: 当天是否 ST; 对 ETF 忽略(BaoStock 对 ETF 恒给 True)。
            day: 交易日, 只影响主板 ST(见 st_limits); None 时按最新规则。

        Returns:
            幅度(小数), 如 0.10。

        Example:
            >>> from datetime import date
            >>> r = CNMarketRules()
            >>> r.limit_pct("300750.SZ", "stock", "宁德时代", False)
            0.2
            >>> r.limit_pct("600243.SH", "stock", "*ST 某某", True, date(2026, 7, 3))
            0.05
        """
        # 1. 显式覆盖优先于一切
        if ticker in self.limit_overrides:
            return float(self.limit_overrides[ticker])
        # 2. ETF: 按名称关键字判断是不是跟踪创业板 / 科创的 20% 品种
        if sec_type == "etf":
            return self.growth_limit if _has_any(name, self.growth_fund_keywords) \
                else self.fund_limit
        # 3. 北交所 30%
        if ticker.endswith(_BSE_SUFFIX):
            return self.bse_limit
        # 4. 创业板 / 科创板 20%(ST 也跟随板块, 不按 st_limits)
        if ticker.startswith(_GROWTH_PREFIXES):
            return self.growth_limit
        # 5. 主板: ST 按日期查表, 否则 10%
        return self.st_limit_on(day) if is_st is True else self.main_limit

    def st_follows_schedule(self, ticker: str, sec_type: str | None) -> bool:
        """该证券打上 ST 后, 涨跌幅是否按 st_limits 随日期变化。

        只有主板股票是: ETF 忽略 ST、北交所和创业板 / 科创板 ST 跟随板块、
        limit_overrides 里的代码用覆盖值。build_cn_market 用它决定哪些列要逐日查表。
        """
        return (ticker not in self.limit_overrides and sec_type != "etf"
                and not ticker.endswith(_BSE_SUFFIX)
                and not ticker.startswith(_GROWTH_PREFIXES))

    def st_limit_on(self, day: date | None = None) -> float:
        """某一天主板 ST 的涨跌幅。

        Args:
            day: 交易日; None 时取 st_limits 的最后一项(最新规则)。早于第一项的
                日期用第一项。

        Returns:
            幅度(小数)。
        """
        if day is None:
            return self.st_limits[-1][1]
        return float(self.st_limit_series(np.array([day], dtype="datetime64[D]"))[0])

    def st_limit_series(self, days: np.ndarray) -> np.ndarray:
        """向量化的 st_limit_on: 给 build_cn_market 一次查完所有 bar 用。

        Args:
            days: datetime64 数组(任意精度, 按日比较)。

        Returns:
            与 days 同形状的 float 数组。

        Example:
            >>> days = np.array(["2026-07-03", "2026-07-06"], dtype="datetime64[D]")
            >>> CNMarketRules().st_limit_series(days).tolist()
            [0.05, 0.1]
        """
        # searchsorted(side="right") - 1 定位每个日期落在哪个生效区间
        starts = np.array([d for d, _ in self.st_limits], dtype="datetime64[D]")
        pcts = np.array([p for _, p in self.st_limits], dtype=float)
        idx = np.searchsorted(starts, np.asarray(days).astype("datetime64[D]"), side="right") - 1
        # 早于第一项的日期 idx 为 -1, clip 到 0 用第一项
        return pcts[np.clip(idx, 0, None)]

    def is_t0(self, ticker: str, sec_type: str | None, name: str | None) -> bool:
        """是否 T+0(当天买入当天可卖)。

        Args:
            ticker: 标准代码; 在 t0_overrides 里的直接用覆盖值。
            sec_type: 证券类型; 只有 "etf" 才可能 T+0。
            name: 证券简称, 按 t0_fund_keywords 做关键字判断。

        Returns:
            是否 T+0。

        Example:
            >>> CNMarketRules().is_t0("518880.SH", "etf", "华安黄金ETF")
            True
        """
        if ticker in self.t0_overrides:
            return bool(self.t0_overrides[ticker])
        return sec_type == "etf" and _has_any(name, self.t0_fund_keywords)

    def tick(self, sec_type: str | None) -> float:
        """最小价位(元): ETF 0.001, 其余 0.01。"""
        return 0.001 if sec_type == "etf" else self.tick_size


def default_rules(market: str) -> MarketRules:
    """按市场名给出默认规则。

    Args:
        market: "cn" 或 "us"。

    Returns:
        CNMarketRules 或 USMarketRules。

    Raises:
        ValueError: market 不是这两个之一。
    """
    if market == "cn":
        return CNMarketRules()
    if market == "us":
        return USMarketRules()
    raise ValueError(f"未知市场 {market!r}, 只支持 'cn' / 'us'")


def limit_prices(pre_close: np.ndarray, pct: np.ndarray, tick: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """按交易所规则算涨跌停价: 前收 x (1 ± 幅度), 按最小价位四舍五入。

    Args:
        pre_close: 前收盘价(官方口径)。
        pct: 涨跌停幅度(小数)。
        tick: 最小价位。
        三者可以是任意同形状或可广播的数组。

    Returns:
        (涨停价, 跌停价); pre_close 为 NaN 的位置结果也是 NaN。

    Example:
        >>> up, down = limit_prices(np.array([10.05]), np.array([0.10]), np.array([0.01]))
        >>> float(up[0]), float(down[0])
        (11.06, 9.05)
    """
    pre = np.asarray(pre_close, dtype=float)
    return _round_half_up(pre * (1 + pct), tick), _round_half_up(pre * (1 - pct), tick)


def _round_half_up(x: np.ndarray, tick: np.ndarray) -> np.ndarray:
    """按 tick 四舍五入(0.5 进位); np.round 是银行家舍入, 不符合交易所规则。"""
    # 1e-9 吸收浮点误差: 10.05*1.1 可能是 11.055000000000001 或 11.054999999999999, 都应得 11.06
    return np.round(np.floor(x / tick + 0.5 + 1e-9) * tick, 6)


def _has_any(text: str | None, keywords: tuple[str, ...]) -> bool:
    """text 里是否含 keywords 中的任一个; text 为空时为 False。"""
    return bool(text) and any(k in text for k in keywords)
