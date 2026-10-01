"""市场配置: 默认费率、行情 + 规则 + 费率的打包, 以及 YAML 读取。

    cfg = load_config("us")                 # 读 setting/us.yaml
    market = cfg.load(start="2024-01-01")   # 交易池的 MarketData
    result = cfg.run(strategy, start="2024-01-01", end="2024-12-31")

字段写错、类型不对、数值越界都在加载时报错(ConfigError), 不会拖到回测中途。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from event_backtest.engine import BacktestResult, Context, ContextStrategy
from event_backtest.engine import run as run_backtest
from event_backtest.fees import FeeSchedule
from event_backtest.market import DateLike, MarketData, load_market
from event_backtest.rules import MarketRules, default_rules
from event_backtest.slippage import SlippageModel, make_slippage

SETTING_DIR = Path(__file__).parent / "setting"
_FREQUENCIES = ("5", "15", "30", "60")
_FILL_PRICES = ("open", "vwap")
# 市场配置 yaml 允许的顶层段
_TOP_KEYS = ("market", "data", "universe", "benchmark", "account", "execution", "fees", "rules")


class ConfigError(ValueError):
    """配置文件写错了。"""


def default_fees(market: str) -> dict[str, FeeSchedule]:
    """按市场给出常见费率(按 sec_type 分档)。

    A 股: 佣金万 2.5(最低 5 元), 股票加印花税(卖出 0.05%)与过户费(双向 0.001%)。
    美股: 默认 0 佣金(很多券商已零佣), 需要时在配置里覆盖。

    Args:
        market: "cn" 或 "us"。

    Returns:
        {sec_type: FeeSchedule}。

    Raises:
        ValueError: market 不是这两个之一。
    """
    if market == "cn":
        return {
            "etf": FeeSchedule(commission=0.00025, min_commission=5.0),
            "stock": FeeSchedule(commission=0.00025, min_commission=5.0,
                                 stamp_duty=0.0005, transfer_fee=0.00001),
        }
    if market == "us":
        return {"stock": FeeSchedule()}
    raise ValueError(f"未知市场 {market!r}, 只支持 'cn' / 'us'")


@dataclass(frozen=True)
class MarketConfig:
    """一个市场的回测配置; 规则与费率不写就用默认。

    Attributes:
        market: "cn" / "us"。
        db_path: 数据库文件路径(相对路径在加载时已解析成绝对路径)。
        initial_cash: 初始现金。
        fill_price: 市价单成交价来源, "open" / "vwap"。
        freq: A 股分钟线周期; None 表示日线。
        tickers: 交易池; None 表示库里出现的全部证券。
        benchmark: 基准代码(目前只记录, 不参与计算)。
        slippage: 滑点模型; None 表示不调价。
        rules: 交易规则; None 表示按 market 取默认。
        fees: {sec_type: FeeSchedule}; None 表示按 market 取默认。
        path: 来源 yaml 路径。
    """

    market: str
    db_path: str | Path
    initial_cash: float = 1_000_000.0
    fill_price: str = "open"
    freq: str | None = None
    tickers: tuple[str, ...] | None = None
    benchmark: str | None = None
    slippage: SlippageModel | None = None
    rules: MarketRules | None = None
    fees: dict[str, FeeSchedule] | None = None
    path: Path | None = None

    def resolved_rules(self) -> MarketRules:
        """取生效的交易规则: 配置里写了的用它, 否则按市场取默认。"""
        return self.rules or default_rules(self.market)

    def resolved_fees(self) -> dict[str, FeeSchedule]:
        """取生效的费率表: 配置里写了的用它, 否则按市场取默认。"""
        return self.fees if self.fees is not None else default_fees(self.market)

    def load(self, start: DateLike | None = None, end: DateLike | None = None,
             freq: str | int | None = None) -> MarketData:
        """读出行情。

        Args:
            start / end: 交易日, 两端含; 默认不限。
            freq: 临时覆盖配置里的 freq; None 表示用配置值。

        Returns:
            MarketData, 已套用本配置的交易规则。
        """
        return load_market(self.db_path, self.market, self.tickers, start, end,
                           freq=self.freq if freq is None else freq,
                           rules=self.resolved_rules())

    def run(self, strategy: ContextStrategy | Callable[[Context], None],
            start: DateLike | None = None, end: DateLike | None = None,
            freq: str | int | None = None) -> BacktestResult:
        """读出行情并跑一遍回测, 规则 / 费率 / 成交价 / 滑点 / 初始资金都来自本配置。

        Args:
            strategy: 策略对象(回调协议或声明式适配器)。
            start / end: 交易日, 两端含。
            freq: 临时覆盖配置里的 freq。

        Returns:
            BacktestResult; meta 里带上本配置的上下文(market / benchmark / db_path /
            fill_price / freq 等), 后续 summarize / benchmark_nav 可以直接用。
        """
        market = self.load(start, end, freq)
        result = run_backtest(strategy, market, initial_cash=self.initial_cash,
                              fees=self.resolved_fees(), fill_price=self.fill_price,
                              slippage=self.slippage, rules=self.resolved_rules())
        result.meta.update(self.describe())
        return result

    def describe(self) -> dict[str, Any]:
        """本配置的可序列化摘要(给 BacktestResult.meta 与 run.json 清单用)。

        Returns:
            含 market / benchmark / db_path / config_path / initial_cash / fill_price /
            freq / tickers / slippage / fees 的字典。
        """
        return {
            "market": self.market,
            "benchmark": self.benchmark,
            "db_path": str(self.db_path),
            "config_path": str(self.path) if self.path is not None else None,
            "initial_cash": float(self.initial_cash),
            "fill_price": self.fill_price,
            "freq": self.freq,
            "tickers": list(self.tickers) if self.tickers else None,
            "slippage": _describe_slippage(self.slippage),
            "fees": {name: asdict(schedule) for name, schedule in self.resolved_fees().items()},
        }


def _describe_slippage(model: SlippageModel | None) -> dict[str, Any] | None:
    """滑点模型 -> 可序列化摘要; None 表示不调价。"""
    if model is None:
        return None
    if is_dataclass(model):
        return {"type": type(model).__name__, **asdict(model)}
    return {"type": type(model).__name__}


def load_config(which: str | Path = "us", *,
                setting_dir: str | Path = SETTING_DIR) -> MarketConfig:
    """读市场配置并校验。

    Args:
        which: 市场名(在 setting_dir 里找 <which>.yaml), 或一个 .yaml 路径。
        setting_dir: 按市场名查找时的目录, 默认同目录下的 setting/。

    Returns:
        MarketConfig; db_path 已解析成绝对路径。

    Raises:
        ConfigError: 文件不存在, 或任何字段校验不通过。
    """
    path = Path(which)
    if path.suffix not in (".yaml", ".yml"):
        path = Path(setting_dir) / f"{which}.yaml"
    if not path.is_file():
        raise ConfigError(f"找不到配置 {str(which)!r}({path})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: 顶层必须是一个映射")
    return _parse_config(raw, path)


# ---------------------------------------------------------------- 解析

def _parse_config(raw: dict[str, Any], path: Path) -> MarketConfig:
    """校验一份市场配置并组装成 MarketConfig。"""
    where = path.name
    _check_keys(raw, _TOP_KEYS, where)
    market = raw.get("market")
    if market not in ("cn", "us"):
        raise ConfigError(f"{where}: market 只能是 'cn' / 'us', 实际是 {market!r}")

    data = _section(raw, "data", ("db_path", "freq"), where, required=True)
    if not data.get("db_path"):
        raise ConfigError(f"{where}: 缺 data.db_path")
    # 相对路径相对 yaml 所在目录解析
    db_path = Path(str(data["db_path"]))
    if not db_path.is_absolute():
        db_path = (path.parent / db_path).resolve()

    freq = data.get("freq")
    if freq is not None:
        freq = str(freq).strip()
        if freq not in _FREQUENCIES:
            raise ConfigError(f"{where}: data.freq 不支持 {freq!r}, 可选 {list(_FREQUENCIES)}")
        if market == "us":
            raise ConfigError(f"{where}: 美股只有日线, data.freq 只能为空")

    universe = raw.get("universe")
    if universe is not None and (not isinstance(universe, list) or not universe):
        raise ConfigError(f"{where}: universe 必须是非空列表")
    tickers = tuple(str(t) for t in universe) if universe else None

    benchmark = raw.get("benchmark")
    account = _section(raw, "account", ("initial_cash",), where)
    execution = _section(raw, "execution", ("fill_price", "slippage"), where)
    fill_price = str(execution.get("fill_price", "open"))
    if fill_price not in _FILL_PRICES:
        raise ConfigError(f"{where}: execution.fill_price 只能是 {list(_FILL_PRICES)}")

    return MarketConfig(
        market=market,
        db_path=db_path,
        initial_cash=_number(account.get("initial_cash", 1_000_000.0),
                             f"{where}: account.initial_cash", positive=True),
        fill_price=fill_price,
        freq=freq,
        tickers=tickers,
        benchmark=str(benchmark) if benchmark is not None else None,
        slippage=_parse_slippage(execution.get("slippage"), f"{where}: execution.slippage"),
        rules=_parse_rules(market, raw.get("rules"), where),
        fees=_parse_fees(market, raw.get("fees"), where),
        path=path,
    )


def _parse_rules(market: str, raw: Any, where: str) -> MarketRules:
    """rules 段 -> 交易规则: 先取默认规则, 再用 yaml 里写了的字段覆盖。"""
    rules = default_rules(market)
    if raw is None:
        return rules
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: rules 必须是映射")
    # 允许覆盖的字段 = 该规则 dataclass 的字段(美股没有 A 股那些板块字段)
    allowed = tuple(f.name for f in fields(rules))
    _check_keys(raw, allowed, f"{where}: rules")
    kw: dict[str, Any] = {}
    for key, value in raw.items():
        loc = f"{where}: rules.{key}"
        if key == "lot_size":
            kw[key] = _positive_int(value, loc)
        elif key == "st_limits":
            if not isinstance(value, dict) or not value:
                raise ConfigError(f"{loc} 必须是非空映射 {{生效日: 幅度}}")
            kw[key] = tuple(sorted((_date(d, loc), _number(p, f"{loc}.{d}"))
                                   for d, p in value.items()))
        elif key in ("growth_fund_keywords", "t0_fund_keywords"):
            kw[key] = _str_tuple(value, loc)
        elif key == "limit_overrides":
            kw[key] = {str(t): _number(p, loc) for t, p in _mapping(value, loc).items()}
        elif key == "t0_overrides":
            kw[key] = {str(t): _bool(p, loc) for t, p in _mapping(value, loc).items()}
        else:
            kw[key] = _number(value, loc)
    return replace(rules, **kw)


def _parse_slippage(raw: Any, where: str) -> SlippageModel | None:
    """execution.slippage -> 滑点模型; 不写返回 None。

    三种写法: {type: none} / {type: fixed, spread: 0.01} /
    {type: volume_share, volume_limit: 0.025, price_impact: 0.1}。
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} 必须是映射")
    kind = raw.get("type")
    if not isinstance(kind, str) or not kind:
        raise ConfigError(f"{where} 缺 type")
    loc = f"{where}({kind})"
    if kind == "none":
        _check_keys(raw, ("type",), loc)
        return make_slippage("none")
    if kind == "fixed":
        _check_keys(raw, ("type", "spread"), loc)
        return make_slippage("fixed", spread=_number(raw.get("spread", 0.0), f"{loc}.spread"))
    if kind == "volume_share":
        _check_keys(raw, ("type", "volume_limit", "price_impact"), loc)
        limit = _number(raw.get("volume_limit", 0.025), f"{loc}.volume_limit", positive=True)
        if limit > 1:
            raise ConfigError(f"{loc}.volume_limit 不能大于 1, 实际是 {limit!r}")
        impact = _number(raw.get("price_impact", 0.1), f"{loc}.price_impact")
        return make_slippage("volume_share", volume_limit=limit, price_impact=impact)
    raise ConfigError(f"{loc}: 不认识的 type {kind!r}, 可用 none / fixed / volume_share")


def _parse_fees(market: str, raw: Any, where: str) -> dict[str, FeeSchedule]:
    """fees 段 -> {sec_type: FeeSchedule}; 不写用默认费率。

    每类证券的字段可以只写一部分(其余用 FeeSchedule 的默认 0)。
    """
    if raw is None:
        return default_fees(market)
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{where}: fees 必须是非空映射, 如 {{stock: {{commission: 0.0}}}}")
    allowed = tuple(f.name for f in fields(FeeSchedule))
    out: dict[str, FeeSchedule] = {}
    for sec_type, body in raw.items():
        loc = f"{where}: fees.{sec_type}"
        if not isinstance(body, dict):
            raise ConfigError(f"{loc} 必须是映射")
        _check_keys(body, allowed, loc)
        out[str(sec_type)] = FeeSchedule(**{k: _number(v, f"{loc}.{k}")
                                            for k, v in body.items()})
    return out


# ---------------------------------------------------------------- 小工具

def _check_keys(raw: dict[str, Any], allowed: tuple[str, ...], loc: str) -> None:
    """有不认识的字段就报错并列出可用字段(多半是拼错)。"""
    if unknown := [k for k in raw if k not in allowed]:
        raise ConfigError(f"{loc}: 不认识的字段 {unknown}, 可用的有 {list(allowed)}")


def _section(raw: dict[str, Any], key: str, allowed: tuple[str, ...], where: str, *,
             required: bool = False) -> dict[str, Any]:
    """取出一个段(映射)并检查段内字段名; 缺段时 required=True 报错, 否则返回空 dict。"""
    body = raw.get(key)
    if body is None:
        if required:
            raise ConfigError(f"{where}: 缺 {key} 段")
        return {}
    if not isinstance(body, dict):
        raise ConfigError(f"{where}: {key} 必须是映射")
    _check_keys(body, allowed, f"{where}: {key}")
    return body


def _mapping(value: Any, loc: str) -> dict[Any, Any]:
    """可选的映射字段: None 当空映射, 其他非映射报错。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{loc} 必须是映射")
    return value


def _number(value: Any, loc: str, *, positive: bool = False) -> float:
    """校验成数字并返回 float; bool 不算数字(避免 True 被当成 1)。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{loc} 必须是数字, 实际是 {value!r}")
    if value < 0 or (positive and value == 0):
        raise ConfigError(f"{loc} 必须{'> 0' if positive else '>= 0'}, 实际是 {value!r}")
    return float(value)


def _positive_int(value: Any, loc: str) -> int:
    """校验成正整数; bool 和小数都不行。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{loc} 必须是正整数, 实际是 {value!r}")
    return value


def _bool(value: Any, loc: str) -> bool:
    """校验成布尔值。"""
    if not isinstance(value, bool):
        raise ConfigError(f"{loc} 必须是 true / false, 实际是 {value!r}")
    return value


def _str_tuple(value: Any, loc: str) -> tuple[str, ...]:
    """校验成非空字符串组成的列表, 返回元组。"""
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise ConfigError(f"{loc} 必须是字符串列表")
    return tuple(value)


def _date(value: Any, loc: str) -> date:
    """把 yaml 读出的日期(或 ISO 字符串)转成 date。"""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ConfigError(f"{loc}: {value!r} 不是日期(如 2026-07-06)") from None
