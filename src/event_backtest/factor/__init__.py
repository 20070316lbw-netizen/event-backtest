"""因子: YAML 定义 + 自包含求值器(不依赖 minibacktest)。

YAML 字段一律写全称(parameters / operation / variable / parameter / constant), 格式与
minievent 保持一致。因子以 T+0 为前提书写: 第 t 根可以用截至第 t 根收盘(含)的数据;
防未来函数由引擎负责(第 t 根算信号, 第 t+1 根成交), 因子不要自己 shift(1)。

因子看到的行情(都已对齐成 (T, N)):
    price / close        后复权收盘价 close * adj_factor
    open / high / low    同样乘 adj_factor
    volume               原始成交量

一份因子 yaml 长这样(见 relative_volume.yaml / momentum.yaml)::

    name: relative_volume
    parameters: [window]
    steps:
      - id: previous_volume
        operation: shift
        input: {variable: volume}
        periods: {constant: 1}
      - id: result
        operation: divide
        numerator: {variable: volume}
        denominator: {variable: previous_volume}
    output: result

Example:
    >>> from event_backtest.factor import compute, load_specs, market_frames
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from event_backtest.market import MarketData

__all__ = [
    "FACTOR_DIR",
    "FactorError",
    "FactorSpec",
    "compute",
    "load_specs",
    "market_frames",
]

FACTOR_DIR = Path(__file__).parent

# yaml 顶层允许的字段
_TOP_KEYS = ("name", "description", "parameters", "steps", "output")

# 每种运算需要的字段(按书写顺序)。求值时按这张表取操作数。
_OPERATIONS: dict[str, tuple[str, ...]] = {
    "add": ("left", "right"),
    "subtract": ("left", "right"),
    "multiply": ("left", "right"),
    "divide": ("numerator", "denominator"),
    "elementwise_max": ("left", "right"),
    "elementwise_min": ("left", "right"),
    "shift": ("input", "periods"),
    "rolling_mean": ("input", "window"),
    "rolling_std": ("input", "window"),
    "rolling_min": ("input", "window"),
    "rolling_max": ("input", "window"),
    "rolling_sum": ("input", "window"),
    "rolling_corr": ("left", "right", "window"),
    "cross_section_rank": ("input",),
}
# rolling_* 到 pandas 滚动方法名的映射
_ROLLING = {"rolling_mean": "mean", "rolling_std": "std", "rolling_min": "min",
            "rolling_max": "max", "rolling_sum": "sum"}


class FactorError(ValueError):
    """因子 yaml 写错, 或求值时引用了不存在的变量 / 参数。"""


@dataclass(frozen=True)
class FactorSpec:
    """一个已静态校验的因子定义。

    Attributes:
        name: 因子名(默认取文件名)。
        description: 说明文字。
        params: 参数名列表, 按书写顺序。
        steps: 步骤列表, 每步是一个 dict(id / operation / 各运算字段)。
        output: 最终输出引用的变量名(某个步骤 id 或行情字段)。
        path: 来源 yaml 路径; 手动构造时可以是 None。
    """

    name: str
    description: str
    params: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]
    output: str
    path: Path | None = None


def load_specs(factor_dir: str | Path = FACTOR_DIR) -> dict[str, FactorSpec]:
    """扫描目录下所有 *.yaml, 解析并静态校验成 {因子名: FactorSpec}。

    Args:
        factor_dir: 因子目录, 默认本包目录。

    Returns:
        {因子名: FactorSpec}, 按文件名排序。

    Raises:
        FactorError: 某个 yaml 校验不通过, 或两个文件用了同一个因子名。
    """
    specs: dict[str, FactorSpec] = {}
    for path in sorted(Path(factor_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = parse_factor(raw, path=path)
        if spec.name in specs:
            raise FactorError(
                f"因子名 {spec.name!r} 重复: {specs[spec.name].path.name} 和 {path.name}")
        specs[spec.name] = spec
    return specs


def parse_factor(raw: Any, *, path: Path | None = None) -> FactorSpec:
    """校验一份因子 yaml(全称字段)并返回 FactorSpec。

    校验是静态的: 只查字段名、运算名、步骤 id、output 引用是否存在, 不跑数据。

    Args:
        raw: yaml.safe_load 出来的内容。
        path: 来源路径, 只用于报错信息与默认因子名。

    Returns:
        FactorSpec。

    Raises:
        FactorError: 顶层不是映射、字段名不认识、steps 为空、运算不存在或缺字段、
            步骤 id 重复、output 引用不到。
    """
    where = path.name if path else "因子"
    if not isinstance(raw, dict):
        raise FactorError(f"{where}: 顶层必须是一个映射(name / parameters / steps / output)")
    _check_unknown(raw, _TOP_KEYS, where)

    params = raw.get("parameters") or ()
    if not isinstance(params, (list, tuple)) or not all(isinstance(p, str) and p for p in params):
        raise FactorError(f"{where}: parameters 必须是字符串列表")

    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise FactorError(f"{where}: steps 必须是非空列表")

    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for number, step in enumerate(steps, start=1):
        parsed.append(_parse_step(step, number, where, seen))

    output = raw.get("output")
    if not isinstance(output, str) or not output:
        raise FactorError(f"{where}: 缺 output")
    # output 只能引用行情字段或前面某个步骤的 id
    known = {"price", "close", "open", "high", "low", "volume"} | seen
    if output not in known:
        raise FactorError(
            f"{where}: output {output!r} 不是行情字段也不是步骤 id, 可用的有 {sorted(known)}")

    return FactorSpec(
        name=str(raw.get("name") or (path.stem if path else "factor")),
        description=str(raw.get("description") or ""),
        params=tuple(params),
        steps=tuple(parsed),
        output=output,
        path=path,
    )


def _parse_step(step: Any, number: int, where: str, seen: set[str]) -> dict[str, Any]:
    """校验因子里的一个步骤, 并把它的 id 记进 seen。"""
    if not isinstance(step, dict):
        raise FactorError(f"{where} 第 {number} 步必须是映射")
    step_id = step.get("id")
    loc = f"{where} 的步骤 {step_id!r}" if step_id else f"{where} 第 {number} 步"
    operation = step.get("operation")
    if not operation:
        raise FactorError(f"{loc} 缺 operation")
    if operation not in _OPERATIONS:
        raise FactorError(
            f"{loc} 的 operation {operation!r} 不认识, 可用的有 {sorted(_OPERATIONS)}")
    fields = _OPERATIONS[operation]
    # 只允许 id / operation / 该运算的字段, 多写少写都报错
    _check_unknown(step, ("id", "operation", *fields), loc)
    if missing := [f for f in fields if f not in step]:
        raise FactorError(f"{loc}({operation}) 缺字段 {missing}")
    for field in fields:
        _check_operand(step[field], f"{loc} 的 {field}")
    if step_id:
        if step_id in seen:
            raise FactorError(f"{where}: 步骤 id {step_id!r} 重复")
        seen.add(step_id)
    return step


def _check_operand(operand: Any, loc: str) -> None:
    """校验操作数写法: 恰好一个键, 且键是 variable / parameter / constant。"""
    if not isinstance(operand, dict) or len(operand) != 1:
        raise FactorError(f"{loc} 写法不对, 应为恰好一个键的映射: variable / parameter / constant")
    (kind, _), = operand.items()
    if kind not in ("variable", "parameter", "constant"):
        raise FactorError(
            f"{loc} 的操作数类型 {kind!r} 不认识, 可用 variable / parameter / constant")


def market_frames(market: MarketData) -> dict[str, np.ndarray]:
    """MarketData -> 因子用的行情字典: 价格后复权, 成交量不动。

    因子不声明自己要用哪些字段, 这里总是把 open / high / low / volume 全部给出去,
    多传不影响计算。

    Args:
        market: 对齐后的行情。

    Returns:
        {"price" / "close": adj_close, "open" / "high" / "low": 乘了 adj_factor,
         "volume": 原始成交量}; 都是 (T, N) float 数组。
    """
    adj = market.adj_factor
    price = market.adj_close
    return {
        "price": price,
        "close": price,
        "open": market.open * adj,
        "high": market.high * adj,
        "low": market.low * adj,
        "volume": market.volume,
    }


def compute(spec: FactorSpec, frames: Mapping[str, np.ndarray],
            params: Mapping[str, Any]) -> np.ndarray:
    """算一个因子, 返回 (T, N) float64 数组。

    逐步执行 spec.steps: 每一步的操作数从行情字典或前面步骤的结果里取, 结果以步骤 id
    存回局部字典, 最后返回 output 指向的那个数组。

    Args:
        spec: 因子定义。
        frames: market_frames 的结果; 同一份行情算多个因子时复用。
        params: 因子参数, 如 {"window": 16}; 必须和 spec.params 完全一致。

    Returns:
        (T, N) float64, 缺失为 NaN。

    Raises:
        FactorError: 参数不匹配, 或引用了不存在的变量 / 参数 / output。
    """
    _check_params(spec, params)
    values: dict[str, Any] = dict(frames)
    for step in spec.steps:
        operation = step["operation"]
        # 按运算字段表逐个解析操作数
        args = {field: _operand(step[field], values, params, spec)
                for field in _OPERATIONS[operation]}
        result = _apply(operation, args)
        if step.get("id"):
            values[step["id"]] = result
    if spec.output not in values:
        raise FactorError(f"因子 {spec.name!r}: output {spec.output!r} 没有可用的值")
    return np.asarray(values[spec.output], dtype=float)


def _check_params(spec: FactorSpec, params: Mapping[str, Any]) -> None:
    """参数必须和 spec.params 完全一致(不缺不多), 否则报错。"""
    missing = [p for p in spec.params if p not in params]
    extra = [p for p in params if p not in spec.params]
    if missing or extra:
        raise FactorError(f"因子 {spec.name!r} 参数不对: 缺 {missing}, 多 {extra}, "
                          f"声明的是 {list(spec.params)}")


def _operand(operand: Mapping[str, Any], values: Mapping[str, Any],
             params: Mapping[str, Any], spec: FactorSpec) -> Any:
    """解析一个操作数: {variable: x} / {parameter: x} / {constant: v}。"""
    (kind, value), = operand.items()
    if kind == "variable":
        if value not in values:
            raise FactorError(
                f"因子 {spec.name!r}: 变量 {value!r} 不存在, 可用的有 {sorted(values)}")
        return values[value]
    if kind == "parameter":
        if value not in params:
            raise FactorError(f"因子 {spec.name!r}: 参数 {value!r} 没有给")
        return params[value]
    return float(value)


def _apply(operation: str, args: Mapping[str, Any]) -> np.ndarray:
    """执行一个运算, 返回 (T, N) 数组。"""
    if operation == "add":
        return np.asarray(args["left"], float) + np.asarray(args["right"], float)
    if operation == "subtract":
        return np.asarray(args["left"], float) - np.asarray(args["right"], float)
    if operation == "multiply":
        return np.asarray(args["left"], float) * np.asarray(args["right"], float)
    if operation == "divide":
        return _divide(args["numerator"], args["denominator"])
    if operation == "elementwise_max":
        return np.maximum(np.asarray(args["left"], float), np.asarray(args["right"], float))
    if operation == "elementwise_min":
        return np.minimum(np.asarray(args["left"], float), np.asarray(args["right"], float))
    if operation == "shift":
        return _shift(np.asarray(args["input"], float), int(args["periods"]))
    if operation in _ROLLING:
        return _rolling(_ROLLING[operation], np.asarray(args["input"], float), int(args["window"]))
    if operation == "rolling_corr":
        return _rolling_corr(np.asarray(args["left"], float),
                             np.asarray(args["right"], float), int(args["window"]))
    return _cross_section_rank(np.asarray(args["input"], float))


def _divide(numerator: Any, denominator: Any) -> np.ndarray:
    """逐元素相除; 分母为 0 的位置记 NaN(而不是 inf), 免得污染后续比较。"""
    num = np.asarray(numerator, float)
    den = np.asarray(denominator, float)
    shape = np.broadcast_shapes(num.shape, den.shape)
    out = np.full(shape, np.nan)
    np.divide(num, den, out=out, where=np.broadcast_to(den, shape) != 0)
    return out


def _shift(values: np.ndarray, periods: int) -> np.ndarray:
    """往后挪 periods 根 bar: 第 t 根取第 t-periods 根, 前面补 NaN。

    Args:
        values: (T, N) 数组。
        periods: 往后挪的 bar 数, 必须 >= 0。

    Returns:
        同形状数组; periods >= T 时全为 NaN。
    """
    if periods < 0:
        raise FactorError(f"shift 的 periods 不能为负: {periods}")
    if periods == 0:
        return values.astype(float)
    out = np.full(values.shape, np.nan)
    if periods < values.shape[0]:
        out[periods:] = values[:-periods]
    return out


def _rolling(method: str, values: np.ndarray, window: int) -> np.ndarray:
    """按时间轴对每一列做滚动统计; 前 window-1 根为 NaN(min_periods=window)。"""
    if window <= 0:
        raise FactorError(f"rolling 的 window 必须为正: {window}")
    rolled = pd.DataFrame(values).rolling(window, min_periods=window)
    return getattr(rolled, method)().to_numpy(dtype=float)


def _rolling_corr(left: np.ndarray, right: np.ndarray, window: int) -> np.ndarray:
    """两列各自按时间轴做 window 长的滚动相关系数。"""
    if window <= 0:
        raise FactorError(f"rolling_corr 的 window 必须为正: {window}")
    ld, rd = pd.DataFrame(left), pd.DataFrame(right)
    return ld.rolling(window, min_periods=window).corr(rd).to_numpy(dtype=float)


def _cross_section_rank(values: np.ndarray) -> np.ndarray:
    """每个时点(行)在证券(列)之间排名, 归一化到 (0, 1]。"""
    return pd.DataFrame(values).rank(axis=1, pct=True).to_numpy(dtype=float)


def _check_unknown(raw: Mapping[str, Any], allowed: Sequence[str], loc: str) -> None:
    """有不认识的字段就报错, 并列出可用字段(多半是拼错或写了缩写)。"""
    if unknown := [k for k in raw if k not in allowed]:
        raise FactorError(f"{loc} 有不认识的字段 {unknown}, 可用的有 {list(allowed)}")
