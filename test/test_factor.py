"""因子求值器: shift / rolling / 四则 / 参数校验 / 无未来函数。"""
from __future__ import annotations

import numpy as np
import pytest

from event_backtest.factor import FactorError, compute, parse_factor


def _ratio_spec():
    return parse_factor({
        "name": "f", "parameters": ["window"],
        "steps": [
            {"id": "prev", "operation": "shift",
             "input": {"variable": "price"}, "periods": {"constant": 1}},
            {"id": "result", "operation": "divide",
             "numerator": {"variable": "price"}, "denominator": {"variable": "prev"}},
        ],
        "output": "result",
    })


def test_shift_and_divide():
    spec = _ratio_spec()
    frames = {"price": np.array([[1.0], [2.0], [4.0], [8.0]])}
    out = compute(spec, frames, {"window": 1})
    assert np.isnan(out[0, 0])
    assert list(out[1:, 0]) == pytest.approx([2.0, 2.0, 2.0])


def test_rolling_mean_values():
    spec = parse_factor({
        "name": "ma", "parameters": ["window"],
        "steps": [{"id": "result", "operation": "rolling_mean",
                   "input": {"variable": "price"}, "window": {"parameter": "window"}}],
        "output": "result",
    })
    out = compute(spec, {"price": np.array([[1.0], [2.0], [3.0], [4.0]])}, {"window": 2})
    assert np.isnan(out[0, 0])
    assert list(out[1:, 0]) == pytest.approx([1.5, 2.5, 3.5])


def test_no_future_leak():
    spec = parse_factor({
        "name": "ma", "parameters": ["window"],
        "steps": [{"id": "result", "operation": "rolling_mean",
                   "input": {"variable": "price"}, "window": {"parameter": "window"}}],
        "output": "result",
    })
    a = np.arange(1, 11, dtype=float).reshape(-1, 1)
    frames = {"price": a.copy()}
    base = compute(spec, frames, {"window": 3})
    frames2 = {"price": a.copy()}
    frames2["price"][5:] = 999.0
    changed = compute(spec, frames2, {"window": 3})
    assert np.allclose(base[:5], changed[:5], equal_nan=True)
    assert not np.allclose(base[5:], changed[5:], equal_nan=True)


def test_parameter_checks():
    spec = _ratio_spec()
    frames = {"price": np.ones((3, 1))}
    with pytest.raises(FactorError):
        compute(spec, frames, {})                # 缺参数
    with pytest.raises(FactorError):
        compute(spec, frames, {"window": 1, "x": 2})   # 多参数


def test_unknown_operation_and_output():
    with pytest.raises(FactorError):
        parse_factor({"name": "bad", "steps": [{"id": "r", "operation": "nope"}], "output": "r"})
    with pytest.raises(FactorError):
        parse_factor({"name": "bad", "steps": [
            {"id": "r", "operation": "shift", "input": {"variable": "price"},
             "periods": {"constant": 1}}], "output": "missing"})


def test_divide_by_zero_is_nan():
    spec = parse_factor({
        "name": "d",
        "steps": [{"id": "result", "operation": "divide",
                   "numerator": {"constant": 1.0}, "denominator": {"variable": "volume"}}],
        "output": "result",
    })
    out = compute(spec, {"volume": np.array([[0.0], [2.0]])}, {})
    assert np.isnan(out[0, 0])
    assert out[1, 0] == pytest.approx(0.5)
