"""信号: enter / every_bar / all&any / NaN / 校验。"""
from __future__ import annotations

import numpy as np
import pytest
from helpers import make_market

from event_backtest.signal import SignalError, generate_events, parse_signal


def _signal(**over):
    raw = {"name": "s", "trigger": "enter",
           "all": [{"factor": "x", "operation": "greater_than", "value": 2}]}
    raw.update(over)
    return parse_signal(raw)


def test_enter_only_on_rising_edge():
    m = make_market([1.0, 3.0, 3.0, 1.0, 3.0])
    events = generate_events(_signal(), {"x": m.close}, m, strategy_name="t")
    assert list(events["bar"]) == [1, 4]


def test_every_bar():
    m = make_market([1.0, 3.0, 3.0, 1.0, 3.0])
    events = generate_events(_signal(trigger="every_bar"), {"x": m.close}, m)
    assert list(events["bar"]) == [1, 2, 4]


def test_nan_is_not_satisfied():
    m = make_market([1.0, 3.0, 3.0])
    values = {"x": np.array([[1.0], [np.nan], [5.0]])}
    events = generate_events(_signal(), values, m)
    assert list(events["bar"]) == [2]


def test_all_and_any_with_every_bar():
    m = make_market([1.0, 3.0, 5.0])
    values = {"a": m.close, "b": np.array([[0.0], [1.0], [0.0]])}
    sig_all = parse_signal({"name": "s", "trigger": "every_bar", "all": [
        {"factor": "a", "operation": "greater_than", "value": 2},
        {"factor": "b", "operation": "greater_than", "value": 0.5}]})
    assert list(generate_events(sig_all, values, m)["bar"]) == [1]
    sig_any = parse_signal({"name": "s", "trigger": "every_bar", "any": [
        {"factor": "a", "operation": "greater_than", "value": 2},
        {"factor": "b", "operation": "greater_than", "value": 0.5}]})
    assert list(generate_events(sig_any, values, m)["bar"]) == [1, 2]


def test_parse_errors():
    with pytest.raises(SignalError):
        parse_signal({"name": "s"})                            # 缺 all/any
    with pytest.raises(SignalError):
        parse_signal({"name": "s", "all": [], "any": []})      # 两个都写
    with pytest.raises(SignalError):
        parse_signal({"name": "s", "all": [{"factor": "x", "operation": "nope", "value": 1}]})
    with pytest.raises(SignalError):
        parse_signal({"name": "s", "all": [
            {"factor": "x", "operation": "greater_than", "value": float("nan")}]})


def test_missing_factor_raises():
    m = make_market([1.0, 2.0])
    with pytest.raises(SignalError):
        generate_events(_signal(), {"y": m.close}, m)
