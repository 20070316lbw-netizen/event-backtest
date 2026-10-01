"""事件研究: 事件后收益、基线与空表处理。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import make_market

from event_backtest.evaluation import study_events


def _events(*bars):
    return pd.DataFrame([
        {"event_id": f"e{b}", "bar": b, "ts": pd.Timestamp("1970-01-02"),
         "ticker": "T0", "signal": "s"}
        for b in bars
    ])


def test_forward_returns_and_baseline():
    m = make_market([[1.0], [2.0], [4.0], [8.0]])
    study = study_events(m, _events(1), horizons=(1, 2))
    paths = study.paths.set_index("horizon")
    assert paths.loc[1, "ret"] == pytest.approx(4 / 2 - 1)
    assert paths.loc[2, "ret"] == pytest.approx(8 / 2 - 1)
    # 单调序列: horizon1 的所有窗口收益都是 1, 基线为 1, 超额为 0
    assert paths.loc[1, "baseline_ret"] == pytest.approx(1.0)
    assert paths.loc[1, "excess_ret"] == pytest.approx(0.0)
    summary = study.summary.set_index("horizon")
    assert summary.loc[1, "n"] == 1
    assert summary.loc[1, "mean_ret"] == pytest.approx(1.0)


def test_missing_tail_is_nan():
    m = make_market([[1.0], [2.0], [3.0]])
    study = study_events(m, _events(2), horizons=(1, 2))   # bar2 + 2 越界
    paths = study.paths.set_index("horizon")
    assert np.isnan(paths.loc[2, "ret"])
    assert study.summary.set_index("horizon").loc[2, "n"] == 0


def test_empty_events():
    m = make_market([[1.0], [2.0]])
    empty = pd.DataFrame(columns=["event_id", "bar", "ts", "ticker", "signal"])
    study = study_events(m, empty)
    assert study.paths.empty
    assert list(study.summary.columns) == ["horizon", "n_events", "n", "mean_ret",
                                           "median_ret", "win_rate", "baseline", "excess"]
