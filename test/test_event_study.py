"""事件研究: 事件后收益、基线与空表处理; 显著性检验与聚类。"""
from __future__ import annotations

from dataclasses import replace

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


def _market_from_returns(returns, *, start: str = "2024-01-01", freq: str = "D"):
    """用日收益序列造一段行情: close[k+1]/close[k] - 1 = returns[k]。"""
    close = 100.0 * np.cumprod(1.0 + np.asarray(returns, dtype=float))
    close = np.concatenate([[100.0], close])
    market = make_market(close[:, None], tickers=["AAA"])
    stamps = pd.date_range(start, periods=len(close), freq=freq)
    return replace(market, ts=stamps.to_numpy())


def _events_at(market, *bars, ticker: str = "AAA"):
    """事件时间戳取自行情, 这样"按日聚类"才有意义。"""
    return pd.DataFrame([
        {"event_id": f"e{b}", "bar": b, "ts": pd.Timestamp(market.ts[b]),
         "ticker": ticker, "signal": "s"}
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
    assert study.tests.empty
    assert list(study.tests.columns) == ["horizon", "n", "n_obs", "n_days", "mean", "t",
                                         "p", "ci_low", "ci_high", "p_boot", "stars"]
    assert study.tests_table().empty


def _jump_market():
    """5 个事件各自后面跟着一跳(幅度略有差异, 免得标准差为 0)。"""
    event_bars = (4, 14, 24, 34, 44)
    jumps = (0.021, 0.019, 0.022, 0.018, 0.020)
    returns = np.zeros(60)
    for bar, value in zip(event_bars, jumps, strict=True):
        returns[bar] = value
    market = _market_from_returns(returns)
    return market, _events_at(market, *event_bars), event_bars


def test_tests_detect_signal_after_events():
    market, events, _ = _jump_market()
    study = study_events(market, events, horizons=(1, 5), bootstrap=2000, seed=0)
    row = study.tests.set_index("horizon").loc[1]

    assert row["n"] == 5 and row["n_obs"] == 5 and row["n_days"] == 5
    assert row["mean"] > 0.015
    assert row["p"] < 0.01 and row["p_boot"] < 0.01
    assert row["stars"] == "***"
    assert row["ci_low"] > 0                       # 区间不含 0
    # horizon 5 把那一跳摊到 5 根里, 效应还在但更小
    assert study.tests.set_index("horizon").loc[5, "mean"] < row["mean"]


def test_excess_and_raw_return_are_different_targets():
    market, events, _ = _jump_market()
    excess = study_events(market, events, horizons=(1,), test_on="excess", bootstrap=0)
    raw = study_events(market, events, horizons=(1,), test_on="ret", bootstrap=0)
    # 基线含这些跳本身(正值), 所以"超额"必然小于"原始收益"
    assert raw.tests.iloc[0]["mean"] > excess.tests.iloc[0]["mean"]
    assert np.isnan(excess.tests.iloc[0]["p_boot"])     # bootstrap=0 不算


def test_tests_do_not_flag_pure_noise():
    rng = np.random.default_rng(3)
    market = _market_from_returns(rng.normal(0.0, 0.01, size=80))
    events = _events_at(market, *range(20, 60, 4))
    study = study_events(market, events, horizons=(1,), bootstrap=2000, seed=0)
    row = study.tests.iloc[0]
    assert row["n"] == 10
    assert row["p"] > 0.05
    assert row["ci_low"] < 0 < row["ci_high"]
    assert row["stars"] == ""


def test_cluster_by_day_collapses_same_day_events():
    returns = np.zeros(6)
    returns[0], returns[1] = 0.01, 0.03
    market = _market_from_returns(returns, freq="12h")
    events = _events_at(market, 0, 1)              # 两根 bar 同一天
    assert events["ts"].dt.normalize().nunique() == 1

    clustered = study_events(market, events, horizons=(1,), cluster="day", bootstrap=0)
    row = clustered.tests.iloc[0]
    assert row["n"] == 2 and row["n_days"] == 1 and row["n_obs"] == 1
    assert np.isnan(row["p"])                      # 聚类后只剩 1 个观测, 不硬算

    iid = study_events(market, events, horizons=(1,), cluster="none", bootstrap=0)
    assert iid.tests.iloc[0]["n_obs"] == 2
    assert np.isfinite(iid.tests.iloc[0]["p"])


def test_tests_table_formats_numbers():
    market, events, _ = _jump_market()
    study = study_events(market, events, horizons=(1,), bootstrap=200, seed=1)
    table = study.tests_table()
    assert table.iloc[0]["mean"].endswith("%")
    assert table.iloc[0]["stars"] == "***"
    assert len(table.iloc[0]["p"].split(".")[-1]) == 4


def test_bad_arguments_are_rejected():
    market, events, _ = _jump_market()
    with pytest.raises(ValueError, match="test_on"):
        study_events(market, events, test_on="nope")
    with pytest.raises(ValueError, match="cluster"):
        study_events(market, events, cluster="ticker")
    with pytest.raises(ValueError, match="alpha"):
        study_events(market, events, alpha=1.5)
