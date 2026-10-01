"""事件研究的统计工具: t 检验 p 值、bootstrap 区间与重抽检验。"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from event_backtest.evaluation.stats import (
    bootstrap_mean,
    mean_t_test,
    stars,
    two_sided_t_p,
)


def test_two_sided_t_p_matches_textbook_critical_values():
    # t_{0.025, df}: 双侧 p 正好是 0.05
    for df, critical in ((1, 12.706204736), (10, 2.228138852), (30, 2.042272456)):
        assert two_sided_t_p(critical, df) == pytest.approx(0.05, abs=1e-8)
    assert two_sided_t_p(0.0, 5) == pytest.approx(1.0)
    assert two_sided_t_p(1.0, 1) == pytest.approx(0.5)      # df=1 是柯西分布
    assert np.isnan(two_sided_t_p(np.nan, 5))
    assert np.isnan(two_sided_t_p(1.0, 0))


def test_mean_t_test_matches_scipy():
    rng = np.random.default_rng(11)
    data = rng.normal(0.01, 0.02, size=50)
    mean, t, p, n = mean_t_test(data)
    reference = sps.ttest_1samp(data, 0.0)
    assert n == 50
    assert mean == pytest.approx(float(data.mean()))
    assert t == pytest.approx(float(reference.statistic))
    assert p == pytest.approx(float(reference.pvalue))


def test_mean_t_test_guards():
    mean, t, p, n = mean_t_test([])
    assert n == 0 and np.isnan(t) and np.isnan(p)

    mean, t, p, n = mean_t_test([0.02])
    assert n == 1 and mean == pytest.approx(0.02) and np.isnan(t) and np.isnan(p)

    mean, t, p, n = mean_t_test([0.02, 0.02, 0.02])          # 标准差为 0
    assert n == 3 and np.isnan(t) and np.isnan(p)

    mean, t, p, n = mean_t_test([0.01, np.nan, 0.03, np.inf])   # 非有限被丢掉
    assert n == 2 and mean == pytest.approx(0.02) and np.isfinite(p)


def test_bootstrap_is_deterministic_and_brackets_sample_mean():
    data = [0.004, 0.006, 0.005, 0.0035, 0.0065, 0.0045, 0.0055, 0.0048]
    first = bootstrap_mean(data, n_boot=2000, seed=7)
    assert first == bootstrap_mean(data, n_boot=2000, seed=7)     # 同种子可复现
    low, high, p = first
    assert low < float(np.mean(data)) < high
    assert 0.0 < p <= 1.0


def test_bootstrap_detects_strong_effect_and_ignores_noise():
    strong = [0.02, 0.021, 0.019, 0.022, 0.018, 0.02, 0.021, 0.019]
    low, high, p = bootstrap_mean(strong, n_boot=2000, seed=0)
    assert p < 0.01 and low > 0 and high > 0

    noise = [0.01, -0.02, 0.03, -0.01, 0.005, -0.015, 0.02, -0.005]
    _, _, p_noise = bootstrap_mean(noise, n_boot=2000, seed=0)
    assert p_noise > 0.05


def test_bootstrap_guards():
    assert all(np.isnan(value) for value in bootstrap_mean([0.1]))
    assert all(np.isnan(value) for value in bootstrap_mean([0.1, 0.2], n_boot=0))
    assert all(np.isnan(value) for value in bootstrap_mean([]))


def test_stars_thresholds():
    assert stars(0.009) == "***"
    assert stars(0.049) == "**"
    assert stars(0.099) == "*"
    assert stars(0.1) == ""
    assert stars(0.5) == ""
    assert stars(np.nan) == ""
