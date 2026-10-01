"""事件研究用的基础统计: t 检验 p 值、bootstrap 区间与检验。

t 分布用 scipy.stats(不自己实现不完全 Beta); bootstrap 用 numpy 的 Generator,
固定种子结果可复现。

约定:
    - 单样本 t 检验的零假设是"均值 = 0"; p 值是双侧的, 等价于
      scipy.stats.ttest_1samp(x, 0.0) 的 pvalue。
    - bootstrap 的 p 值把样本平移到 H0(均值 0)下再有放回重抽, 属于零假设下的重抽检验;
      区间是未平移样本的百分位区间。用 (1 + 超过次数) / (n_boot + 1) 避免 p = 0。
    - 样本少于 2 个、或标准差为 0 时返回 NaN, 不硬算。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as _stats

__all__ = ["bootstrap_mean", "mean_t_test", "stars", "two_sided_t_p"]

# bootstrap 每次处理多少个重抽样本, 避免 (n_boot, n) 一次性吃太多内存
_CHUNK = 256


def two_sided_t_p(t: float, df: float) -> float:
    """t 分布的双侧 p 值 P(|T_df| >= |t|)。

    Args:
        t: t 统计量。
        df: 自由度(必须 > 0)。

    Returns:
        双侧 p 值; t / df 非有限或 df <= 0 时返回 NaN。

    Example:
        >>> round(two_sided_t_p(2.228138852, 10), 4)   # t_{0.025, 10}
        0.05
    """
    if not np.isfinite(t) or not np.isfinite(df) or df <= 0:
        return math.nan
    return float(2.0 * _stats.t.sf(abs(float(t)), float(df)))


def mean_t_test(values: Sequence[float]) -> tuple[float, float, float, int]:
    """单样本 t 检验(零假设: 均值 = 0)。

    Args:
        values: 观测值; 非有限的会被丢掉。

    Returns:
        (均值, t 统计量, 双侧 p 值, 有效样本数); 样本 < 2 或标准差为 0 时
        t / p 是 NaN。
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = int(data.size)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(data.mean())
    if n < 2:
        return mean, math.nan, math.nan, n
    sd = float(data.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return mean, math.nan, math.nan, n
    statistic, p_value = _stats.ttest_1samp(data, 0.0)
    return mean, float(statistic), float(p_value), n


def bootstrap_mean(values: Sequence[float], *, n_boot: int = 1000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float, float]:
    """均值的 bootstrap 百分位区间 + 零假设下的重抽检验 p 值。

    做法: 有放回重抽 n 个观测 B 次; 区间取重抽均值的 [alpha/2, 1-alpha/2] 分位;
    p 值先把样本平移成均值 0, 再重抽, 用 |均值*| >= |样本均值| 的比例(加一平滑)。

    Args:
        values: 观测值; 非有限的会被丢掉。
        n_boot: 重抽次数; <= 0 时全部返回 NaN。
        alpha: 显著性水平(区间是 1 - alpha 的百分位区间)。
        seed: 随机种子, 固定它结果可复现。

    Returns:
        (ci_low, ci_high, p_value); 样本 < 2 或 n_boot <= 0 时三个都是 NaN。
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = int(data.size)
    if n < 2 or n_boot <= 0:
        return math.nan, math.nan, math.nan

    observed = float(data.mean())
    centered = data - observed            # H0: 均值 = 0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    null_means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, _CHUNK):
        stop = min(start + _CHUNK, n_boot)
        picks = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = data[picks].mean(axis=1)
        null_means[start:stop] = centered[picks].mean(axis=1)

    ci_low, ci_high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    exceeded = int(np.sum(np.abs(null_means) >= abs(observed)))
    p_value = (1.0 + exceeded) / (n_boot + 1.0)
    return float(ci_low), float(ci_high), float(p_value)


def stars(p_value: float) -> str:
    """p 值的显著性星号: *** < 0.01, ** < 0.05, * < 0.1, 否则空串。"""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""
