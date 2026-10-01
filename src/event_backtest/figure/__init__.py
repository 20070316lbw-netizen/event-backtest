"""图表: tearsheet(单场景)、对比图(多场景)与 walk-forward 样本外净值。"""
from __future__ import annotations

from event_backtest.figure.tearsheet import (
    plot_comparison,
    plot_tearsheet,
    plot_walk_forward,
)

__all__ = ["plot_comparison", "plot_tearsheet", "plot_walk_forward"]
