"""图表: tearsheet(单场景)、对比图(多场景)与 walk-forward 样本外净值。

两套输出并存, 数据口径一致:
    - `plot_*`: matplotlib Figure, 存 PNG(见 tearsheet.py);
    - `plot_*_html`: 自包含交互式 HTML, 存单文件(见 html.py)。
"""
from __future__ import annotations

from event_backtest.figure.html import (
    HtmlReport,
    plot_comparison_html,
    plot_tearsheet_html,
    plot_walk_forward_html,
)
from event_backtest.figure.tearsheet import (
    plot_comparison,
    plot_tearsheet,
    plot_walk_forward,
)

__all__ = [
    "HtmlReport",
    "plot_comparison",
    "plot_comparison_html",
    "plot_tearsheet",
    "plot_tearsheet_html",
    "plot_walk_forward",
    "plot_walk_forward_html",
]
