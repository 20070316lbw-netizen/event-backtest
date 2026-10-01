"""CLI: list 命令。"""
from __future__ import annotations

from event_backtest.cli import main


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "配置" in out and "策略" in out and "因子" in out
    assert "volume_breakout" in out
