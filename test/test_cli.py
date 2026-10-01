"""CLI: list 命令与 --json 报告。"""
from __future__ import annotations

import json
from types import SimpleNamespace

from helpers import make_market

from event_backtest import Run, Strategy, run
from event_backtest.cli import _report_json, main
from event_backtest.fees import FeeSchedule
from event_backtest.rules import MarketRules


class _Noop(Strategy):
    def handle_data(self, ctx):
        pass


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "配置" in out and "策略" in out and "因子" in out
    assert "volume_breakout" in out


def test_json_report_is_serializable():
    market = make_market([[10.0], [10.2], [10.5]])
    result = run(_Noop(), market, initial_cash=1000.0, fees=FeeSchedule(),
                 rules=MarketRules())
    result.meta.update({"market": "us"})
    payload = _report_json(Run(result, strategy="noop"), None, SimpleNamespace(output=None))

    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["strategy"] == "noop" and restored["market"] == "us"
    assert restored["output"] is None and restored["event_study"] is None
    assert restored["metrics"]["total_return"] == 0.0
