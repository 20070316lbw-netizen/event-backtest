"""配置解析: 滑点段与字段校验。"""
from __future__ import annotations

import pytest

from event_backtest.config import ConfigError, load_config
from event_backtest.slippage import FixedSlippage, VolumeShareSlippage


def _write(tmp_path, body: str):
    path = tmp_path / "m.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_fixed_slippage_parsed(tmp_path):
    path = _write(tmp_path, "market: us\ndata: {db_path: x.db}\n"
                             "execution:\n  fill_price: open\n"
                             "  slippage: {type: fixed, spread: 0.02}\n")
    cfg = load_config(path)
    assert isinstance(cfg.slippage, FixedSlippage)
    assert cfg.slippage.spread == pytest.approx(0.02)


def test_volume_share_slippage_parsed(tmp_path):
    path = _write(tmp_path, "market: cn\ndata: {db_path: x.db}\n"
                             "execution:\n  slippage: {type: volume_share, volume_limit: 0.1,"
                             " price_impact: 0.2}\n")
    cfg = load_config(path)
    assert isinstance(cfg.slippage, VolumeShareSlippage)
    assert cfg.slippage.volume_limit == pytest.approx(0.1)


def test_unknown_slippage_type(tmp_path):
    path = _write(tmp_path, "market: us\ndata: {db_path: x.db}\n"
                             "execution:\n  slippage: {type: nope}\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_field_rejected(tmp_path):
    path = _write(tmp_path, "market: us\ndata: {db_path: x.db}\nfoo: 1\n")
    with pytest.raises(ConfigError):
        load_config(path)
