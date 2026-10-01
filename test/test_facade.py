"""EventBackTest 门面: 构造覆盖、run / study / compare / sweep / walk_forward。"""
from __future__ import annotations

import numpy as np
import pytest
from helpers import make_market

from event_backtest import EventBackTest, MarketConfig


def _market(bars: int = 120, tickers=("AAA",)):
    rng = np.random.default_rng(0)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.01, size=(bars, len(tickers))), axis=0)
    volume = np.ones((bars, len(tickers)))
    volume[::10] *= 5.0
    return make_market(close, volume=volume, tickers=list(tickers))


def test_from_market_runs_and_reports():
    data = _market()
    backtest = EventBackTest.from_market(data, market_name="us", initial_cash=1000.0,
                                         fees={}, strategy="volume_breakout")
    assert backtest.name == "us" and backtest.benchmark is None
    assert backtest.load_market() is data          # 已加载的行情直接用, 不碰数据库

    run = backtest.run()
    assert run.market == "us" and run.strategy == "volume_breakout"
    assert run.result.meta["market"] == "us"
    assert run.summary()["bars"] == 120
    assert len(run.sections()) > 0


def test_config_overrides_apply():
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0, benchmark="AAA")
    backtest = EventBackTest(cfg, initial_cash=5000.0, universe=["BBB"],
                             strategy="buy_hold", market_data=_market())
    assert backtest.config.initial_cash == 5000.0
    assert backtest.config.tickers == ("BBB",)
    assert backtest.benchmark == "AAA"
    assert backtest.run().result.meta["initial_cash"] == 5000.0


def test_universe_none_means_everything_in_db():
    cfg = MarketConfig(market="us", db_path="unused.db", tickers=("AAA",))
    backtest = EventBackTest(cfg, universe=None, market_data=_market())
    assert backtest.config.tickers is None          # 显式 None = 库里全部证券


def test_missing_strategy_is_clear():
    with pytest.raises(ValueError, match="没有指定策略"):
        EventBackTest.from_market(_market()).run()


def test_study_and_compare():
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    backtest = EventBackTest(cfg, strategy="volume_breakout", market_data=_market())
    study = backtest.study(bootstrap=100)
    assert len(study.tests) == 16
    assert set(study.tests.columns) >= {"horizon", "p", "ci_low", "stars"}

    runs = backtest.compare({"base": {}, "big": {"initial_cash": 2000.0}})
    assert set(runs) == {"base", "big"}
    assert runs["big"].result.meta["initial_cash"] == 2000.0
    assert runs["big"].market == "us"


def test_compare_requires_strategy_per_scenario():
    cfg = MarketConfig(market="us", db_path="unused.db")
    backtest = EventBackTest(cfg, strategy=None, market_data=_market())
    with pytest.raises(ValueError, match="没有策略"):
        backtest.compare({"a": {}})


def test_sweep_and_walk_forward_need_config():
    plain = EventBackTest.from_market(_market(), strategy="volume_breakout")
    with pytest.raises(ValueError, match="市场配置"):
        plain.sweep({"exit.hold_bars": [4, 8]})
    with pytest.raises(ValueError, match="市场配置"):
        plain.walk_forward()


def test_sweep_and_walk_forward_with_config():
    cfg = MarketConfig(market="us", db_path="unused.db", initial_cash=1000.0)
    market = _market(bars=200)
    backtest = EventBackTest(cfg, strategy="volume_breakout")

    outcome = backtest.sweep({"exit.hold_bars": [4, 8]}, market=market)
    assert len(outcome.table) == 2
    assert outcome.best_params["exit.hold_bars"] in (4, 8)

    folds = backtest.walk_forward(market=market)
    assert len(folds.folds) == 2 and len(folds.oos_nav) > 0
