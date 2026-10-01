"""抓取 A 股数据写入 data/ashare.db(参数区在下面, 直接改或走命令行)。

用 sources.cn(BaoStock) 抓、liudb.ashare 清洗并写库。默认抓三样:
    1. 交易日历: [start, end] 整段覆盖;
    2. 证券基本资料: universe 里的全部代码;
    3. 日线(含交易状态: 前收 / 成交额 / 停牌 / ST): [start, end]。

量级控制: 默认最近 10 年、只抓 setting/cn.yaml 的 ETF 池 + 下面 STOCKS 里那一小组
股票。分钟线不在这里(数据量大, 且 BaoStock 收盘后才完整)。

重要: BaoStock 的 **ETF 日线大多只有 2026 年之后**的, 所以 ETF 拿不到 10 年历史;
要 10 年就靠 STOCKS 那一组。想多抓就在 STOCKS 里加代码, 重跑即可(按主键覆盖, 不会重复)。

用法:
    uv run python scripts/update_ashare.py --dry-run     # 只看计划, 不联网不写库
    uv run python scripts/update_ashare.py               # 实际抓取
    uv run python scripts/update_ashare.py --years 10 --start 2016-10-01
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import liudb.ashare
import pandas as pd
from loguru import logger
from sources import cn

from event_backtest import load_config

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 参数区(改这里)
DB = REPO / "data" / "ashare.db"   # 命令行 --db 可覆盖
YEARS = 10                         # 默认回填最近多少年(命令行 --years / --start 可覆盖)
INCLUDE_STOCKS = True              # False = 只抓 ETF

# ETF 池直接读 setting/cn.yaml 的 universe, 避免两处写法漂移
ETF_TICKERS: tuple[str, ...] = tuple(load_config("cn").tickers or ())

# 10 年历史的股票池(先少抓一些): 大盘 + 消费 + 金融 + 制造 + 新能源
STOCKS: tuple[str, ...] = (
    "600519.SH",   # 贵州茅台
    "601318.SH",   # 中国平安
    "600036.SH",   # 招商银行
    "000001.SZ",   # 平安银行
    "000333.SZ",   # 美的集团
    "000651.SZ",   # 格力电器
    "002415.SZ",   # 海康威视
    "002594.SZ",   # 比亚迪
    "600900.SH",   # 长江电力
    "601166.SH",   # 兴业银行
)
# ----------------------------------------------------------------


def default_start(years: int, today: date | None = None) -> date:
    """最近 years 年的起点(同月同日), 2 月 29 日自动退到 2 月 28 日。"""
    today = today or date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数; 全部可选, 默认用参数区的值。"""
    parser = argparse.ArgumentParser(description="抓取 A 股数据写入 ashare.db")
    parser.add_argument("--db", type=Path, default=DB, help=f"数据库路径, 默认 {DB}")
    parser.add_argument("--years", type=int, default=YEARS, help="回填最近多少年")
    parser.add_argument("--start", type=pd.Timestamp, default=None, help="起始日期, 覆盖 --years")
    parser.add_argument("--end", type=pd.Timestamp, default=None, help="结束日期, 默认今天")
    parser.add_argument("--no-stocks", action="store_true", help="只抓 ETF 池")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划, 不联网不写库")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """抓交易日历 + 证券资料 + 日线, 写入 ashare.db; 返回进程退出码。"""
    args = parse_args(argv)
    start = args.start.date() if args.start is not None else default_start(args.years)
    end = args.end.date() if args.end is not None else date.today()
    tickers = list(ETF_TICKERS) + (list(STOCKS) if INCLUDE_STOCKS and not args.no_stocks else [])

    logger.info(f"数据库={args.db}")
    logger.info(f"区间={start} ~ {end}; 证券={len(tickers)} 只"
                f"(ETF {len(ETF_TICKERS)} + 股票 {len(tickers) - len(ETF_TICKERS)})")
    if args.dry_run:
        logger.info(f"[dry-run] 计划抓取 {tickers}, 不联网不写库")
        return 0

    args.db.parent.mkdir(parents=True, exist_ok=True)
    liudb.ashare.init_schema(path=str(args.db))

    calendar = cn.get_cn_trade_calendar(start, end)
    liudb.ashare.save_trade_calendar(calendar, path=str(args.db))
    logger.info(f"交易日历: {len(calendar)} 天 -> {args.db.name}")

    basic = cn.get_cn_stock_basic(tickers)
    liudb.ashare.save_stock_basic(basic, path=str(args.db))
    logger.info(f"证券资料: {len(basic)} 只")

    bars = cn.get_cn_daily_bars(tickers, start, end)
    liudb.ashare.save_prices(bars, path=str(args.db))
    logger.info(f"日线: {len(bars)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
