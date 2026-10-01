# event-backtest

[![CI](https://github.com/20070316lbw-netizen/event-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/event-backtest/actions/workflows/ci.yml)

轻量事件驱动回测框架, 同时支持 **A 股**与**美股**。设计参考
[zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) 的事件循环与订单
模型, 以及 minievent 的对齐数组数据层、市场规则与声明式 YAML 策略约定。

- **数据**: 从 liudb(`ashare.db` / `sp500.db`)读成对齐的 `(T, N)` 数组。
- **策略**: 声明式 YAML(因子 + signal + exit + sizing), 不写代码。
- **因子**: 自包含求值器, YAML 格式与 minievent 一致, 不依赖 minibacktest。
- **撮合**: 第 t 根收盘算信号, 第 t+1 根成交; A 股 T+1 / 涨跌停 / 整手 / 费用都在这一层。
- **滑点**: 固定价差 / 成交量比例。
- **订单簿记**: 每一单都有状态与原因(拒单 / 期末未成交 / 部分成交), 信号能追到成交。
- **评估**: 组合指标、回合交易统计、事件研究(带 t 检验与 bootstrap 区间)。
- **输出**: parquet / csv + `run.json` 运行清单 + 自包含交互式 HTML。

## 环境

```bash
uv sync           # 依赖锁定在 uv.lock
uv run pytest     # 测试(含 docstring 里的示例)
uv run ruff check .
```

## 快速开始: 一个类跑一圈

给调用者的入口就一个类: 超参数式构造 + 有补全的方法。

```python
from event_backtest import EventBackTest, FixedSlippage

backtest = EventBackTest(
    market="us",                 # "cn" / "us" / yaml 路径 / MarketConfig
    start="2016-10-01", end="2026-09-30",
    universe=None,               # None = 库里全部证券; 不传 = 用配置里的
    initial_cash=1_000_000, fill_price="open",
    strategy="volume_breakout",
)
run = backtest.run()                    # -> Run(净值 / 成交 / 委托 / 上下文)
run.metrics().sharpe                    # 类型化指标, 编辑器能补全
run.summary()["total_return"]           # 也可以当 dict 用(与终端表同源)
run.execution()["execution_rate"]       # 执行率 / 拒单 / 换手
run.print()                             # 终端分组表
run.study(bootstrap=2000)               # 事件研究 + 显著性检验
run.save("outputs/us_full", html=True)  # parquet / csv / run.json + tearsheet.html

backtest.walk_forward()                                   # 样本外
backtest.sweep({"exit.hold_bars": [8, 16, 32]})           # 扫参(样本内)
runs = backtest.compare({"基准": {}, "滑点0.05": {"slippage": FixedSlippage(0.05)}})
```

- `Run`: 结果 + 配置 + 口径绑在一起, `summary()` / `metrics()` / `html()` / `save()`
  都不用再重复传 market / benchmark。
- `run.json`: 每次落盘写一份清单(包版本 / git commit / 配置 / 区间 / 费率 / 滑点 /
  指标快照 / 文件列表), 两次运行才可比、可追溯。

## 命令行

```bash
event-backtest list                     # 列出配置 / 策略 / 因子
event-backtest show factor momentum     # 打印某个因子 yaml(带注释)与解析结果
event-backtest show strategy volume_breakout

# A 股 30 分钟, 声明式策略, 输出结果与事件研究
event-backtest run --config cn --db data/ashare.db --freq 30 --strategy volume_breakout --start 2026-01-05 --end 2026-09-24 --event-study --output outputs/volume_breakout

# 美股日线 + 自包含 HTML
event-backtest run --config us --strategy volume_breakout --html --start 2024-01-01 --end 2024-12-31 --output outputs/us_demo

# A 股 10 年日线(股票池; 数据先用 scripts/update_ashare.py 抓进 data/ashare.db)
event-backtest run --config cn_stock --strategy volume_breakout --start 2016-10-10 --end 2026-09-30

# 机器可读输出(stdout 只有 JSON, 日志走 stderr), 方便脚本 / agent 调用
event-backtest run --config us --strategy volume_breakout --json
```

| 参数 | 作用 |
|---|---|
| `--config` | setting/<名字>.yaml 或 yaml 路径 |
| `--strategy` | 策略 yaml 名 / yaml 路径 / 内置代码策略(buy_hold / sma_cross) |
| `--start` `--end` `--freq` `--db` | 覆盖配置里的区间 / 周期 / 数据库 |
| `--output` | 结果输出目录(见下) |
| `--event-study` `--bootstrap N` | 事件研究 + 显著性(bootstrap 重抽次数, 0 = 不算) |
| `--html` | 额外写自包含 `tearsheet.html`(需配合 `--output`) |
| `--json` | stdout 只输出 JSON |
| `--fast` `--slow` | 内置 sma_cross 的窗口 |

## 输出文件

```
nav.parquet       逐 bar 净值             fills.parquet     成交明细
orders.parquet    委托明细(含被拒)        trades.parquet    回合交易(含期末未平仓)
performance.csv   组合指标                trade_stats.csv   回合统计
execution.csv     执行率 / 换手 / 费用占比 run.json          运行清单
event_paths.parquet / event_summary.csv / event_tests.csv    (--event-study)
tearsheet.html                                               (--html)
```

- **委托明细** `orders`: `status`(open / filled / rejected / cancelled)、没成交时的
  `reason`(如"涨停买不进"/"现金不足"/"数量不足一手"/"回测结束未成交")、`tag`
  (`enter` / `exit:stop_loss` / `exit:take_profit` / `exit:hold_bars`)与
  `event_id`(触发它的信号事件); 部分成交看 `amount` vs `filled`。
- **成交明细** `fills`: 同样带 `event_id`, 所以"这笔成交是哪个信号下的"能直接查。
- **回合交易** `trades`: `open=True` 表示回测结束时还没平仓(用最后收盘价盯市),
  `entry_event` 是开仓信号; 统计里 `trades` 只数已平仓, `open_trades` 单独计数。
- 终端报告在有拒单时会多打一张"拒单原因"表。

## Python: 组件没有被藏起来

门面只是把组件按"一次完整运行"组装好, 要细粒度控制照旧可以单独用:

```python
from event_backtest import (load_config, resolve_strategy, run, study_events,
                            run_walk_forward, search, save_run, save_comparison,
                            slice_dates, load_market_cached)

cfg = load_config("cn")
market = cfg.cached_load(start="2026-01-05", end="2026-09-24", freq="30")  # 复用同一份
strategy, spec = resolve_strategy("volume_breakout")
result = run(strategy, market, initial_cash=cfg.initial_cash, fees=cfg.resolved_fees(),
             fill_price=cfg.fill_price, slippage=cfg.slippage, rules=cfg.resolved_rules())
study = study_events(market, spec.events(spec.compute(market), market), strategy_name=spec.name)
study.summary        # 每个 horizon 的均值 / 基线 / 超额 / 胜率
study.tests_table()  # 显著性: 均值 / t / p / bootstrap 区间 / 星号
save_run("outputs/one", result, market="cn", study=study, html=True)

covid = slice_dates(market, "2020-02-01", "2020-04-30")     # 只看某一段
```

主要组件: `load_config` / `MarketConfig`(`.load` / `.cached_load` / `.run` / `.describe`) /
`resolve_strategy` / `engine.run` / `Run` / `save_run` / `save_comparison` /
`study_events` / `run_walk_forward` / `search` / `load_market_cached` / `slice_dates`。

> `load_market_cached` 返回的是**同一份对象**(按参数组合 LRU, 默认存 2 份), 别就地改数组。

## 市场配置

`src/event_backtest/setting/<market>.yaml`, 段为
`market / data / universe / benchmark / account / execution / fees / rules`, 加载时校验字段:

```yaml
market: us
data:
  db_path: ../../../data/sp500_10y.db   # 相对本文件所在目录; 不存在时直接报错
  freq: null                            # A 股分钟线: "5"/"15"/"30"/"60"
universe: [AAPL, MSFT]                  # 不写 = 库里全部证券
benchmark: null                         # 基准代码; 给了就在报告里算超额 / Beta
account: {initial_cash: 100000}
execution:
  fill_price: open                      # open / vwap(vwap 需要成交额数据)
  slippage: {type: volume_share, volume_limit: 0.025, price_impact: 0.1}
  # 或 {type: fixed, spread: 0.01}
fees:
  stock: {commission: 0.0, min_commission: 0.0, stamp_duty: 0.0, transfer_fee: 0.0}
```

现成的配置: `cn`(A 股 ETF)、`us`(美股日线)、`cn_stock`(A 股 10 年股票池)。

## 策略 YAML 速查

`src/event_backtest/strategy/<name>.yaml` —— 五个段, 每段都可选:

| 段 | 字段 | 说明 |
|---|---|---|
| `name` | — | 策略名(报告 / event_id 里用) |
| `walk_forward` | `folds` `start` `train_days` `test_days` `embargo_days` | 样本外切分; 不写就不做 |
| `factors` | `- <因子名>: {参数}` | 启用哪些因子; 可写 `as: 别名` 给输出改名 |
| `signal` | `name` `trigger` `all` / `any` | 条件列表; 每条 `{factor, operation, value}` |
| `exit` | `hold_bars` `take_profit` `stop_loss` | 出场规则; 都不写 = 持有到结束 |
| `sizing` | `percent` `max_positions` | 单笔仓位与持仓上限 |

- `trigger`: `enter`(条件由不满足变满足时触发一次, 默认) / `every_bar`(每根满足都触发)。
- signal 的可用运算只有 6 个比较: `greater_than` / `greater_than_or_equal` /
  `less_than` / `less_than_or_equal` / `equal` / `not_equal`; 因子为 NaN 一律算不满足。
- **sizing 语义**: 写了 `percent` 就用它; 否则设了 `max_positions` 就按
  `1 / max_positions` 等权分摊(推荐); 都没有才退到全池等权 `1 / N`(池子一大每笔
  仓位就趋近于 0)。
- 时序: 第 t 根收盘后用截至第 t 根的数据算信号, 第 t+1 根成交 —— **不要自己 `shift(1)`**。

```yaml
name: 放量突破
walk_forward: {folds: 2, train_days: 60, test_days: 30, embargo_days: 5}
factors:
  - relative_volume: {window: 16}
  - momentum: {window: 8}
signal:
  name: volume_breakout
  trigger: enter
  all:
    - {factor: relative_volume, operation: greater_than, value: 2}
    - {factor: momentum, operation: greater_than, value: 0}
exit: {hold_bars: 16, take_profit: 0.06, stop_loss: 0.04}
sizing: {max_positions: 3}
```

现成的策略: `volume_breakout`(放量突破)、`momentum_only`(纯动量)。两个文件都写了逐字段
注释, 可以照抄改; `event-backtest show strategy <名字>` 会把注释一起打出来。

## 因子 YAML 速查

`src/event_backtest/factor/<name>.yaml`: 一张"行情 -> 一列数"的配方, 按 `steps` 顺序算,
`output` 指向结果。

| 字段 | 说明 |
|---|---|
| `name` / `description` | 因子名与说明 |
| `parameters` | 参数名列表; 策略里传的值从这里取 |
| `steps` | 步骤列表, 每步 `id` + `operation` + 该运算的操作数 |
| `output` | 最终输出哪个 step 的 id(或行情字段) |

**操作数三种写法**(每个运算的操作数字段名见下表):

```yaml
{variable: volume}       # 行情字段或前面某个 step 的 id
{parameter: window}      # 策略传进来的参数(要在 parameters 里声明)
{constant: 1.0}          # 常数
```

**行情字段**(`{variable: ...}` 可直接用): `price` / `close`(后复权收盘)、`open` /
`high` / `low`(后复权)、`volume`。

**14 个运算**:

| operation | 操作数 | 含义 |
|---|---|---|
| `shift` | `input` `periods` | 序列整体后移 periods 根 |
| `rolling_mean` | `input` `window` | 过去 window 根均值 |
| `rolling_std` | `input` `window` | 过去 window 根标准差 |
| `rolling_min` / `rolling_max` | `input` `window` | 过去 window 根最小 / 最大 |
| `rolling_sum` | `input` `window` | 过去 window 根求和 |
| `rolling_corr` | `left` `right` `window` | 两个序列的滚动相关系数 |
| `cross_section_rank` | `input` | 同一根 bar 上按证券排名(0~1) |
| `add` / `subtract` / `multiply` | `left` `right` | 逐元素加减乘 |
| `divide` | `numerator` `denominator` | 逐元素除; 分母为 0 记 NaN |
| `elementwise_max` / `elementwise_min` | `left` `right` | 逐元素取大 / 小 |

```yaml
name: relative_volume
description: 当前 bar 成交量除以之前 window 根 bar 的平均成交量
parameters: [window]
steps:
  - {id: previous_volume, operation: shift, input: {variable: volume}, periods: {constant: 1}}
  - {id: average_previous_volume, operation: rolling_mean, input: {variable: previous_volume}, window: {parameter: window}}
  - {id: result, operation: divide, numerator: {variable: volume}, denominator: {variable: average_previous_volume}}
output: result
```

`event-backtest show factor <名字>` 会把 yaml 原文(带注释)和解析出来的每一步打出来。
写错字段 / 运算名 / 引用不存在的 id 都在加载时报错, 不会拖到回测中途。

## 事件研究与显著性

`--event-study`(或 `Run.study()`)除了描述性汇总, 还会打印一张检验表, 并存
`event_tests.csv`(列: `horizon, n, n_obs, n_days, mean, t, p, ci_low, ci_high,
p_boot, stars`)。口径:

- **检验对象**默认是**超额收益**(事件收益 - 同时段无条件基线), 问的是"信号相对同时段
  基准有没有增量"; `test_on="ret"` 则检验原始收益(多半只是在检验市场漂移)。
- **按交易日聚类**(默认): 同一天、尤其同一根 bar 触发的多个事件并不独立, 直接对事件
  做 t 检验会高估显著性。先把同一天的事件压成一个日均值, 再对日均值做单样本 t 检验;
  `cluster="none"` 才是事件级 iid 检验(仅作对照)。
- **bootstrap**: 对日均值有放回重抽, 区间取重抽均值的百分位; p 值把样本平移到均值 0
  下再重抽(零假设下的重抽检验)。用 `seed` 固定随机性, 默认 1000 次。
- **样本太少不算**: 聚类后不足 2 个观测、或标准差为 0 时, t / p / 区间都是 NaN。
- **没有多重比较校正**: 每个 horizon 各检各的, 16 个 horizon 一起看时单看某一行的
  "显著"要打折扣; 要更严格自己按 horizon 数做 Bonferroni / BH。
- 统计量用 `scipy.stats`(t 检验); bootstrap 用 numpy 自己实现, 便于固定种子复现。

## Walk-forward(样本外)

在策略 yaml 里加一段(不写就不做; `start` 省略 = 用数据第一个交易日):

```yaml
walk_forward:
  folds: 2
  start: 2026-01-05   # 可省
  train_days: 60
  test_days: 30
  embargo_days: 5
```

跑法: `EventBackTest.walk_forward()` 或 `scripts/run_walk_forward.py`。它把历史切成若干折,
每折 训练 -> 隔离 -> 测试, 各折测试段首尾相接, 拼成一条样本外净值。

- **训练步骤是一个可选的选择器** `select(spec, train_market) -> spec`。
  `scripts/run_walk_forward.py` 里填上 `GRID` 就会在每折**训练段**扫网格、挑最优参数再用于
  **测试段**(这才是参数优化); `GRID` 留空则每折沿用同一份策略, 只是分折样本外评估。
- 选参只在训练段评估, 依据由 `SELECT_BY` 决定(默认 sharpe); 每折选到的参数会打印出来,
  方便看参数稳不稳。
- 正确性: 选择器只看到训练段之前的数据(测试段不外泄); 各折测试段不重叠、之间隔
  `embargo_days`; 每折用同一套费率 / 滑点, 样本外净值按折内收益连乘拼接。

## 参数扫描

`EventBackTest.sweep({点路径: 候选取值})` 或 `scripts/sweep.py`。点路径就是策略 yaml 的
层级, `[i]` 是列表下标, **中间缺的段会自动补出来**:

```python
outcome = backtest.sweep({
    "factors[0].relative_volume.window": [8, 16, 32],
    "signal.all[0].value": [1.5, 2.0, 3.0],
    "sizing.percent": [0.1, 0.2],          # 策略里没写 sizing 段也能扫
}, metric="calmar")                        # 任意报告指标都能当依据
outcome.table / outcome.best / outcome.best_params
```

> 注意: `sweep` 是在**同一段数据**上挑最好的一组, 属于样本内, 不能直接当可交易结论;
> 要样本外验证, 用 `GRID` 当 walk-forward 的 `select`。

## 场景对比

```python
runs = backtest.compare({"基准": {}, "滑点0.05": {"slippage": FixedSlippage(0.05)}})
save_comparison("outputs/compare", runs, title="场景对比")
# -> comparison.csv / comparison.png / comparison.html + 每个场景一份 <场景名>.html
```

场景里的键: `strategy`(这个场景用哪个策略) + 其余任何 `MarketConfig` 字段
(`slippage` / `initial_cash` / `fees` / `rules` ... 按 dataclass 字段名给)。
`scripts/run_backtest.py` 就是这个套路的完整例子。

## 交互式 HTML 报告

`plot_tearsheet_html` / `plot_comparison_html` / `plot_walk_forward_html` 返回
`HtmlReport`, `.save(path)` 落成**单个自包含 HTML**: 数据内联 JSON, 样式与脚本内联,
不引用 CDN, 断网 / 直接发文件都能打开。

- 内容: KPI 卡 + 净值(带基准线) + 回撤 + 月度收益 + 回合收益分布 + 指标分组表 +
  回合交易表(含未平仓与触发事件) + 委托明细 / 拒单原因; 对比页是多场景叠加 + 可排序
  指标表; walk-forward 页是样本外净值 + 各折收益 + 折明细。
- 交互: 十字光标 + tooltip、图例点击开关、区间缩放(1月/3月/6月/1年/3年/5年/10年)、
  表格点表头排序、深浅色切换(默认跟随系统)。
- 口径与终端一致: 指标分组复用 `report.result_sections`, 执行 / 拒单复用 `metrics`,
  所以 HTML、终端表和 parquet 三边对得上。

## 切片器

`slice_dates(market, start, end)` 把行情复制一份、只保留指定日期范围(两端含), 用来
单独看某个区间(例如疫情那段); `slice_market(market, i0, i1)` 是按 bar 下标切。

## 已知限制

- **ETF 历史短**: BaoStock 的 ETF 日线大多只有 2026 年之后的, 长区间要用股票池
  (`cn_stock.yaml`); 数据由 `scripts/update_ashare.py` 抓。
- **幸存者偏差**: 数据里没有退市成员, 长区间回测天然偏乐观。
- **事件研究**不做多重比较校正(见上)。
- **扫参是样本内**; 要样本外请走 walk-forward。
- **`load_market_cached` 返回同一份对象**, 就地改数组会影响后续调用。

## 目录

```
src/event_backtest/
  market.py       MarketData + liudb 加载 (cn / us) + 切片 + 行情缓存
  rules.py        市场规则 (CNMarketRules / USMarketRules)
  fees.py         费率
  slippage.py     固定 / 成交量比例滑点
  factor/         因子 YAML + 自包含求值器
  signal/         条件 -> 事件
  strategy/       回调基类 + 声明式 YAML 策略 + walk_forward 解析
  broker.py       订单 / 组合 / 撮合
  engine.py       主循环 / Context / run / BacktestResult
  walkforward.py  折划分 + select 钩子 + 样本外拼接
  sweep.py        点路径网格 + 搜索 + select 工厂
  evaluation/     事件研究 + 显著性统计
  metrics.py      组合与交易统计(含未平仓盯市)
  report.py       rich 中文表(单场景 / 对比 / walk-forward)
  figure/         PNG(matplotlib) + 自包含交互式 HTML(零依赖原生 JS/SVG)
  results.py      Run / 类型化指标 / save_run / save_comparison
  facade.py       EventBackTest(给调用者的超参数式入口)
  benchmark.py    基准买入持有净值
  config.py       市场配置
  setting/        cn.yaml / us.yaml / cn_stock.yaml
  cli.py          命令行(list / show / run)
scripts/          run_backtest.py / run_walk_forward.py / sweep.py / update_ashare.py
```

## 约定

- 因子以 T+0 为前提书写: 第 t 根用截至第 t 根收盘的数据, 不自己 `shift(1)`; 防未来
  函数由引擎负责(第 t 根算信号, 第 t+1 根成交)。
- A 股: T+1(新交易日首根 bar 解禁)、涨跌停(一字涨停买不进 / 一字跌停卖不出)、
  100 股整手(目标仓位单也按整手委托)、按 `sec_type` 的佣金 / 印花税 / 过户费。
- 美股: T+0、无涨跌停、1 股整手。
- 事件研究的基线是同时段无条件均值, 只作描述; 显著性检验按交易日聚类, 不做多重比较
  校正, 所以"显著"要结合 horizon 数量一起看。

## 变更记录

见 [CHANGELOG.md](CHANGELOG.md)。
