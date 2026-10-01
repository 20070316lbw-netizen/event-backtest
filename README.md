# event-backtest

[![CI](https://github.com/20070316lbw-netizen/event-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/event-backtest/actions/workflows/ci.yml)

轻量事件驱动回测框架, 同时支持 **A 股**与**美股**。设计参考
[zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) 的事件循环与订单
模型, 以及 minievent 的对齐数组数据层、市场规则与声明式 YAML 策略约定。

- **数据**: 直接从 liudb(`ashare.db` / `sp500.db`)读成对齐的 `(T, N)` 数组。
- **策略**: 声明式 YAML(因子 + signal + exit + sizing), 不写代码。
- **因子**: 自包含求值器, YAML 格式与 minievent 一致, 不依赖 minibacktest。
- **撮合**: 第 t 根收盘算信号, 第 t+1 根成交; A 股 T+1 / 涨跌停 / 整手 / 费用都在这一层。
- **滑点**: 固定价差 / 成交量比例。
- **评估**: 组合指标、回合交易统计、事件研究。

## 环境

```bash
uv sync           # 依赖锁定在 uv.lock
uv run pytest     # 测试
uv run ruff check .
```

## 命令行

```bash
event-backtest list

# A 股 30 分钟, 声明式策略, 输出结果与事件研究
event-backtest run --config cn --db data/ashare.db --freq 30 \
  --strategy volume_breakout --start 2026-01-05 --end 2026-09-24 \
  --event-study --output outputs/volume_breakout

# 美股日线
event-backtest run --config us --strategy volume_breakout \
  --start 2024-01-01 --end 2024-12-31

# A 股 10 年日线(股票池; 数据先用 scripts/update_ashare.py 抓进 data/ashare.db)
event-backtest run --config cn_stock --strategy volume_breakout \
  --start 2016-10-10 --end 2026-09-30

# 加 --html: 除了 parquet / csv, 再写一份自包含交互式 tearsheet.html
event-backtest run --config us --strategy volume_breakout --html \
  --start 2024-01-01 --end 2024-12-31 --output outputs/us_demo
```

输出目录含 `nav.parquet` / `fills.parquet` / `orders.parquet` / `trades.parquet` /
`performance.csv` / `trade_stats.csv` / `execution.csv`; 加 `--event-study` 时再写
`event_paths.parquet` / `event_summary.csv` / `event_tests.csv`; 加 `--html` 再写
`tearsheet.html`。

`orders.parquet` 是**委托明细**: 每一单都有 `status`(open / filled / rejected /
cancelled)、没成交时的 `reason`(如"涨停买不进"/"现金不足"/"回测结束未成交")与
`tag`(`enter` / `exit:stop_loss` / `exit:take_profit` / `exit:hold_bars`),
部分成交看 `amount` vs `filled`。这样"某个信号为什么没成交 / 为什么平仓"能直接查;
`execution.csv`(即 `metrics.execution_stats`)给执行率(数量口径)、换手(双边成交额 /
平均净值)与费用占比; 终端报告在有拒单时会多打一张"拒单原因"表。

## Python

```python
from event_backtest import DeclarativeStrategy, load_config, load_strategy, study_events

cfg = load_config("cn")
market = cfg.load(start="2026-01-05", end="2026-09-24", freq="30")
spec = load_strategy("volume_breakout")
result = cfg.run(DeclarativeStrategy(spec), start="2026-01-05", end="2026-09-24", freq="30")
study = study_events(market, spec.events(spec.compute(market), market), strategy_name=spec.name)
study.summary        # 每个 horizon 的均值 / 基线 / 超额 / 胜率
study.tests_table()  # 显著性: 均值 / t / p / bootstrap 区间 / 星号
```

## 事件研究的显著性

`--event-study` 除了描述性汇总, 还会打印一张检验表, 并存
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

## 市场配置

`src/event_backtest/setting/<market>.yaml`, 段为
`market / data / universe / account / execution / fees / rules`, 加载时校验字段:

```yaml
execution:
  fill_price: open                       # open / vwap
  slippage: {type: volume_share, volume_limit: 0.025, price_impact: 0.1}
  # 或 {type: fixed, spread: 0.01}
```

## 策略 YAML

`src/event_backtest/strategy/<name>.yaml`:

```yaml
name: 放量突破
factors:
  - relative_volume: {window: 16}
  - momentum: {window: 8}
signal:
  name: volume_breakout
  trigger: enter        # enter(上升沿) / every_bar
  all:
    - {factor: relative_volume, operation: greater_than, value: 2}
exit:
  hold_bars: 16
  take_profit: 0.06
  stop_loss: 0.04
sizing:
  max_positions: 3
```

## 因子 YAML

格式与 minievent 一致(全称字段), 见 `factor/relative_volume.yaml`。支持的运算:

`shift` / `rolling_mean` / `rolling_std` / `rolling_min` / `rolling_max` /
`rolling_sum` / `rolling_corr` / `cross_section_rank` / `add` / `subtract` /
`multiply` / `divide` / `elementwise_max` / `elementwise_min`。操作数写法
`{variable: ...}`(行情字段或前面步骤的 id) / `{parameter: ...}` / `{constant: ...}`。

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

跑法用脚本 `scripts/run_walk_forward.py`(改顶部参数, 编辑器点运行)。它把历史切成
若干折, 每折 训练 -> 隔离 -> 测试, 各折测试段首尾相接, 拼成一条样本外净值。

- **训练步骤是一个可选的选择器** `select(spec, train_market) -> spec`。
  `scripts/run_walk_forward.py` 里填上 `GRID` 就会在每折**训练段**扫网格、挑最优参数再用于
  **测试段**(这才是参数优化); `GRID` 留空则每折沿用同一份策略, 只是分折样本外评估。
- 选参只在训练段评估, 依据由 `SELECT_BY` 决定(默认 sharpe); 每折选到的参数会打印出来,
  方便看参数稳不稳。
- 正确性: 选择器只看到训练段之前的数据(测试段不外泄); 各折测试段不重叠、之间隔
  `embargo_days`; 每折用同一套费率 / 滑点, 样本外净值按折内收益连乘拼接。

## 切片器

`slice_dates(market, start, end)` 把行情复制一份、只保留指定日期范围(两端含), 用来
单独看某个区间(例如疫情那段):

```python
from event_backtest import slice_dates
covid = slice_dates(market, "2020-02-01", "2020-04-30")
```

> 已知限制: 因为没有包含退市成员的数据, 几乎总是用最近十年, 幸存者偏差存在。

## 目录

```
src/event_backtest/
  market.py       MarketData + liudb 加载 (cn / us) + 按 bar / 按日期切片
  rules.py        市场规则 (CNMarketRules / USMarketRules)
  fees.py         费率
  slippage.py     固定 / 成交量比例滑点
  factor/         因子 YAML + 自包含求值器
  signal/         条件 -> 事件
  strategy/       回调基类 + 声明式 YAML 策略 + walk_forward 解析
  broker.py       订单 / 组合 / 撮合
  engine.py       主循环 / Context / run
  walkforward.py  折划分 + select 钩子 + 样本外拼接
  sweep.py        点路径网格 + 搜索 + select 工厂(最小版, 不是框架)
  evaluation/     事件研究
  metrics.py      组合与交易统计
  report.py       rich 中文表(单场景 / 对比 / walk-forward)
  figure/         PNG(matplotlib) + 自包含交互式 HTML(零依赖原生 JS/SVG)
  benchmark.py    基准买入持有净值
  config.py       市场配置
  setting/        cn.yaml / us.yaml
  cli.py          命令行
```

## 交互式 HTML 报告

`figure/plot_tearsheet_html` / `plot_comparison_html` / `plot_walk_forward_html`
返回 `HtmlReport`, `.save(path)` 落成**单个自包含 HTML**: 数据内联 JSON, 样式与脚本
内联, 不引用 CDN, 断网 / 直接发文件都能打开。

- 内容: KPI 卡 + 净值(带基准线) + 回撤 + 月度收益 + 回合收益分布 + 指标分组表 +
  回合交易表 + 委托明细 / 拒单原因(tearsheet); 多场景净值叠加 + 可排序指标表(对比);
  样本外净值 + 各折收益 + 折明细(walk-forward)。
- 交互: 十字光标 + tooltip、图例点击开关、区间缩放(1月/3月/6月/1年/3年/5年/10年)、
  表格点表头排序、深浅色切换(默认跟随系统)。
- 口径与终端一致: 指标分组复用 `report.result_sections`, 执行 / 拒单复用 `metrics`,
  所以 HTML、终端表和 parquet 三边对得上。

```bash
# 脚本里三套都出: scripts/run_backtest.py -> comparison.html + 每个场景一份
#                  scripts/run_walk_forward.py -> oos.html
```

## 待办

- **扫参的表达力**: `sweep.set_path` 只能改**已存在**的字段(缺 `sizing` 段时扫
  `sizing.percent` 会 KeyError), 且可选指标只有 `evaluate_spec` 里那 5 个。
- **信号归因到成交**: `orders.tag` 能区分入场 / 出场规则, 但还没有把成交对回具体的
  `event_id`。

## 约定

- 因子以 T+0 为前提书写: 第 t 根用截至第 t 根收盘的数据, 不自己 `shift(1)`; 防未来
  函数由引擎负责(第 t 根算信号, 第 t+1 根成交)。
- A 股: T+1(新交易日首根 bar 解禁)、涨跌停(一字涨停买不进 / 一字跌停卖不出)、
  100 股整手、按 `sec_type` 的佣金 / 印花税 / 过户费。
- 美股: T+0、无涨跌停、1 股整手。
- 事件研究的基线是同时段无条件均值, 只作描述; 显著性检验按交易日聚类, 不做多重比较
  校正, 所以"显著"要结合 horizon 数量一起看。
