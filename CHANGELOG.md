# CHANGELOG

## 0.1.0 (未发布)

### 调用者接口
- 新增 `EventBackTest` 门面: 超参数式构造, 一条链跑完 run / study / walk_forward /
  sweep / compare; 组件(load_config / engine.run / resolve_strategy / study_events /
  run_walk_forward / search / save_run)保持公开可组合。
- 新增 `Run`: 结果 + 配置 + 口径绑定, `summary()` / `trades()` / `execution()` /
  `study()` / `html()` / `save()` / `print()` 不再重复传 market / benchmark。
- 新增 `save_run`(公开落盘)与 `run.json` 运行清单(版本 / git commit / 配置 / 区间 /
  费率 / 滑点 / 指标快照 / 文件列表)。
- 新增 `Run.metrics()`: 类型化指标视图(属性名可补全, 缺字段是 None), 与
  `summary()` / 终端表 / HTML 同源。
- 新增 `save_comparison()`: 一次写 comparison.csv / png / html + 每个场景一份 tearsheet,
  替掉 scripts/run_backtest.py 里的样板。
- 新增 `load_market_cached()` 与 `MarketConfig.cached_load()`: 显式行情缓存(按参数组合
  LRU, 默认 2 份), 适合一份行情喂很多次回测; 注意返回的是同一份对象。
- CLI 新增 `show factor|strategy <名字>`: 打印 yaml 原文(注释即文档)与解析要点。
- 删除仓库根目录里未被引用的游离 `config.py` 与空目录 `chores/`。
- 新增 `BacktestResult.meta` / `MarketConfig.describe()`: 结果自带运行上下文。
- 新增 `resolve_strategy`: 统一解析 yaml 名 / yaml 路径 / 内置代码策略 / 策略对象。
- 新增 `evaluation.stats`: t 检验 p 值、bootstrap 区间与零假设重抽检验;
  `EventStudy.tests` 与 `tests_table()`。
- CLI: 新增 `--json`、`--bootstrap N`; run 子命令改为 `EventBackTest` 的消费者。
- 新增 `py.typed`(PEP 561)与 `ContextStrategy` 协议, 公开签名逐步收紧。
- 修正带 `DataFrame` / `ndarray` 字段的 dataclass 相等语义(`eq=False`): 以前
  `result_a == result_b` 会抛 "truth value of a Series is ambiguous", 且不可哈希。

### 策略与评估
- 事件研究新增显著性: 默认按交易日聚类的单样本 t 检验 + bootstrap 百分位区间,
  `test_on="excess"|"ret"`、`cluster="day"|"none"`、`bootstrap` / `alpha` / `seed` 可调;
  每个 horizon 单独检验, 未做多重比较校正。
- 未知因子运算不再静默回退成截面排名, 直接报 `FactorError`。
- 修复: 成交量比例滑点截量后未按整手取整, A 股会成交零股。
- 修复: `order_target_percent` 会委托零股(如 196 股 -> 成交 100 -> 剩 96 股零头被拒单);
  现在委托量先按整手取整, 清仓时连零股一起卖。真实数据上 "数量不足一手" 从 249 笔降到 0。
- 变更: 不写 `sizing.percent` 时的默认仓位从"全池等权 1/N"改成"按 `max_positions`
  等权分摊 1/max_positions"; 两者都没有才用 1/N(池子一大原口径几乎等于不投)。
- 信号归因到成交: `Order` / `Fill` / `orders.parquet` / `fills.parquet` / 回合交易都带
  `event_id`(回合里叫 `entry_event`), 并贯通 walk-forward 的事件注入。
- 回合交易含**期末未平仓**(用最后收盘价盯市, `open=True`, `exit_ts` 为 NaT);
  `trade_stats` 增加 `open_trades`, 胜率等只统计已平仓, 不再让"买入持有"显示成 0 笔交易。
- 扫参: 点路径中间缺的段会自动补出来(没写 `sizing` 段也能扫 `sizing.percent`);
  `metric` 可选报告里的任意数值指标(calmar / sortino / win_rate / execution_rate ...),
  写错直接报 `SweepError` 并列出可选值。

### 回测与数据
- 订单簿记: `orders`(拒单原因 / 期末未成交 / 部分成交 / 来源 tag)、执行率、换手与
  费用占比; 终端多一张"拒单原因"表。
- `fill_price` 校验: 未知取值报错; `vwap` 在没有成交额的数据上开跑前报错。
- 数据库文件不存在时报清晰的 `FileNotFoundError`。
- 新增 `scripts/update_ashare.py`(BaoStock -> liudb)与 `setting/cn_stock.yaml`
  (10 年 A 股股票池)。
- 新增自包含交互式 HTML 报告: tearsheet / 场景对比 / walk-forward, 零第三方依赖。

### 工程
- 新增 GitHub Actions(ruff + pytest 3.12 / 3.13)。
- 依赖新增 `scipy`(t 分布)与 `py.typed`。
- 性能: 全 S&P 500(503 只 / 2511 根 / 122.6 万行)行情加载 136.9s -> 0.85s ——
  元凶是"哪些证券没行情"的检查把 `set(列)` 写在推导式里, 每只证券都重建一遍。
- 文档: README 系统性重写(快速开始 / 命令行 / 输出文件 / 策略与因子 YAML 速查 /
  事件研究 / walk-forward / 扫参 / HTML / 已知限制), 因子与策略 yaml 全部加逐字段注释。
