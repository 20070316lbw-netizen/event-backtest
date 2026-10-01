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
