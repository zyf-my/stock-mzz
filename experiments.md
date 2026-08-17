# 实验记录

每次有效训练追加一条。没有数字的「大概好了」不算完成。对比实验一次只改窗口、特征集合、损失、模型四者之一。

## cs-lgbm-001
- 日期：2026-08-17
- 代码/配置：`configs/baseline.yaml`
- 输入特征：当天 raw 99 + 全市场 z-score 99 + 行业 z-score 99 + cat [0,1,2,3,4,6,7,8]；不用 cat_5
- 是否看历史：窗口 = 0（仅当天）
- 模型：LightGBM 回归，n_estimators=400, lr=0.05, num_leaves=31；训练按日最多 800 只股票
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.099910
- train mean RankIC（如有）：未算
- 耗时 / 硬件：本机 16GB RAM，CPU torch；valid 预测文件 21:19 写出
- 结论：保留作截面底仓。超过单因子 0.079，低于门槛 0.12
- 下一步：阶段 3 看崩溃日；再加历史统计特征（提示 1）

## hist-lgbm-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm.yaml`
- 输入特征：截面基线全部特征 + 过去 10 日 21 列的 mean/std/last（窗口 `[t-L, t)`）
- 是否看历史：窗口 = 10
- 模型：LightGBM 回归，超参与 cs-lgbm-001 相同
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.084406（对照 cs-lgbm-001 为 0.099910）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：产物约 22:06–22:07
- 结论：**抛弃作为主模型**。历史块占了 42% 增益，但验证集掉了约 0.016，负 RankIC 日从 49 增到 76。部分最差日略改善，整体更差。
- 下一步：已改 history.source=cs_zscore（相对强弱轨迹），重跑 hist_lgbm.yaml

## hist-lgbm-002
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm.yaml`（`source: cs_zscore`）
- 输入特征：截面基线 + 过去 10 日、每天先截面 z-score 再 mean/std/last
- 是否看历史：窗口 = 10，`[t-L, t)`
- 模型：LightGBM 回归，超参与 cs-lgbm-001 相同
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.104192（对照基线 0.099910，raw 历史 0.084406）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：产物约 22:36
- 结论：**保留为当前最强单模型**。修好了 raw 历史的掉分；负 RankIC 日 49→39。仍低于门槛 0.12，提升约 0.004，243 天验证集上属于小幅、方向正确。
- 下一步：按日 LambdaRank（同一套特征，只改损失）

## hist-rank-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_rank.yaml`
- 输入特征：与 hist-lgbm-002 相同（cs 历史）
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LGBMRanker LambdaRank；按日 5 档相关性；num_leaves=24, max_depth=6, min_child_samples=100, lambda_l2=2.0, n_estimators=250
- 损失：lambdarank（组=交易日）
- valid mean RankIC：-0.021285（对照 hist-lgbm-002 为 0.104192）
- train mean RankIC（如有）：未算
- 耗时 / 硬件：加载 87s，摊平 56s，拟合 75s，合计约 284s
- 结论：**抛弃**。防过拟合加得太死，排序标签又压成 5 档，分数接近随机且略反。主模型仍是 hist-lgbm-002。
- 下一步：不要继续堆 Ranker 变体；可试更轻的正则或回到回归 + 别的时序特征。不要覆盖 `hist_lgbm` 产物。

## hist-lgbm-l5-001
- 日期：2026-08-17
- 代码/配置：`configs/hist_lgbm_l5.yaml`
- 输入特征：与 hist-lgbm-002 相同，只把窗口 10 改为 5
- 是否看历史：窗口 = 5，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.097031（对照 L=10 为 0.104192，截面基线 0.099910）
- 耗时 / 硬件：合计约 381s
- 结论：**抛弃**。收窗口没有涨分，略低于 L=10。主模型仍是 hist-lgbm-002。
- 下一步：不要再缩窗口；主提交用 L=10 的 `task1_hist_lgbm.npy`

## fusion-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion.yaml`，`scripts/train_fusion.py`
- 输入特征：不重训。时序支=`hist_lgbm` valid 0.1042；截面支=`baseline` valid 0.0999
- 是否看历史：沿用两支已有模型；融合规则只在 valid 上估
- 模型：双轴融合。试了原始分加权、截面名次加权、行业内去均值、按当天覆盖度门控
- 损失：不训练树；验收 RankIC
- valid mean RankIC：0.106185（锁定 raw_blend，时序权重 0.7）
- 耗时 / 硬件：读数约 112s，融合本身数秒
- 结论：**保留为当前最强提交** `submissions/task1_fusion.npy`。名次融合几乎同分（0.10618）；覆盖度门控、行业中性化没有明显超过简单加权。相对单模 +0.002，243 天上可能有噪声，但方向符合官方「时间维 + 股票维融合」。
- 下一步：主提交在确认 `fusion_gru_blend` 之前仍用本文件。不要为 0.002 再海搜两棵树的权重。

## hist-ms-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_ms.yaml`
- 输入特征：关掉行业 z-score；10 日 mean/std/last/delta/ewm + 5 日 last；每天 900 只
- 是否看历史：窗口 10 + 短窗 5，source=cs_zscore
- 模型：LightGBM 回归，n_estimators=500, colsample=0.7, min_child_samples=40
- 损失：MSE 回归 y1
- valid mean RankIC：0.097166（对照 hist-lgbm-002 为 0.104192，融合 0.106185）
- 结论：**抛弃**。多尺度/ewm/delta 再加树，没有超过 L=10 的简单相对强弱。不要跑 `fusion_ms`。
- 下一步：主提交仍是 `submissions/task1_fusion.npy`；时序支不要再加复杂统计

## hist-wide-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_wide.yaml`
- 输入特征：相对 hist-lgbm-002 **去掉 raw 和行业 z-score**；每天最多 2000 只；按行业分层
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- valid mean RankIC：0.090571（对照 002 为 0.104192，融合 0.106185）
- 结论：**抛弃**。一次改了三样（砍 raw、加样本、分层），把 002 里约 23% 增益的 raw 砍掉了，分数掉到 0.091。不是「样本越多越好」。
- 下一步：停止改 002 的特征配方。主提交仍是 `task1_fusion.npy`。

## gru-001
- 日期：2026-08-18
- 代码/配置：`configs/gru.yaml`，`scripts/train_gru.py`
- 输入特征：21 列当天及过去的截面 z-score 窗口；不用类别、不用 raw
- 是否看历史：窗口 = 10，含当天（官方 X[t]→y1[t]）
- 模型：1 层 GRU hidden=64，每天最多 800 只，CPU
- 损失：MSE 回归 y1；验收 RankIC
- valid mean RankIC：0.086522（对照截面树 0.099910，时序树 0.104192，树融合 0.106185）
- train mean RankIC（如有）：未算
- 结论：**作为融合支保留，不作主模型**。单模弱于截面树，但与树的日均分数 Spearman 仅 0.37–0.48（两棵树之间 0.81–0.98）。在 hist 的 39 个负 RankIC 日里救回 24 天。
- 下一步：`fusion-gru-blend-001` 已落到 0.110069，用这版 MSE GRU，不要换成 cats/ic

## fusion-gru-blend-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion_gru_blend.yaml`
- 输入特征：不重训。GRU 支=`gru_valid.npy` 0.0865；树融合支=`fusion_valid.npy` 0.1062
- 是否看历史：沿用已有模型
- 模型：valid 上锁定 raw_blend，GRU 权重 0.25
- 损失：不训练；验收 RankIC
- valid mean RankIC：0.110069
- 耗时 / 硬件：读数 75s
- 结论：**当前最强。** 产物 `submissions/task1_fusion_gru_blend.npy`。比树融合 +0.004。不要覆盖旧的 `task1_fusion.npy` 之前先把本文件当候选主提交。
- 下一步：后面对照实验均未超过 0.110；停止堆 GRU 变体和 LambdaRank

## gru-cats-001
- 日期：2026-08-18
- 代码/配置：`configs/gru_cats.yaml`
- 输入特征：相对 gru-001 只加 cat_1、cat_6 embedding
- 是否看历史：窗口 = 10，含当天
- 模型：同 gru-001
- 损失：MSE
- valid mean RankIC：0.086718（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 436s，early stop 于 epoch 6；最佳是 epoch 3 的 0.0867
- 结论：**抛弃。** 与不加类别几乎同分。类别已被树用透，再塞进 GRU 只让第 1 轮掉到 0.057，没有新信号。不要跑 `fusion_gru_cats`。

## gru-ic-001
- 日期：2026-08-18
- 代码/配置：`configs/gru_ic.yaml`
- 输入特征：与 gru-001 相同
- 是否看历史：窗口 = 10，含当天
- 模型：同 gru-001
- 损失：当天 batch 上 1−Pearson（只改损失）
- valid mean RankIC：0.090548（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 254s；epoch 1 就是最佳 0.0905，之后每轮 valid 下滑
- 结论：**单模略好，融合更差。** 单模 +0.004，但和树更同向，三支融合只有 0.1084，低于 MSE GRU 的 0.1101。Pearson IC 让 GRU 去抢树已经会的排序，互补性被吃掉。融合支仍用 gru-001。

## fusion-gru-ic-001
- 日期：2026-08-18
- 代码/配置：`configs/fusion_gru_ic.yaml`
- valid mean RankIC：0.108402（锁定 rank_blend，GRU 权重 0.25）
- 结论：**抛弃。** 低于 fusion-gru-blend-001 的 0.110069。

## hist-n1200-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_n1200.yaml`
- 输入特征：相对 hist-lgbm-002 关掉行业 z-score，每天 800→1200，保留 raw
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- valid mean RankIC：0.103940（对照 002 为 0.104192）
- 耗时 / 硬件：合计 272s
- 结论：**抛弃。** 多抽股票没有涨分，243 天上属于噪声。每天 800 只够用。不要再加样本。

## hist-rank-mild-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_rank_mild.yaml`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10
- 模型：LGBMRanker；15 档；超参与回归版对齐（400 棵、leaves=31）
- 损失：lambdarank
- valid mean RankIC：−0.083801（对照 hist-rank-001 为 −0.021，回归 002 为 0.104192）
- 结论：**抛弃，停止 LambdaRank。** 正则放轻之后排得更反。y1 已经是截面分位数，MSE 本身就是在拟合排序；LambdaRank 优化的是 NDCG 头部品，和全市场 Spearman 不是同一目标。重要性更集中在 cat_6。不要再开 Ranker 变体。

模板：

```text
## 实验 ID
- 日期：
- 代码/配置：
- 输入特征：
- 是否看历史：窗口 =
- 模型：
- 损失：
- valid mean RankIC：
- train mean RankIC（如有）：
- 耗时 / 硬件：
- 结论（保留 / 抛弃 / 作为融合支）：
- 下一步：
```
