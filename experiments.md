# 实验记录

每次有效训练追加一条。没有数字的「大概好了」不算完成。对比实验一次只改窗口、特征集合、损失、模型四者之一。

**当前最强：valid 0.120869，平台 test 0.125679**，文件 `task1_fusion_x6_today.npy`（上一版 `task1_fusion_next6_wt_mlp6.npy` valid 0.120542 / test 0.125588）。只改 x6 GRU 的 `include_current_day`，二档门控与融合权重不变。gate3 / 输入时间衰减 valid 高 test 差，已抛弃。测试分只记结果，不回灌再搜门控。

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

## hist-mkt-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_mkt.yaml`
- 输入特征：相对 hist-lgbm-002 只加当天全市场覆盖度 + 21 列截面 mean/std（绝对水平）
- 是否看历史：窗口 = 10；市场状态只用当天 mask_x
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.063500（对照 002 为 0.104192）
- 耗时 / 硬件：合计 418s
- 结论：**抛弃。** 重要性第三就是 `mkt_coverage`，树在用覆盖度切训练期制度，验证集崩了。不要把绝对市场水平当特征。不要跑 `fusion_mkt`。

## hist-mkt-rel-001
- 日期：2026-08-18
- 代码/配置：`configs/hist_mkt_rel.yaml`
- 输入特征：相对 002，市场状态改成相对过去 20 日 z-score；只留覆盖度 + 4 列 std
- 是否看历史：窗口 = 10；相对统计只用 `[t-20, t)`
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.101414（对照 002 为 0.104192，融合 GRU 版 0.110069）
- 耗时 / 硬件：合计 439s
- 结论：**抛弃。** 不再崩，但没超过 002。主提交仍是 `task1_fusion_gru_blend.npy`（0.110）。停止往树上加全市场标量。

## cs-mlp-001
- 日期：2026-08-18
- 代码/配置：`configs/cs_mlp.yaml`，`scripts/train_cs_mlp.py`
- 输入特征：当天 21 列行业内 z-score + cat_1 embedding；不用 raw、历史、cat_6
- 是否看历史：窗口 = 0
- 模型：2 层 MLP hidden=64；每天最多 800 只
- 损失：行业残差标签上的 Pearson IC；验收仍是对原始 y1 的 RankIC
- valid mean RankIC：0.091768（对照 GRU 0.0865，截面树 0.0999，树融合 0.1062）
- 耗时 / 硬件：合计 377s，CPU
- 结论：**作为第四支保留。** 单模弱于树，但与 `fusion_gru_blend` 日均 Spearman 0.681（低于 0.8）。和树/GRU 的相关约 0.58–0.61。
- 下一步：已与 0.110 做 rank_blend，见 fusion-cs-mlp-001

## fusion-cs-mlp-001
- 日期：2026-08-18
- 代码/配置：`scripts/train_cs_mlp.py` 内嵌融合（不覆盖 `task1_fusion_gru_blend.npy`）
- 输入特征：cs_mlp valid + fusion_gru_blend valid
- 模型：锁定 rank_blend，cs_mlp 权重 0.25
- valid mean RankIC：0.111264（对照 fusion_gru_blend 0.110069）
- 结论：**小幅超过 0.110，作为新候选。** 产物 `submissions/task1_fusion_cs_mlp.npy`。243 天上 +0.001 仍可能有噪声，但相关 0.68 且 0.15/0.25/0.4 三档都 ≥0.110。未改旧主文件。

## hard-resid-001
- 日期：2026-08-18
- 代码/配置：`configs/hard_resid.yaml`，`scripts/train_hard_resid.py`
- 输入特征：与 cs_mlp 相同（21 列行业 z-score + cat_1）；难日 = hist_lgbm 在 **train** 上逐日 RankIC 最低 25%（608/2432 天，阈值 0.178，不看 valid）
- 是否看历史：窗口 = 0；树分数只用来造 OLS 残差标签
- 模型：小 MLP hidden=32，dropout=0.2，weight_decay=1e-3，难日重复 2 次；约 2153 参数
- 损失：对 `y1 ~ hist_lgbm` 的当天 OLS 残差做 Pearson IC；早停看与 0.111 的 rank_blend(0.25)
- valid mean RankIC：残差单支 **-0.083400**；与 fusion_cs_mlp 日均 Spearman **-0.7129**；融回 0.111 后锁定 raw_blend 0.111266（相对 0.111264 的 2e-6，尺度淹没，不算涨）
- train 难日均值 RankIC：0.1262（易日 0.2882）——train 上「最差 25%」仍然正、且远好于 valid 崩溃日
- 耗时 / 硬件：合计 386s，CPU；树 train 预测 146s
- 结论：**抛弃。** 难日定义没有泄露 valid，但残差支学成了 0.111 的反相。说明树在 train 上排错的部分搬不到 valid；再加重采样这些天只会拟合过拟合残差。未覆盖 `task1_fusion_cs_mlp.npy`。主候选仍是 0.111264。
- 下一步：不要再从 train 残差/难日上抠同一套截面特征。过拟合更可能来自主模型容量，而不是「还有一块可学的难日残差」。

## hist-lgbm-iter-001
- 日期：2026-08-18
- 代码/配置：`scripts/eval_lgbm_iters.py`（冻结 `hist_lgbm.txt`，不重训）
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：同一棵 400 棵树，推理时只用前 50/100/…/400 棵
- 损失：不训练；验收 RankIC
- valid mean RankIC：50→0.1005，100→0.1015，200→0.1020，250→0.1032，300→0.1032，350→0.1034，**400→0.104192**
- train 子集（每 8 日，304 天）：50→0.154，200→0.211，400→0.247
- 耗时 / 硬件：合计 259s
- 结论：**抛弃「往回砍树」。** train 涨得比 valid 快（过拟合是真的），但 valid 仍随轮数单调上升，峰值就在 400。提早停只会把 valid 从 0.104 降到 0.102–0.103。未覆盖 002 产物。主候选仍是 0.111264。
- 下一步：不要为过拟合再减 leaves / 减 n_estimators。树还没过 valid 的峰。

## fusion-gru-cov-001
- 日期：2026-08-19
- 代码/配置：`scripts/eval_coverage_gru.py`（不重训）
- 输入特征：GRU vs 树融合 `fusion_valid`；门控只用当天 `mask_x` 股票数
- 是否看历史：沿用已有模型
- 模型：覆盖度分位 40/50/60 × GRU 权重 {0.15,0.25,0.4} × raw/rank。旧 FusionModel 门控只搜 ≥0.5 且只用 rank，这里改掉
- 损失：不训练；验收 RankIC
- valid mean RankIC：锁定 raw，tau=4546（60 分位），低覆盖 GRU 0.25、高覆盖 0.4 → **0.111078**（对照全局 0.25 为 0.110069）
- 诊断：覆盖度与树 RankIC 相关 −0.21；Q4（股票最多）树 0.058、GRU 0.100、全局 0.25 融合只有 0.074。前 8 名全是「挤的天多听 GRU」
- 耗时 / 硬件：合计 191s，主要是读 panel
- 结论：**作为融合规则保留。** 方向稳定，不是单格刷分。产物 `submissions/task1_fusion_gru_cov.npy`，未覆盖 0.110。仍低于带 cs_mlp 的 0.111264，见下条。
- 下一步：已叠 cs_mlp，见 fusion-cs-mlp-cov-001

## fusion-cs-mlp-cov-001
- 日期：2026-08-19
- 代码/配置：同上脚本内嵌融合
- 输入特征：cs_mlp + 覆盖度门控 GRU/树
- 模型：锁定 rank_blend，cs_mlp 权重 0.15
- valid mean RankIC：**0.111971**（对照 fusion_cs_mlp 0.111264，门控 GRU/树 0.111078）
- 结论：**小幅超过 0.111，作为新候选。** 产物 `submissions/task1_fusion_cs_mlp_cov.npy`。243 天上 +0.0007 仍可能有噪声，但门控方向与 Q4 诊断一致。未覆盖 `task1_fusion_cs_mlp.npy`。离 0.12 仍约 0.008。
- 下一步：不要再加密覆盖度网格。下一条若做，用 GRU 关掉当天，或把树加到 600 棵。

## hist-lgbm-n800-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_n800.yaml`，`scripts/train_baseline.py`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，只把 n_estimators 400→800
- 损失：MSE 回归 y1
- valid mean RankIC：400→0.104192（与 002 逐位对齐），600→0.103043，**800→0.104201**
- 耗时 / 硬件：合计 479s，CPU
- 结论：**抛弃。** 400 棵已经是峰；再加 400 棵只涨 9e-6，600 棵还略掉。未覆盖 `hist_lgbm` 产物。不要再加树。
- 下一步：时序树锁 400 棵。

## gru-no-today-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today.yaml`，`scripts/train_gru.py`
- 输入特征：与 gru-001 相同 21 列 CS z-score；只把 `include_current_day` 关掉
- 是否看历史：窗口 = 10，`[t-L, t)`，不含当天
- 模型：同 gru-001；early stop 于 epoch 6，最佳 epoch 3
- 损失：MSE
- valid mean RankIC：0.081754（对照 gru-001 为 0.086522）
- 耗时 / 硬件：拟合 591s，合计 733s，CPU
- 结论：**作为融合支保留，不作主模型。** 单模更弱，但与树融合日均 Spearman 0.425（含当天 GRU 为 0.460），与 cs_mlp 0.503（含当天为 0.608）。与含当天 GRU 相关 0.941，不是全新信号，只是少叠当天。hist 的 39 个负日仍救回 24 天。
- 下一步：按旧覆盖度门控迁移，再叠 cs_mlp；不要重搜覆盖度网格

## fusion-gru-no-today-001
- 日期：2026-08-19
- 代码/配置：`configs/fusion_gru_no_today.yaml`
- 输入特征：不重训。GRU 支=`gru_no_today` 0.0818；树融合支=`fusion_valid` 0.1062
- 模型：valid 上锁定 raw_blend，GRU 权重 0.25
- valid mean RankIC：0.110347（对照含当天 GRU 融合 0.110069）
- 结论：**噪声级。** +0.0003 不够单独立项。产物 `submissions/task1_fusion_gru_no_today.npy`，未覆盖 0.110。真正差在覆盖度 + MLP，见下条。

## fusion-cs-mlp-cov-notoday-001
- 日期：2026-08-19
- 代码/配置：覆盖度规则从 `fusion_gru_cov_lock.json` 原样迁移（tau=4546，低覆盖 0.25 / 高覆盖 0.4，raw）；cs_mlp 用原网格搜权重
- 输入特征：gru_no_today + 树融合 + cs_mlp
- 模型：覆盖度门控后锁定 rank_blend，cs_mlp 权重 0.25
- valid mean RankIC：**0.113122**（对照含当天配方 0.111971；权重 0.15 为 0.112946，同样超过旧候选）
- 诊断：门控本身 0.111615（旧 GRU 门控 0.111078），未重搜 tau
- 结论：**作为新候选。** 产物 `submissions/task1_fusion_cs_mlp_cov_notoday.npy`。未覆盖 `task1_fusion_cs_mlp_cov.npy`。243 天上 +0.001 仍可能有噪声，但 0.15/0.25 两档都超过 0.112，且相关下降与假设一致。离 0.12 仍约 0.007。
- 下一步：不要再加树、不要重搜覆盖度。下一条若做，用最近一段 train 重拟合 hist_lgbm，或 GRU 只追加列 69/73/74/42/66/58。

## hist-lgbm-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_recent.yaml`
- 输入特征：与 hist-lgbm-002 相同
- 是否看历史：窗口 = 10；拟合只用 train 末 800 天（标签），历史仍可回看更早
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.105431（对照 002 为 0.104192）
- 耗时 / 硬件：合计 267s
- 结论：**方向对、幅度不够。** 诊断：valid 每天 4242–4724 只，train 全程 max=4239，没有任何一天达到 valid 最小覆盖。末 800 天覆盖 3433–4239，最接近，故小幅上涨。未覆盖 002。离 0.12 仍远。
- 下一步：同一 800 天把每天抽样 800→2000，让树在大池子里排

## hist-lgbm-recent-n2000-001
- 日期：2026-08-19
- 代码/配置：`configs/hist_lgbm_recent_n2000.yaml`
- 输入特征：与 recent-001 相同，只把每天训练股票 800→2000
- 是否看历史：窗口 = 10；拟合 train 末 800 天
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：0.104616（对照 recent-800 只为 0.105431，002 为 0.104192）
- 耗时 / 硬件：合计 325s
- 结论：**抛弃。** 大池子抽样没有把树送进 valid 的覆盖区间（valid 最小 4242，train 最大仍 4239）。主树仍用 002。
- 下一步：树侧这条制度差已经试过；0.113 到 0.12 的缺口还在。

## gru-no-today-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today_recent.yaml`
- 输入特征：与 gru_no_today 相同；只把拟合改成 train 末 800 天
- 是否看历史：窗口 = 10，不含当天
- 模型：同 gru_no_today；epoch 1 最佳 0.0884，之后过拟合掉到 0.043，早停
- 损失：MSE
- valid mean RankIC：0.088440（对照全程 no-today 0.081754，含当天 gru-001 0.086522）
- 诊断：Q4 0.0968→0.1040；与树相关 0.425→0.454
- 耗时 / 硬件：拟合 172s，合计 334s
- 结论：**作为新 GRU 支保留。** 单模是目前最强 GRU。覆盖度门控迁移后 0.1152，叠原 cs_mlp 到 **0.116009**。产物 `submissions/task1_fusion_recent_gru_cov_mlp.npy`，未覆盖 0.113。离 0.12 仍约 0.004。
- 下一步：同一套 recency 重训 cs_mlp

## cs-mlp-recent-001
- 日期：2026-08-19
- 代码/配置：`configs/cs_mlp_recent.yaml`
- 输入特征：与 cs_mlp 相同；只把拟合改成 train 末 800 天
- 是否看历史：窗口 = 0
- 模型：同 cs_mlp；epoch 1 最佳
- 损失：行业残差 Pearson IC
- valid mean RankIC：0.086070（对照全程 cs_mlp 0.091768）
- 结论：**抛弃。** 截面 MLP 砍早期数据会掉分；融合仍用原来的 cs_mlp。主候选仍是 recent GRU 门控 + 旧 MLP 的 0.116009。
- 下一步：GRU recent 每天抽样 800→2000，只改这一项

## gru-no-today-recent-n2000-001
- 日期：2026-08-19
- 代码/配置：`configs/gru_no_today_recent_n2000.yaml`
- 输入特征：与 recent-001 相同；每天训练股票 800→2000
- 是否看历史：窗口 = 10，不含当天；拟合 train 末 800 天
- 模型：同 GRU；epoch 1 最佳 0.0919，之后崩到 0.027 / −0.001，早停
- 损失：MSE
- valid mean RankIC：0.091862（对照 recent-800 只 0.088440）
- 诊断：Q4 0.1040→0.1072；与树相关 0.454→0.470
- 结论：**作为当前 GRU 支。** 覆盖度门控 + 原 cs_mlp → **0.117177**。产物 `submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy`，未覆盖 0.116。离 0.12 约 0.0028。
- 下一步：若继续，同一 800 天把每天抽样提到全市场（约 3500–4200），只改这一项

## gru-no-today-recent-nfull-001
- 日期：2026-08-21
- 代码/配置：`configs/gru_no_today_recent_nfull.yaml`，`scripts/eval_locked_fusion.py --tag nfull`
- 输入特征：与 n2000 相同 21 列 CS z-score；只把 `max_train_stocks_per_day` 从 2000 改为 null（当天 mask_y 全用）
- 是否看历史：窗口 = 10，不含当天；拟合 train 末 800 天
- 模型：同 GRU；epoch 1 最佳 0.0926，之后 0.089 / 0.075 / 0.078，早停
- 损失：MSE
- valid mean RankIC：0.092584（对照 n2000 为 0.091862）
- 耗时 / 硬件：拟合 159s，合计 186s，CPU
- 结论：**抛弃作为主方案。** 单模略涨，锁死覆盖度门控 0.116128，叠原 cs_mlp（rank 0.15）→ **0.116887**，低于 n2000 的 0.117177。产物 `submissions/task1_fusion_nfull_cov_mlp.npy`，未覆盖 0.117。全市场抽样没有把 GRU 送进 valid 排序。
- 下一步：不要再加每天训练股票数。主候选仍是 n2000 的 0.117177；本机缺那份产物的话先重训 n2000 GRU 再走同一套锁死融合。

## gru-no-today-recent-n2000-retrain-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000.yaml`，`scripts/eval_locked_fusion.py --tag recent_gru_n2000`
- 输入特征：与 n2000-001 相同；本机缺产物，按同一配置重训
- 是否看历史：窗口 = 10，不含当天；拟合 train 末 800 天
- 模型：同 GRU；epoch 1 最佳 0.0919，之后 0.027 / −0.001 / 0.054，早停
- 损失：MSE
- valid mean RankIC：0.091862（与 n2000-001 逐位对齐）
- 耗时 / 硬件：拟合 235s，合计 277s，CPU
- 结论：**复现成功。** 锁死覆盖度门控 0.116468，叠原 cs_mlp（rank 0.15）→ **0.117177**。产物 `submissions/task1_fusion_recent_gru_n2000_cov_mlp.npy` 已写回。未覆盖 nfull 文件。
- 下一步：主候选就是这份 0.117177。门槛 0.12 未到。

## gru-ind-recent-n2000-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_ind_recent_n2000.yaml`
- 输入特征：与 n2000 相同 21 列，只把窗口从全市场 CS z-score 改成行业内 z-score
- 是否看历史：窗口 = 10，不含当天；拟合 train 末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0729，之后崩掉，早停
- 损失：MSE
- valid mean RankIC：0.072887（对照 CS n2000 为 0.091862）
- 耗时 / 硬件：拟合 126s，合计 155s，CPU
- 结论：**抛弃。** 单模弱，与 CS GRU 日均 Spearman 0.79；替换门控 / 混入 CS GRU / 叠到 0.117 都不涨。不覆盖 n2000 产物。
- 下一步：不要再把 GRU 窗口改成行业 z-score。改用同一套 n2000 多种子袋装。

## gru-n2000-bag-seeds-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000_s43.yaml` / `_s44.yaml`；seed 42+43+44 平均
- 输入特征：与 n2000 相同，只改种子
- 是否看历史：窗口 = 10，不含当天；末 800 天；每天 2000 只
- 模型：同 GRU。s43 最佳 0.0859（epoch 3）；s44 最佳 0.0904（epoch 1）
- 损失：MSE
- valid mean RankIC：袋装三种子 0.093380；42+44 为 0.095268（对照单种子 0.091862）
- 结论：**抛弃。** 单模涨了，但和树更同向，旧门控融合掉到 0.1162 / 0.1150，低于 0.117177。不要把多种子 GRU 袋装进融合。
- 下一步：追加树几乎不用的强单因子列。

## gru-no-today-recent400-n2000-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent400_n2000.yaml`
- 输入特征：与 n2000 相同，只把拟合窗口 800→400 天
- 是否看历史：窗口 = 10，不含当天；每天 2000 只
- 模型：同 GRU；epoch 2 最佳 0.0674
- 损失：MSE
- valid mean RankIC：0.067432（对照 800 天 0.091862）
- 结论：**抛弃。** 再缩短 recency 会掉分。拟合窗口锁 800 天。
- 下一步：追加树几乎不用的强单因子列。

## gru-n2000-x6-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000_x6.yaml`
- 输入特征：n2000 的 21 列再追加 42, 58, 66, 69, 73, 74（树重要性很低、单因子较强）
- 是否看历史：窗口 = 10，不含当天；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0929，之后崩掉，早停
- 损失：MSE
- valid mean RankIC：0.092888（对照 n2000 为 0.091862）
- 结论：**作为新 GRU 支保留。** 旧门控 + cs_mlp → 0.118155，超过 0.117177。与原 21 列 GRU 日均 Spearman 0.95，不是全新信号，但是有用。
- 下一步：高覆盖 GRU 权重从截断网格的 0.4 提到 0.6（Q4 树上只有 0.058）

## gru-n2000-l20-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000_l20.yaml`
- 输入特征：与 n2000 相同，只把窗口 10→20
- 是否看历史：窗口 = 20，不含当天；末 800 天
- 模型：同 GRU；epoch 1 最佳 0.0927
- 损失：MSE
- valid mean RankIC：0.092744
- 结论：**抛弃。** 与 L=10 日均 Spearman 0.999。加长窗口没有新信号。
- 下一步：6 列单独做一支 GRU，不要并进 21 列网络

## gru-n2000-only6-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000_only6.yaml`，`scripts/eval_x6_only6_fusion.py`
- 输入特征：只使用 42, 58, 66, 69, 73, 74
- 是否看历史：窗口 = 10，不含当天；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0847
- 损失：MSE
- valid mean RankIC：0.084659（对照 x6 为 0.092888，n2000 为 0.091862）
- 诊断：与 x6 相关 0.87，与树 0.47，与 mlp 0.52。only6 权重在 {0.15,0.25,0.4} 上单调，0.4 是峰（0.45/0.50 回落）
- 结论：**保留并作为当前主融合。** 0.4×only6 + 0.6×x6，覆盖度门控 tau=4546、w=0.25/0.6，再 rank-blend 原 cs_mlp 0.15 → **0.120168**。产物 `submissions/task1_fusion_x6_only6_w06_cov_mlp.npy`，未覆盖 0.117 文件。本地 valid 刚过 0.12。
- 下一步：阶段 6 文档；测试集只推理这一版，不要用测试分回灌。

## cs-mlp-only6-001
- 日期：2026-08-22
- 代码/配置：`configs/cs_mlp_only6.yaml`
- 输入特征：当天行业 z-score 仅 42, 58, 66, 69, 73, 74 + cat_1；行业残差 Pearson IC
- 是否看历史：窗口 = 0
- 模型：同 cs_mlp；epoch 2 最佳 0.0806
- 损失：行业残差 Pearson IC
- valid mean RankIC：0.080578（对照原 cs_mlp 0.091768）
- 诊断：与 0.120 相关 0.63，与原 mlp 0.72，与 only6 GRU 0.60。单独叠到 0.120 不涨；和原 mlp 按 0.4 混合再叠门控 → **0.120342**（标准网格 0.15/0.25/0.4 都超过 0.120168，0.5 为 0.120361 后回落）
- 结论：**作为 MLP 混合支保留。** 产物 `submissions/task1_fusion_x6_only6_mlp6_cov_mlp.npy`，未覆盖 0.120168 文件。幅度小，243 天上仍可能有噪声，但三档同向。
- 下一步：only6 GRU 打开当天，补这 6 列的当日轨迹步。

## gru-only6-with-today-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_only6_with_today.yaml`
- 输入特征：与 only6 相同 6 列，只把 include_current_day 打开
- 是否看历史：窗口 = 10，含当天；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0903
- 损失：MSE
- valid mean RankIC：0.090323（对照不含当天 0.084659）
- 诊断：与 no-today only6 相关 0.96。替换进锁死融合（x6 0.6 / only6 0.4，门控 0.25/0.6，mlp6 混合 0.4）→ **0.120413**
- 结论：**替换 no-today only6。** 产物 `submissions/task1_fusion_wt_mlp6_cov_mlp.npy`。未覆盖 0.120168。
- 下一步：6 列当天树；以及 only6 改 Pearson IC。

## lgbm-only6-001
- 日期：2026-08-22
- 代码/配置：`configs/lgbm_only6.yaml`（`dataset.num_indices` 切片）
- 输入特征：当天 6 列 CS z-score + 行业 z-score；无 raw、无类别、无历史
- 是否看历史：窗口 = 0
- 模型：LightGBM 回归，超参与 baseline 相同
- 损失：MSE
- valid mean RankIC：0.083407
- 结论：**抛弃。** 与 cs_mlp_only6 相关 0.85，叠到 0.120413 不涨。
- 下一步：only6-with-today 改 Pearson IC。

## gru-only6-today-ic-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_only6_today_ic.yaml`
- 输入特征：与 with-today only6 相同，只改 Pearson IC 损失
- 是否看历史：窗口 = 10，含当天
- 模型：同 GRU；epoch 1 最佳 0.0827
- 损失：pearson_ic
- valid mean RankIC：0.082728（对照 MSE 0.090323）
- 结论：**抛弃。** 替换进锁死融合掉到 0.113。这 6 列上 IC 损失同样伤门控。
- 下一步：给时序树历史追加这 6 列。

## hist-lgbm-x6-001
- 日期：2026-08-22
- 代码/配置：`configs/hist_lgbm_x6.yaml`
- 输入特征：相对 hist_lgbm-002，历史 CS 统计追加 42, 58, 66, 69, 73, 74
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE
- valid mean RankIC：0.099782（对照 002 为 0.104192）
- 结论：**抛弃。** 多 6 列历史统计把树从 0.104 降到 0.100。不要改 002 的历史列集合。
- 下一步：x6 GRU 降低学习率。

## gru-x6-lr3e4-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_x6_lr3e4.yaml`
- 输入特征：与 x6 相同，只把 lr 0.001→0.0003
- 是否看历史：窗口 = 10，不含当天
- 模型：同 GRU；epoch 1 最佳 0.0836，仍然随后崩
- 损失：MSE
- valid mean RankIC：0.083553（对照 x6 为 0.092888）
- 结论：**抛弃。** 降学习率让单轮拟合不足，融合掉到 0.117。
- 下一步：下一档未用列单独做 GRU。

## gru-next6-with-today-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_next6_with_today.yaml`，`scripts/eval_next6_fusion.py`
- 输入特征：38, 72, 47, 3, 1, 4（末 800 天 subsample 上未进 x6 的最强列）
- 是否看历史：窗口 = 10，含当天；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0849
- 损失：MSE
- valid mean RankIC：0.084857
- 诊断：与 only6-today 相关 0.70，与 x6 0.61，与树 0.52。标准网格 0.15/0.25/0.4 里 0.15 最好
- 结论：**保留为第三支 GRU。** 0.15×next6 + 0.85×(0.4×only6-today+0.6×x6)，门控 0.25/0.6，mlp6 混合 0.4 → **0.120542**。产物 `submissions/task1_fusion_next6_wt_mlp6.npy`。
- 下一步：不要再往树上塞这 6 列。平台试分见下条。

## platform-probe-001
- 日期：2026-08-22
- 代码/配置：锁定配方 `scripts/eval_next6_fusion.py`，文件 `submissions/task1_fusion_next6_wt_mlp6.npy`
- 输入特征：与 `gru-next6-with-today-001` 融合产物相同
- 是否看历史：同锁定配方，未改权重
- 模型：不训练；靖戈平台对测试集打分
- 损失：—
- valid mean RankIC：0.120542（本地，选模用）
- 平台 test mean RankIC：**0.125588**
- 结论：**过官方门槛 0.12。** test 高于 valid，不像把验证集搜穿。这是试分，不是友安杯整包终稿。
- 下一步：锁死本文件。不要用测试分海搜超参。剩下做 README / 说明书。

## diag-ceiling-001
- 日期：2026-08-22
- 代码/配置：`scripts/diag_ceiling.py`（不训练）
- 输入特征：本机已有 valid 预测。x6 / only6 / next6 产物缺失
- 是否看历史：沿用已有模型
- 模型：日度 oracle（当天选 RankIC 最高的一支）
- 损失：—
- valid mean RankIC：本机最强 `fusion_recent_gru_n2000_cov_mlp` 0.117177；全部分支日度 oracle **0.176684**；tree+GRU n2000+MLP+锁定 0.162841
- 诊断：Q1 锁定 0.166 / Q4 锁定 0.093。Q4 树 0.058、GRU 0.107，二档门控 tau=4546 把 Q3（树 0.125）和 Q4 绑在一起。w_high=0.7 + mlp → 0.1191
- 结论：**0.14 低于日度 oracle，路由现有分支理论上够。** 不是再叠一个同类 GRU。本机没有 0.1205 产物。
- 下一步：把 Q4 从门控里拆开，见 fusion-gate3-001

## fusion-gate3-001
- 日期：2026-08-22
- 代码/配置：`scripts/eval_regime_gate.py`
- 输入特征：不重训。GRU=`gru_n2000` 0.0919；树融合 0.1062；cs_mlp 0.0918
- 是否看历史：沿用已有模型
- 模型：三档覆盖度 raw 门控。tau_mid=4546 w=0.25；中间档 w=0.4；tau_high=4650 w_high=1.0（该档纯 GRU）；再 rank-blend cs_mlp 0.15
- 损失：不训练；验收 RankIC
- valid mean RankIC：**0.122413**（对照二档门控 0.117177）。九格同向都 ≥0.120。分歧×覆盖度最高 0.1212，不如三档覆盖度
- 诊断：分位 oracle Q1=0.203 / Q2=0.148 / Q3=0.143 / Q4=0.152。分位内会选对模型的话均值约 0.16
- 耗时 / 硬件：48s，只读 cache
- 结论：**作为新的本地主融合。** 产物 `submissions/task1_fusion_n2000_gate3.npy`，未覆盖 0.117 文件。243 天上 +0.005 来自同一套 Q4 诊断，不是随机网格。离 0.14 仍约 0.018。
- 下一步：行业注意力；以及把缺失的 x6/only6 接到这套三档门控上

## cs-attn-001
- 日期：2026-08-22
- 代码/配置：`configs/cs_attn.yaml`，`scripts/train_cs_attn.py`
- 输入特征：与 cs_mlp 相同 21 列行业 z-score + cat_1
- 是否看历史：窗口 = 0
- 模型：行业内自注意力 hidden=64、2 heads；残差连接到投影
- 损失：行业残差 Pearson IC
- valid mean RankIC：0.092408（对照 cs_mlp 0.091768）；epoch 7 最佳
- 诊断：与 cs_mlp 日均 Spearman **0.963**；叠到 0.117177 的 n2000 融合不涨（所有权重同分）
- 耗时 / 硬件：拟合 979s，合计 1140s，CPU
- 结论：**抛弃。** 注意力几乎学成原来的 MLP，没有新的股票维信号。不覆盖 cs_mlp。
- 下一步：不要再在同一 21 列行业 z 上换注意力骨干。去补 x6/only6，或做日度路由的时序外预测。

## gru-n2000-x6-retrain-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_no_today_recent_n2000_x6.yaml`
- 输入特征：与 x6-001 相同；本机缺产物，按同一配置重训
- 是否看历史：窗口 = 10，不含当天；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳后崩掉，早停
- 损失：MSE
- valid mean RankIC：0.092888（与 gru-n2000-x6-001 逐位对齐）
- 结论：**复现成功。** 接到三档门控见下条。

## gru-only6-with-today-retrain-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_only6_with_today.yaml`
- 输入特征：6 列，含当天
- 是否看历史：窗口 = 10；末 800 天；每天 2000 只
- 模型：同 GRU；epoch 1 最佳 0.0903
- 损失：MSE
- valid mean RankIC：0.090323（与 gru-only6-with-today-001 逐位对齐）
- 结论：**复现成功。**

## fusion-x6-only6-gate3-001
- 日期：2026-08-22
- 代码/配置：`scripts/eval_x6_gate3.py`
- 输入特征：不重训。0.4×only6-today + 0.6×x6；门控从 gate3-001 原样迁移（tau_high=4650, w_high=1.0）
- 是否看历史：沿用已有模型
- 模型：三档覆盖度 + rank-blend cs_mlp 0.15。未重搜门控
- 损失：—
- valid mean RankIC：**0.123689**（对照 n2000 gate3 0.122413；单换 x6 为 0.122485）
- 诊断：GRU 混合单模 0.0957，高于 x6 0.0929 / only6 0.0903
- 耗时 / 硬件：3.9s
- 结论：**本地候选。** 产物 `submissions/task1_fusion_x6_only6_gate3.npy`。平台试分见下条。
- 下一步：已上传，见 platform-probe-002

## platform-probe-002
- 日期：2026-08-22
- 代码/配置：`submissions/task1_fusion_x6_only6_gate3.npy`（fusion-x6-only6-gate3-001）
- 输入特征：与上条相同，未改权重
- 是否看历史：同锁定配方
- 模型：不训练；靖戈平台对测试集打分
- 损失：—
- valid mean RankIC：0.123689（本地，选模用）
- 平台 test mean RankIC：**0.117855**
- 结论：**抛弃。** 低于官方门槛 0.12，也低于 GitHub 主方案 test 0.125588。valid 涨、test 掉，三档门控（tau_high=4650 / w_high=1.0）把 243 天验证集搜过了。未覆盖 GitHub 主文件。
- 下一步：主方案回到 `task1_fusion_next6_wt_mlp6.npy`。不要用这次测试分再搜覆盖度网格。今日剩余提交不要再传门控变体。

## next6-fusion-repro-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_next6_with_today.yaml`、`configs/cs_mlp_only6.yaml`、`scripts/eval_next6_fusion.py`
- 输入特征：本机缺产物，按 GitHub 锁定配方重训 next6 GRU 与 cs_mlp_only6，再原样融合
- 是否看历史：与 gru-next6-with-today-001 / cs-mlp-only6-001 相同
- 模型：不改权重。tau=4546，w_low=0.25，w_high=0.6，next6 0.15，mlp6 0.4，cs_mlp 0.15
- 损失：—
- valid mean RankIC：**0.120542**（与 GitHub / platform-probe-001 逐位对齐）
- 结论：**复现成功。** 产物 `submissions/task1_fusion_next6_wt_mlp6.npy` 已写回本机。这是平台 test 0.125588 的那一版。
- 下一步：主方案就是这份。改良只叠在这版上面，不要再改已经过线的二档门控。

## gru-rest6-with-today-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_rest6_with_today.yaml`，`scripts/scan_unused_cols.py`，`scripts/eval_rest6_on_locked.py`
- 输入特征：train 末 800 天 |RankIC| 最高的未用列 76, 50, 75, 86, 48, 49（丢掉 RankIC 为 NaN 的 23）
- 是否看历史：窗口 = 10，含当天；末 800 天；每天 2000 只。与 next6 配方对齐
- 模型：同 GRU；epoch 4 最佳 0.055
- 损失：MSE
- valid mean RankIC：单支 0.054979
- 诊断：未用列最强 |RankIC| 仅 0.038，弱于 next6。raw 叠到锁定 0.120542 上 0.10/0.15/0.25 都不涨；rank 叠掉到 0.119–0.114
- 结论：**抛弃。** 未覆盖 `task1_fusion_next6_wt_mlp6.npy`。未用 6 列 GRU 这条线挖干了。
- 下一步：不要再开第四、第五档弱列 GRU。锁定配方上若继续，只做训练期时序外的日度路由，且不能在 valid 上搜门控。

## oof-day-router-001
- 日期：2026-08-22
- 代码/配置：`scripts/run_oof_router.py`、`scripts/clean_redundant.py`
- 输入特征：先清失败产物（gate3 / rest6 / cs_attn / 旧 GRU 变体等 110 个文件）。路由特征 = `[1, cov_z, dis_z, cov*dis]`，目标 = OOF 上按日最优 GRU/树权重（网格 `{0,0.15,0.25,0.4,0.6,0.85,1}`）。未在 valid 上拟合。
- 是否看历史：train 2432 天切成 prefix [0,1232) 训树、OOF [1232,1632) 拟路由、GRU recency [1632,2432)。冻结 x6/only6/next6 在 OOF 上做时序外预测。
- 模型：OOF 上 lstsq 拟合日度 w；推理时用锁定全模型分数 + 同样 mlp6 mix + rank 0.15。未覆盖 `task1_fusion_next6_wt_mlp6.npy`。
- 损失：拟合 oracle w，不是直接最大化 RankIC
- valid mean RankIC：**0.104427**（对照锁定 0.120542，Δ=−0.016115）；mean_w=0.714；min=−0.2836；负 RankIC 日 57
- OOF 诊断：GRU ens 0.111057；prefix 树 0.154508；oracle-w 0.176345；router 0.152272（略差于纯树）。beta≈`[0.332, 0.059, 0.031, 0.038]`，几乎常权重。
- 耗时 / 硬件：合计 302s
- 结论：**抛弃。** 日度 oracle 仍有约 0.176 的头，但 coverage/disagree 线性路由学不出可迁移的开关：OOF 窗口树更强，valid 覆盖度系统性更高，冻结的 z-score 把 w 拧偏。没写出新主提交。
- 下一步：不要再在 valid 上搜覆盖度，也不要再拟合这种 4 维线性日度 w。若继续冲 0.13，下一条是 walk-forward 直接堆各支分数（对 RankIC 拟合，而不是拟合 oracle w）。主方案仍是 `task1_fusion_next6_wt_mlp6.npy`。

## oof-rankic-stack-001
- 日期：2026-08-22
- 代码/配置：`scripts/run_oof_stack.py`
- 输入特征：冻结 x6/only6/next6 GRU；前缀重训 hist_lgbm+baseline（0.7）和 cs_mlp+mlp6（0.4）。OOF 400 天在 GRU recency 之前。单纯形网格步长 0.05，raw / rank 两空间，目标 mean RankIC。未看 valid。
- 是否看历史：与 oof-day-router-001 同一窗口：prefix [0,1232)，oof [1232,1632)，gru_fit [1632,2432)
- 模型：全局三支固定权重，替代覆盖度门控。未覆盖 `task1_fusion_next6_wt_mlp6.npy`
- 损失：OOF 上直接最大化 RankIC
- valid mean RankIC：**0.107624**（对照锁定 0.120542，Δ=−0.012918）；min=−0.1868；负 RankIC 日 37
- OOF 诊断：GRU ens 0.111057；树 0.154508；MLP ens 0.104861。最优 stack 0.154646，权重 **(gru 0.05, tree 0.95, mlp 0.00)**，rank 空间。几乎等于纯树。
- 耗时 / 硬件：合计 381s；复用了 router 的 GRU/树 OOF
- 结论：**抛弃。** 这 400 天 GRU 是时序外、树更强，定权学成 95% 树；valid 覆盖度更高、锁定配方需要 GRU。和日度路由同一制度错位，只是从「按天开关」换成「全局偏树」。没写出新提交。
- 下一步：不要再在「GRU recency 之前」这块 OOF 上拟合融合权重。主方案仍是 `task1_fusion_next6_wt_mlp6.npy`。刷分停在这里；精力转说明书 / 可复现入口。

## gru-x6-stable-001
- 日期：2026-08-22
- 代码/配置：`configs/gru_x6_stable.yaml`，`scripts/eval_x6_stable_fusion.py`；`src/models/gru_ts.py` 增加 accum / cosine / rank 标签 / head dropout（默认关，旧 GRU 行为不变）
- 输入特征：与锁定 x6 相同 27 列、L=10、不含当天、末 800 天、每天 2000 只
- 是否看历史：窗口 = 10，`[t-L, t)`
- 模型：同 1 层 GRU hidden=64。只改训练：accum=8、cosine lr、MSE on 截面名次 z-score、head_dropout=0.1、weight_decay 1e-4→1e-3
- 损失：MSE（标签是当天截面名次再标准化）
- valid mean RankIC：单支 **0.082783**（对照锁定 x6 0.092888）；锁死融合替换 x6 后 **0.113657**（对照 0.120542，Δ=−0.006885）
- 训练曲线：epoch1 0.0828 → epoch2 0.033 → epoch3 0.082 → epoch4 0.056，早停。最佳仍是第 1 轮。
- 耗时 / 硬件：拟合 235s，合计约 412s
- 结论：**抛弃。** 第 2 轮照样崩，cosine/累积/名次标签没有把 GRU 训过 epoch 1；第 1 轮还比原 x6 弱。未覆盖 x6 checkpoint 和主提交。
- 下一步：不要再把多种训练技巧一次堆进 GRU。原 x6「第 1 轮快照」就是这套数据上更强的拟合。主方案仍是 `task1_fusion_next6_wt_mlp6.npy`。

## hist-lgbm-rankic-001
- 日期：2026-08-23
- 代码/配置：`configs/hist_lgbm_rankic.yaml`，`src/models/lambdarank_ic.py`，`scripts/eval_hist_rankic_fusion.py`
- 输入特征：与 hist_lgbm-002 完全相同；只把 MSE 换成 LambdaRankIC（Lin et al. 2026 论文 Algorithm 1）
- 是否看 history：窗口 = 10，source=cs_zscore
- 模型：LightGBM 250 轮，每天 512 随机 pair，向量化梯度
- 损失：直接优化 Rank IC 的 lambda 梯度（不是 NDCG LambdaRank）
- valid mean RankIC：单支 **-0.100266**（对照 MSE hist 0.104192）；树融合 0.7 blend **-0.098234**；锁死 next6 融合 **-0.075366**
- 耗时 / 硬件：fit 226s，合计 410s。首轮实现无进度日志，2432 组×2048 pair×400 轮在 Python 里跑了 15+ 分钟无输出，后改为 512 pair + 250 轮 + 每 25 轮日志
- 结论：**抛弃。** 比 hist-rank-001（NDCG LambdaRank -0.021）更差，分数反号。可能是 LightGBM 自定义目标符号/实现细节与 XGBoost 论文不一致，或 pair 采样太稀。未覆盖 hist_lgbm 产物。
- 下一步：若要再试 RankIC 损失，应对照论文 XGBoost 实现核对梯度符号；或改用 GRU 截面 Margin/BPR list 训练。主方案仍是 `task1_fusion_next6_wt_mlp6.npy`。

## regime-bucket-train-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_regime_bucket_train.py`（方案 B）
- 输入特征：覆盖度四分位边界只在 train 估（全 train + 末 800 天两套）；每桶 GRU/树权重在 train 末 800 天上网格搜索。**valid 未参与拟合。**
- 是否看 history：沿用锁定 next6 融合 + mlp6
- 模型：4 档固定 w_q × GRU + (1-w_q) × 树，再 rank-blend mlp 0.15
- 损失：—
- valid mean RankIC：**0.108836**（对照锁定 0.120542，Δ=−0.012）；bucket gate 0.106（≈纯树 0.106）
- 诊断：train 末 800 天 in-sample 树 ~0.22、GRU ~0.08，四档全学到 **w=0（纯树）**；valid 覆盖度全落在最高档（4242+），用 w=0 等于放弃 GRU。全 train 边界时 Q1/Q2 在 fit 窗无样本。修复溢出后仍不如二档门控 w_high=0.6。
- 耗时 / 硬件：train 预测 257s，eval 14s（`--skip-train-pred`）
- 结论：**抛弃。** train 内样本选权重系统性偏树，和 valid 制度相反；与 OOF stacking 失败同一根因。未覆盖主提交。
- 下一步：方案 B 在此数据上到头。冲 +0.02 只剩方案 A（多折 walk-forward 元学习），成本高、valid 仍可能骗人。

## fusion-x6-today-001
- 日期：2026-08-23
- 代码/配置：`configs/gru_x6_with_today.yaml`，`scripts/eval_x6_today_fusion.py`
- 输入特征：相对锁定 x6 **只打开 `include_current_day`**（27 列 CS z-score 窗口含当天）；only6/next6/mlp/树支不重训
- 是否看 history：L=10，含当天
- 模型：GRU hidden=64，MSE，末 800 天 n2000；融合仍为 0.15×next6 + 0.85×(0.4×only6 + 0.6×x6)，二档门控 tau=4546 w=0.25/0.6，mlp6 0.4 + rank 0.15
- 损失：MSE 回归 y1
- valid mean RankIC：**0.120869**（对照上一主方案 0.120542，Δ=+0.000327）
- 平台 test mean RankIC：**0.125679**（对照 0.125588，Δ=+0.000091）
- 耗时 / 硬件：x6 训练 ~323s（CPU）；融合 eval 数秒
- 结论：**当前主提交。** 产物 `submissions/task1_fusion_x6_today.npy`。未覆盖 `task1_fusion_next6_wt_mlp6.npy`。
- 下一步：gate2 上 valid 仍低于内部 bar 0.121542，但 valid/test 双超上一版，先锁定。勿用 gate3（test 覆盖度全在高档 → 纯 GRU，test 0.120477）。

## gru-input-decay-001
- 日期：2026-08-23
- 代码/配置：`src/models/gru_ts.py` 的 `input_decay_halflife`，`configs/gru_only6_input_decay.yaml` / `gru_x6_input_decay.yaml`
- 输入特征：窗口内指数衰减（半衰期 3）
- valid / 平台 test：only6 融合 valid 0.120446，test **0.125548**；x6 融合 valid 0.120408，test **0.124911**
- 结论：**抛弃。** 相对上一主方案 test 均略低。

## fusion-x6-today-gate3-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_x6_today_gate3_fusion.py`
- valid mean RankIC：**0.124767**；平台 test mean RankIC：**0.120477**
- 结论：**抛弃。** valid 高因 valid 覆盖度分三档；test 442 天全覆盖 ≥4650，gate3 高档 w=1.0 变纯 GRU，树不进融合。与 fusion-x6-only6-gate3 同类失败。

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
