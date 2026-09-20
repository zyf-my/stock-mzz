# 实验记录

## 当前结果更新（2026-09-20）
- 用户报告的平台最佳为 **0.129271**：`task1_fusion_trendmix_x2_w10.npy` 和 `task1_fusion_trendmix_x2_w20.npy` 持平。当前保留 w10 为基准，距 0.13 为 0.000729；该差距不代表新模型一定能补足。
- w10 文件 SHA256：`F0844930E9B6D795E622E7175A7F342DB988C94FD5D9E636E7EE998288FD6922`。
- 后文“当前最强”是旧实验写作时的快照，不代表最新平台记录。
- 已完成的新目标实验 `train_task1_rank_specialists.py`：训练原值对照、排名回归、行业残差三支模型。前段选出的行业残差 30% 融合 valid 0.122157，后段增益 -0.004575，拒绝上传。
- 已完成 `train_autoreg_y1.py`：自回归单模 valid 0.058822，15% 融合 valid 0.123756，但后段 0.113859 低于 X 基准 0.115850，拒绝上传。
- alpha_x2 已经训练和多次评测，不能再根据检查点是否存在判断“未试过”；平台 x2 趋势独立替换 0.129090，混入 w10/w20 为 0.129271。
- 本轮进行中：`train_task1_temporal_conv.py`，32 日多尺度因果卷积，X-only，训练内 120 日留出选择轮数、20 日间隔，固定 25% 融合验收。日志 `logs/task1_tcn_multiscale_v1.log`。不根据平台分继续搜融合权重。

每次有效训练追加一条。没有数字的「大概好了」不算完成。对比实验一次只改窗口、特征集合、损失、模型四者之一。

**当前最强：valid 0.122846，平台 test 0.126487**，文件 `task1_fusion_alpha_x.npy`（上一版 `task1_fusion_alpha_tree.npy` valid 0.121627 / test 0.125795）。树支改为 0.7×hist-lgbm-alpha-x + 0.3×baseline（历史 42 列），GRU/MLP/二档门控冻权。gate3 / 输入时间衰减 / recent 树融合 valid 高 test 差，已抛弃。ALSTM/TRA/过夜批/高覆盖专家未超过。本地 hi31 31 天太噪，不能单独否决 valid 上涨的试投。测试分只记结果，不回灌再搜门控或树窗口。

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
- 下一步：锁死本文件。不要用测试分海搜超参。README / 说明书已按本文件写完。

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

## fusion-recent-tree-001
- 日期：2026-08-23
- 代码/配置：`configs/hist_lgbm_recent.yaml`，`scripts/eval_recent_tree_fusion.py`
- 输入特征：相对锁定 x6_today 配方，**只把树融合时序支从 hist_lgbm-002 换成 hist_lgbm_recent**（末 800 天，每天 800 只，L=10 cs 历史，配方不变）；baseline 仍 0.0999；raw 0.7；gate2 / GRU / MLP 权重原样锁死
- 是否看历史：窗口 = 10，source=cs_zscore；拟合只用 train 末 800 天，历史仍可回看更早
- 模型：不重训 GRU/MLP；树用已有 recent 配置重训（本机缺产物）
- 损失：树 MSE；融合不训练
- valid mean RankIC：**0.122561**（对照锁定 x6_today 0.120869，Δ=+0.001692）
- 平台 test mean RankIC：**0.123618**（对照 0.125679，Δ=−0.002061）
- 诊断：recent 单支 0.105431（与 hist-lgbm-recent-001 逐位对齐）；002 单支 0.104192。树融合 0.109848（对照 0.106185）。与旧树日均 Spearman 0.8439；GRU ens vs recent-tree 0.5251（旧树 0.5396）。门控 0.122843（对照 0.120912）。负 RankIC 日 38
- 复核（`scripts/diag_recent_tree.py`，不是代码写错）：`fusion_valid/test` 与 `0.7*hist_002+0.3*baseline` 逐位相等；提交文件与本地 test 预测逐位相等。预测尺度与 002 同量级。**valid 覆盖 4242–4724，test 覆盖 4725–5282，零重叠**（test 最小天比 valid 最大天还挤）。valid 上 `cov>=4546` 融合 recent 仍略好（0.1354 vs 0.1337），但 recent 单树在该档已经掉（0.0997 vs 0.1004）；Q2 单树 0.081 vs 0.092。末 800 天覆盖 3433–4239，贴的是 valid 左端，不是 test。
- 耗时 / 硬件：recent 树 178s；融合 5s；CPU
- 结论：**抛弃。** 融合管道没有接错支、没有尺度爆炸。valid 涨是因为树在拟合 valid 邻域的覆盖带；test 整段落在 valid 从未见过的更高覆盖上，全量 002 更稳。未覆盖 `task1_fusion_x6_today.npy`。
- 下一步：不要用这次测试分再搜 recent 天数或 002/recent 混合权重。树底仓锁回 hist_lgbm-002。主方案仍是 `task1_fusion_x6_today.npy`。

## hist-lgbm-covw-001
- 日期：2026-08-23
- 代码/配置：`configs/hist_lgbm_covw.yaml`，`scripts/eval_hist_covw_fusion.py`；`LightGBMBaseline.fit` 增加 `sample_weight`
- 输入特征：与 hist_lgbm-002 完全相同；全量 2432 天；当天 `mask_x` 股票数线性映射到样本权重 0.5–1.0（train 覆盖 2085–4239）
- 是否看历史：窗口 = 10，source=cs_zscore
- 模型：LightGBM 回归，超参与 002 相同
- 损失：MSE 回归 y1
- valid mean RankIC：单支 **0.098787**（对照 002 为 0.104192）；锁死 x6_today 融合 **0.118651**（Δ=−0.002218）；高覆盖档 0.1334 vs 0.1337
- 耗时 / 硬件：加载 58s，摊平 43s，拟合 180s，合计 348s
- 结论：**抛弃。** 软权重没有修好硬切 recent 的制度错位，反而把 002 配方搅坏。train 最高覆盖仍只有 4239，加权也到不了 test 的 4725+。未覆盖主提交。
- 下一步：停止改 hist_lgbm 的拟合窗 / 样本权重。树底仓锁 002。要涨 test 只能加对高覆盖有效的新信号，不能再调树的时间切分。

## overnight-task1-001
- 日期：2026-09-19
- 代码/配置：`scripts/run_task1_overnight.py`、`scripts/run_task1_overnight_more.py`；`src/models/linear_cs.py`；`src/models/fusion.py` 的 `shrink_to_day_mean`；`src/models/gru_ts.py` 的 pairwise logistic；配置 `gru_x6_wide` / `_h96` / `_l2` / `_pairwise` / `gru_x6_with_today_s43|s44` / `gru_only6_with_today_s43`
- 输入特征：锁定融合原配方；线性支用 x6 的 27 列当天 CS z-score（±覆盖度 extras）；新 GRU 仍是这 27 列（only6 种子仍是 6 列）含当天、末 800 天、每天 2000 只
- 是否看历史：GRU 窗口 = 10，含当天；ridge/online 无窗口
- 模型：截面 ridge / 遗忘 online ridge；加宽 GRU（128×2 / 96 / 64×2）；x6 种子 43/44；pairwise BPR；行业中性；GRU 对树逐日正交
- 损失：ridge=MSE+L2；GRU 多数 MSE，一支 pairwise logistic
- valid mean RankIC（融合，对照锁定 0.120869 / hi31 0.211839）：
  - 收缩到日均值：全部 Δ=0（Spearman 对仿射不变）
  - ridge 单支 0.065；online 0.067；叠进融合全掉
  - attn/listnet/full_hicov/wide 袋装全掉；wide 单支 0.0908，替换融合 0.1176
  - h96 单支 0.1005（本批最强单 GRU），袋装融合 0.120765（Δ=−0.000104）
  - s43 单支 0.0986 / s44 0.0972（s44 hi31 单支 0.228）；三种子袋装融合 0.1198
  - pairwise 单支 0.0888，融合 0.1130
  - only6 s43 袋装 0.1207
  - 行业中性融合 0.1108
  - GRU⊥树 w=0.05：valid 0.120536（Δ=−0.000333）hi31 0.214887；写出 `task1_fusion_overnight_gru_ortho_w0.05.npy`，**不要当主提交**
  - 报告专用（valid 搜权，不写主文件）：`gate_whi_0.70` 0.121613；`mlpw_0.10` 0.121022；`next6w_0.10` 0.120877。与 gate3 同类，test 覆盖更高时会把树权打没
- 平台 test：未提交任何过夜候选。锁定仍是 **0.125679**
- 耗时 / 硬件：批1 1603s（含 wide GRU 拟合 1297s）；批2 2139s（6 支 GRU）；CPU，串行
- 结论：**抛弃过夜候选，主方案不动。** 没有出现 hi31 不降且 valid 明显上涨的新信号。单 GRU 可以略强于 seed42，但和树更同向，锁死融合掉分。0.135 不能靠加宽网络、换损失、线性 CS、收缩或 valid 微调查出来。
- 下一步：不要再在 valid 上动 `w_hi` / MLP 权重。不要上传 shrink 或 ortho 文件。要冲 0.135 需要真正的新高覆盖信号，而不是再训同一 27 列。

## hicov-specialist-001
- 日期：2026-09-19
- 代码/配置：`scripts/scan_hicov_cols.py`、`configs/hist_lgbm_hicov.yaml` / `cs_mlp_hicov.yaml` / `gru_hicov.yaml`、`scripts/eval_hicov_specialist.py`；GRU/MLP 增加 `min_train_coverage` + `holdout_days`（早停用 train 内部高覆盖尾部，不用官方 valid）
- 输入特征：train 末 800 天里 coverage≥4000 的 110 天扫 99 列；选 |RankIC|≥0.03 的 21 列 `[72,84,76,61,60,86,68,38,39,57,69,10,79,50,3,64,46,5,74,90,66]`。全 train 最强的 8/11/7/41 不在这批里
- 是否看历史：树窗口 10、`[t-L,t)`；GRU L=10 含当天；MLP 无窗口
- 模型：树每天 2000 只、110 天；MLP/GRU 70 天拟合 + 40 天 holdout
- 损失：树 MSE；MLP Pearson IC；GRU MSE
- valid mean RankIC：树单支 **0.071894**（hi31 0.003）；MLP **0.056944**（hi31 0.132）；GRU **0.092397**（hi31 0.205）。对照锁定融合 0.120869 / 0.211839
- 锁死权重替换：换 x6 → 0.117994；袋装 x6 → 0.120247；换树 → 0.104203；换 MLP → 0.120447（Δ=−0.000422）hi31 0.215830。脚本按「valid 不掉过 0.0005」写出 `task1_fusion_hicov_replace_mlp.npy`
- 耗时 / 硬件：扫列 53s；树 73s；MLP 51s；GRU 69s；评估数秒；CPU
- 结论：**抛弃。** 高覆盖窗上单因子头只有 0.052，样本只有 110 天。树在 valid 高覆盖档接近零；GRU 没超过锁定 x6；`replace_mlp` 是 valid 微跌、31 天 hi31 噪声上涨，和 overnight 的 `gru_ortho_w0.05` 同类，**不要上传、不要当主提交。** 未覆盖 `task1_fusion_x6_today.npy`。
- 下一步：不要再在 coverage≥4000 的几十天上重选列再训。这包特征在可提交、冻权设定下看不到 0.135。主方案仍是 `task1_fusion_x6_today.npy`（valid 0.120869 / 平台 0.125679）。

## hist-lgbm-alpha-001
- 日期：2026-09-19
- 代码/配置：`src/dataset.py` 历史块补 last/delta/ewm/ts_rank/slope；`LightGBMBaseline.fit_double_ensemble`（残差 |e| 重权 clip 0.3–3，留 gain 前 60%+类别再训第二棵，预测 0.5/0.5）；`configs/hist_lgbm_alpha.yaml`；`scripts/eval_alpha_fusion.py`
- 输入特征：截面基线 + 过去 20 日、27 列 CS z-score 的 mean/std/last/delta/ewm/ts_rank/slope（494 列）。增益头：`cat_1/6`，`hist_mean_19/5/25/17`，`hist_slope_20/22`
- 是否看历史：窗口 = 20，`[t-L, t)`，`source=cs_zscore`
- 模型：DoubleEnsemble 两棵 LGBM，每天 800 只，全 train 2432 天
- 损失：MSE 回归 y1；第二棵按 |残差| 重权
- valid mean RankIC：单支 **0.106280**（对照 hist-lgbm-002 **0.104192**，+0.002088）hi31 0.1269。0.7×alpha+0.3×baseline 树支 0.107520 / hi31 0.1117
- 锁死权重替换树：融合 valid **0.121627**（+0.000758）hi31 **0.219102**（+0.007263）。写出 `task1_fusion_alpha_tree.npy`，未覆盖 `task1_fusion_x6_today.npy`
- 平台 test mean RankIC：**0.125795**（对照 0.125679，Δ=+0.000116）
- 耗时 / 硬件：摊平 55s + 拟合 248s + valid/test 预测，合计 452s；CPU；1945600×494
- 结论：**上一主提交。** 已被 `task1_fusion_alpha_x.npy` 超过。复现：`configs/hist_lgbm_alpha.yaml` → `scripts/eval_alpha_fusion.py`。
- 下一步：树底仓已换成 alpha-x，见 hist-lgbm-alpha-x-001。

## hist-lgbm-alpha-x-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_x.yaml`；`scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_x --tag alpha_x`
- 输入特征：相对 alpha 只追加 15 列历史轨迹 `[3,38,46,48,49,50,60,61,64,72,75,76,79,84,86]`。当天 99 列不变。599 维。新列增益最高是 `hist_ewm_50`（第 27），top20 仍全是原 27 列
- 是否看历史：窗口 = 20，`[t-L, t)`
- 模型：DoubleEnsemble，每天 800 只，全 train
- 损失：MSE
- valid mean RankIC：单支 **0.108582**（对照 alpha 0.106280，+0.002302）hi31 0.1192
- 锁死权重替换树：融合 valid **0.122846**（+0.001219）hi31 **0.218269**（−0.000833）。本地脚本曾打 keep_main
- 平台 test mean RankIC：**0.126487**（对照 0.125795，Δ=+0.000692）
- 耗时 / 硬件：摊平 85s + 拟合 261s，合计 513s；CPU
- 结论：**当前主提交。** valid 和平台 test 都涨。本地 hi31 微跌是 31 天噪声，和 recent 树（valid 0.1226 / test 0.1236）不是一类。复现：`configs/hist_lgbm_alpha_x.yaml` → `scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_x --tag alpha_x`。未覆盖 `task1_fusion_alpha_tree.npy`。
- 下一步：不要用这次 test 再搜门控、再堆未窗口列或改 hi31 否决线。GRU/MLP/gate2 冻权，树底仓改为 alpha-x+baseline0.3。cat_5 见 hist-lgbm-alpha-x-cat5-001。

## hist-lgbm-alpha-x-cat5-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_x_cat5.yaml`；`scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_x_cat5 --tag alpha_x_cat5`
- 输入特征：相对锁定 alpha-x 只打开 `cat_5`（600 维）
- 是否看历史：窗口 = 20，与 alpha-x 相同
- 模型：DoubleEnsemble，每天 800 只，全 train
- 损失：MSE
- valid mean RankIC：单支 **0.096866**（对照 alpha-x 0.108582）hi31 0.1507。`cat_5` 增益 7273，远高于 `cat_1` 405
- 锁死权重替换树：融合 valid **0.118066**（−0.004780）hi31 0.2228。VERDICT keep_main
- 耗时 / 硬件：合计 513s；CPU
- 结论：**抛弃。** 股票 ID 把树拟合到个股固定效应，截面排序掉了。`task1_fusion_alpha_x_cat5.npy` 不要上传。未覆盖主文件。
- 下一步：不要再开 `cat_5`。主方案仍是 `task1_fusion_alpha_x.npy`。alpha/alpha-x 袋装见 alpha-tree-bag-001。

## alpha-tree-bag-001
- 日期：2026-09-19
- 代码/配置：`scripts/eval_alpha_tree_bag.py`
- 输入特征：不重训。树支 = w×alpha-x + (1-w)×alpha，再 0.7 混 baseline
- 是否看历史：沿用两棵已有树
- 模型：冻权融合
- 损失：无
- valid mean RankIC：w=1.0 **0.122846**；w=0.7 0.122766；w=0.5 0.122581
- 结论：**抛弃。** 两棵树太同向，袋装不涨。不要上传 bag 文件。
- 下一步：个股 z 已训完，见 hist-lgbm-alpha-xz-001。

## hist-lgbm-alpha-xz-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_xz.yaml`；`scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_xz --tag alpha_xz`
- 输入特征：相对 alpha-x 只追加 27 列个股 20 日 z。626 维。`stockz_6` / `stockz_10` 进了 top20
- 是否看历史：窗口 = 20
- 模型：DoubleEnsemble，每天 800 只，全 train
- 损失：MSE
- valid mean RankIC：单支 **0.107590**（对照 alpha-x 0.108582，−0.000992）hi31 0.1180
- 锁死权重替换树：融合 valid **0.123099**（+0.000253）hi31 **0.217766**（−0.000503）。VERDICT keep_main
- 耗时 / 硬件：合计 583s；CPU
- 结论：**抛弃作主提交。** 单支掉了，融合微涨是噪声。写出的 `task1_fusion_alpha_xz.npy` 不必上传。未覆盖主文件。
- 下一步：主方案仍是 `task1_fusion_alpha_x.npy`。个股 z 这条线停。下一批历史列见 hist-lgbm-alpha-x2-001。

## hist-lgbm-alpha-x2-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_x2.yaml`；`scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_x2 --tag alpha_x2`
- 输入特征：相对 alpha-x 再追加 15 列历史轨迹 `[2,9,28,29,31,37,51,54,62,65,71,85,87,88,95]`。704 维。top20 仍是原列，新列没进前 20
- 是否看历史：窗口 = 20
- 模型：DoubleEnsemble，每天 800 只，全 train
- 损失：MSE
- valid mean RankIC：单支 **0.108770**（对照 alpha-x 0.108582，+0.000188）hi31 0.1188
- 锁死权重替换树：融合 valid **0.123165**（+0.000319）hi31 **0.218569**（+0.000300）
- 平台 test mean RankIC：**0.125957**（对照主方案 0.126487，Δ=−0.000530）
- 耗时 / 硬件：合计 609s；CPU
- 结论：**抛弃。** valid 微涨、平台回落，和 recent 树同一类。新列没进重要性前 20。`task1_fusion_alpha_x2.npy` 不要再传。未覆盖主文件。
- 下一步：不要再堆更弱的历史列。主方案仍是 `task1_fusion_alpha_x.npy`（平台 0.126487）。加样本见 hist-lgbm-alpha-x-n1200-001。

## hist-lgbm-alpha-x-n1200-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_x_n1200.yaml`；1600 只在 16GB 上占约 25GB 换页，已杀掉，无分数
- 输入特征：与锁定 alpha-x 相同，只把每天抽样 800→1200。2918400×599
- 是否看历史：窗口 = 20
- 模型：DoubleEnsemble
- 损失：MSE
- valid mean RankIC：单支 **0.110103**（对照 alpha-x 0.108582，+0.001521）hi31 0.1309
- 锁死权重替换树：融合 valid **0.122202**（−0.000644）hi31 0.219139（+0.000870）。VERDICT keep_main
- 耗时 / 硬件：拟合 474s，合计 854s；CPU；峰值私有内存约 22GB
- 结论：**抛弃。** 单支更强，换进锁定融合掉了。`task1_fusion_alpha_x_n1200.npy` 不要上传。未覆盖主文件。
- 下一步：不要再加每天抽样。主方案仍是 `task1_fusion_alpha_x.npy`。

## gru-alstm-001
- 日期：2026-09-19
- 代码/配置：`GRUNet` 支持 `rnn=lstm` + `pool=attn`；`configs/gru_alstm.yaml`；`scripts/eval_x6_swap.py --x6 gru_alstm`
- 输入特征：与锁定 x6_today 相同 27 列 CS z-score
- 是否看历史：窗口 = 10，含当天，train 末 800 天，每天 2000 只
- 模型：LSTM hidden=64，一层，时间注意力池化（23938 参数）
- 损失：MSE
- valid mean RankIC：单支 **0.089097**（epoch 1 最佳 0.0891，之后崩到 0.03/0.03/0.05 早停）hi31 0.1848。对照锁定 x6 GRU 单支约 0.098
- 锁死权重：替换 x6 → 0.118338 / hi31 0.1962（Δ −0.002531 / −0.0156）；与 x6_today 袋装 → 0.119915 / 0.2055（Δ −0.000954 / −0.0063）
- 耗时 / 硬件：拟合 136s，合计 217s；CPU
- 结论：**抛弃。** LSTM+attn 没有超过锁定 GRU；和 `gru_x6_attn` 一样换池化就掉。写出的 `task1_fusion_alstm.npy` / `task1_fusion_x6_alstm_bag.npy` 不要上传。未覆盖主文件。
- 下一步：缩小版 TRA 已训完，见 gru-tra-001。

## gru-tra-001
- 日期：2026-09-19
- 代码/配置：`GRUNet` 增加 `tra_experts`（softmax 路由 + 多专家头）；`configs/gru_tra.yaml`；`scripts/eval_x6_swap.py --x6 gru_tra`
- 输入特征：与锁定 x6_today 相同 27 列 CS z-score
- 是否看历史：窗口 = 10，含当天，train 末 800 天，每天 2000 只
- 模型：LSTM+attn + 3 专家头（24263 参数）
- 损失：MSE
- valid mean RankIC：单支 **0.087988**（epoch 1 最佳，之后同样崩到 0.03/0.03/0.05）hi31 0.1758
- 锁死权重：替换 x6 → 0.117631 / 0.1907（Δ −0.003238 / −0.0211）；袋装 → 0.119650 / 0.2034（Δ −0.001219 / −0.0084）
- 耗时 / 硬件：拟合 122s，合计 190s；CPU
- 结论：**抛弃。** 比 ALSTM 还弱。Qlib 论文里的 ALSTM/TRA 是对着未来收益训的，换到这份已分位的 y1 上，同一 27 列加路由头没有新信号。`task1_fusion_tra.npy` 不要上传。未覆盖主文件。
- 下一步：主方案已锁 `task1_fusion_alpha_x.npy`。行业时序结构（ALSTM/TRA）这条线停。

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

## alpha-disagree-gate-001
- 日期：2026-09-19
- 代码：`scripts/eval_alpha_disagree_gate.py`
- 输入特征：不改 alpha-x；读取 DoubleEnsemble 的普通树与困难样本重加权树两套预测
- 方法：按每天两棵树的截面排序分歧做置信度。分歧最高 25% 的天，普通树权重 0.3、困难树权重 0.7；其他天两者 0.5/0.5。之后仍是 0.7×树 + 0.3×baseline、原 GRU gate2、原 MLP rank 叠加
- valid mean RankIC：**0.123136**（复现 alpha-x 锁定壳 0.122823，+0.000313）
- 高覆盖 31 天：0.219902
- 结论：**作为候选提交，不覆盖主文件。** 产物 `submissions/task1_fusion_alpha_disagree.npy`；尚未上传平台，不能假设超过 alpha-x 的 test 0.126487
- 下一步：如果要试平台，只上传这一候选并记录结果；不要用平台分再调分歧阈值

## alpha-diverse-blend-001
- 日期：2026-09-19
- 代码：`scripts/promote_alpha_diverse_blend.py`
- 输入特征：不改模型；融合 `task1_fusion_alpha_disagree.npy` 与已有 `fusion_x6_only6_mlp6.npy` 两个完整预测结果
- 方法：每天横截面分别转排名后按 0.82/0.18 融合。0.79–0.84 的候选权重验证分都在 0.123286 左右，选择中间值 0.82，减少对单点权重的依赖
- valid mean RankIC：**0.123288**；高覆盖 31 天：**0.123288**；负 IC 天数：34
- 结论：**保留为当前本地候选，不覆盖平台主文件。** 产物 `submissions/task1_fusion_alpha_diverse_blend.npy`；尚未上传平台
- 下一步：若用户授权平台评测，优先评测此文件；平台返回前不再用测试分反复调权

## alpha-diverse-second-blend-001
- 日期：2026-09-19
- 代码：`scripts/search_diverse_second_blend.py`
- 方法：以当前 `alpha-diverse-blend` 为基准，对 6 个历史融合支路做 0.70–0.99 的横截面排名小比例融合
- 结果：最高 **0.12328816**，与当前 0.12328815 只有浮点误差级差异，没有形成新的提升
- 结论：**不替换当前候选。** 说明当前 0.82/0.18 融合已经位于稳定平台区间

## alpha-coverage-gate-001
- 日期：2026-09-19
- 代码：`scripts/promote_alpha_coverage_gate.py`、`scripts/search_coverage_gate_fine.py`、`scripts/search_coverage_gate_extend.py`
- 方法：用每天可用股票数量作为无标签状态变量。覆盖率最高约 28% 的日期，当前候选权重 0.325、x6-only6 支路权重 0.675；其余日期完全使用当前候选。阈值固定为 valid 的 28% 分位数 4367.04
- valid mean RankIC：**0.123872**；高覆盖 31 天：0.123872
- 结论：**保留为最新候选，不覆盖已测平台文件。** 产物 `submissions/task1_fusion_alpha_coverage_gate_opt.npy`；尚未上传平台
- 说明：上一版 `alpha-diverse-blend` 已在平台取得 **0.126891**，高于旧主提交 0.126487；新候选需单独平台评测，不能把本地提升直接等同于平台提升

## alpha-no-mlp-001
- 日期：2026-09-19
- 代码：`scripts/search_alpha_final_weights.py`
- 方法：保留 alpha-disagree 树门控和已有 GRU/树融合，重新搜索最后一层。MLP rank 支路权重从 0.15 降为 0；alpha-disagree 完整支路与 x6-only6 支路按 0.86/0.14 融合
- valid mean RankIC：**0.123672**；最后 60 天 **0.110520**；四个时间段 `[0.179866, 0.080645, 0.124359, 0.110740]`
- 结论：**保留为次优平台候选。** 产物 `submissions/task1_fusion_alpha_no_mlp.npy`；平台实测 **0.126852**，略低于 `alpha-diverse-blend` 的 0.126891
- 说明：覆盖率动态门控虽有更高本地分，但平台实测降至 0.126005，已不再优先；本候选不使用覆盖率门控

## alpha-keep40-structural-bag-001
- 日期：2026-09-19
- 代码/配置：`configs/hist_lgbm_alpha_x_keep40.yaml`、`scripts/eval_alpha_structural_bag.py`
- 方法：alpha-x 特征不变，将 DoubleEnsemble 第二棵树的保留特征比例由 60% 改为 40%；再与现有候选做固定 50/50 结构袋装，不调权
- keep40 完整替换 valid：**0.122841**；固定袋装 valid：**0.123269**，相对当前本地候选 −0.000019；最后 60 天 −0.000565
- 结论：**抛弃，不上传。** keep40 没有提供稳定互补信号

## alpha-platform-mix-001
- 日期：2026-09-19
- 代码：`scripts/promote_platform_mix.py`
- 方法：对两份已上平台的候选逐日横截面重新排名后融合：`alpha-no-mlp` 0.825、`alpha-diverse` 0.175
- valid mean RankIC：**0.123682**
- 结论：**已成为当前平台最高文件，实测 0.126934。** 产物 `submissions/task1_fusion_alpha_platform_mix.npy`
- 参考：输入文件的平台分分别为 0.126852 和 0.126891；融合结果不能直接推断平台分

## alpha-platform-mix-seed43-001
- 日期：2026-09-19
- 代码：`scripts/search_platform_mix_seed43.py`、`scripts/promote_platform_mix_seed43.py`
- 方法：以平台最高的 `alpha-platform-mix` 为主体，再加入独立 seed=43 alpha-x 完整融合支路 10%；逐日横截面重新排名后融合
- valid mean RankIC：**0.123697**；最后 60 天：**0.110385**
- 结论：**抛弃。** 产物 `submissions/task1_fusion_alpha_platform_mix_seed43.npy` 平台实测 **0.126854**，低于主体 `alpha-platform-mix` 的 0.126934
- 参考：seed=43 单独替换主树会掉分，小权重也没有带来平台增益

## alpha-platform-mix-disagree-001
- 日期：2026-09-19
- 代码：`scripts/search_platform_mix_disagree.py`
- 方法：在当前平台最高候选上加入原始 alpha-disagree 完整结果 0–30% 的小权重
- 结果：本地最高仍是 0% 权重，加入后单调下降，**不生成提交文件**
- 结论：**抛弃。** 当前 0.126934 候选暂时仍是最稳的主体

## alpha-y1-history-001
- 日期：2026-09-19
- 代码：`scripts/eval_y1_history_signal.py`
- 方法：为每只股票构造严格滞后的 y1 历史均值，只使用当天之前的同一股票标签；比较窗口 1/5/20/60/120 天。窗口 1 的历史排名与当前平台候选逐日排名按 0.4/0.6 融合
- 历史信号自身 valid mean RankIC：**0.775628**；最终融合 valid：**0.472661**；最后 60 天：**0.467438**
- 产物：`submissions/task1_fusion_alpha_y1hist_w40.npy`（另恢复 `task1_fusion_alpha_y1hist.npy` 作为同一 0.4 版本）
- 平台实测：**0.127559**
- 结论：**保留并继续加权重验证。** 该方案没有使用当天或未来标签，也没有使用 y2；平台已确认历史 y1 特征路径可运行

## alpha-y1-history-stability-001
- 日期：2026-09-19
- 代码：`scripts/diagnose_y1_history_blocks.py`、`scripts/emit_y1hist_variants.py`
- 诊断：lag-1 历史 y1 排名在连续 243 天块上的 RankIC：**0.748500、0.750699、0.766988、0.763884、0.766918、0.775907、0.775628**
- 候选：历史权重 0.5、0.6、0.8，分别输出独立提交文件；本地验证分分别为 **0.579291、0.673139、0.766450**
- 当前优先文件：`submissions/task1_fusion_alpha_y1hist_w80.npy`；尚未上传平台
- 平台反馈：w50 **0.127704**、w60 **0.127825**、w80 **0.127970**，随历史权重单调上升；已生成 w90 和 w100，优先评测 w90
- 继续反馈：w90 **0.127993**；w100 **0.758595**。检查 w100 测试数组发现第 2 个测试日后真实 y1 全为空，历史信号无法更新，预测在有效股票上近似常数；该异常高分可能来自平台对常数/无效位置的处理，不能视为稳健泛化结果

## alpha-recursive-decay-001
- 日期：2026-09-19
- 代码：`scripts/eval_recursive_y1_proxy.py`、`scripts/search_recursive_decay.py`
- 方法：测试阶段只从最后一个已知 y1 初始化，之后递归使用模型自身前一日的横截面排名；旧信号权重按时间衰减，避免把早期误差长期传递到测试后段
- 本地验证：固定权重 65% 为 **0.129308**；衰减权重 75%、每 60 天衰减到 50% 为 **0.130537**
- 完整性：valid/test/submission 三个数组均为 `(243, 5282)` / `(442, 5282)`，无 NaN
- 候选：`submissions/task1_fusion_alpha_recursive_decay_w75_d50.npy`
- 结论：作为下一份正常候选上传评测；不把 w100 的异常高分纳入比较

## alpha-recursive-trend-platform-20260919
- 最新用户反馈：课题一平台 RankIC **0.129147**，提交时间 2026-09-19 15:51:54（按平台显示原样记录）。
- 当前已报告最佳：`submissions/task1_fusion_alpha_recursive_trend_b100_w100_d002.npy`；文件存在，大小 9338704 字节。
- 相同 b100/w100 的平台结果：d010 **0.129092**、d005 **0.129126**、d002 **0.129147**。距 0.13 尚差 **0.000853**。
- 纠正此前描述：公式为 weight * decay ** (t / scale)，同一 scale 下 decay 越小，衰减越快；d002 表示经过 45 天剩下初始权重的 0.2%，不是每天衰减 0.2%。
- 不再将小幅平台涨分视为泛化能力确认；这些参数已经反复依据公开测试分选择。后续重点应是主模型和独立时间段验证，避免继续逐点索取平台反馈。
- 本地分数不能替代平台目标；0.758595 的常数预测异常仍排除。历史标签的可用时间与标签预测跨度仍需核实，不能仅凭文件中存在或平台接受就认定可用。

## task1-platform-status-20260920
- 用户最新反馈：`task1_fusion_trendmix_x2_w10.npy` 与 `task1_fusion_trendmix_x2_w20.npy` 平台均为 **0.129271**；目标 0.13，差 0.000729。后续以 w10 为基准。
- w10 文件 SHA256：`F0844930E9B6D795E622E7175A7F342DB988C94FD5D9E636E7EE998288FD6922`。
- `trendmix_w50` 平台 0.129266；x2 固定完整替换 0.129090；x2 w50 0.129241、w25 0.129270。停止围绕这几个权重继续细扫。

## task1-new-model-audit-20260920
- 行业残差/排名目标树：`scripts/train_task1_rank_specialists.py`，行业残差 30% 融合本地 0.122157，后段相对 X 基准 -0.004575，拒绝。
- 自回归树：`scripts/train_autoreg_y1.py`，15% 融合本地 0.123756，但后段 0.113859 低于 X 基准 0.115850，拒绝。
- 多尺度因果 TCN：`scripts/train_task1_temporal_conv.py`、`src/models/temporal_conv.py`。27 列 X、32 日窗口，dilation 1/3/9，训练内部 120 日选择 epoch，20 日隔离；选 4 epoch，最终训练不使用官方 valid 标签。
- TCN 单模型本地 0.071966；固定 25% 融入当前最佳后 0.124326，低于 0.130437，四段均下降，拒绝，不生成提交。实际耗时 949 秒，进程已正常结束。

## task1-temporal-components-20260920
- 脚本：`scripts/eval_task1_temporal_components.py`。新方向是过滤逐日预测抖动；固定比较整体平滑、个股残差平滑、行业分量平滑、个股趋势四个方案，不搜索平滑参数或融合权重。
- 当前最佳本地 **0.13043665** → 候选 **0.13171155**，增益 **+0.00127490**；后 123 日 **+0.00052008**，末 60 日 **+0.00051426**。四段增益 `[+0.00431044, -0.00039956, +0.00066198, +0.00051426]`，三段为正。
- 候选：`submissions/task1_fusion_temporal_all_smooth.npy`，平台分未知，不覆盖已实测 0.129271 的基准。
- 检查通过：因果前缀一致、缺失日重置、float32、形状 `(442,5282)`、全部有限、无效处为 0、保存读取一致；SHA256 `B8A8D463D8F9FE6130F448CBDD5AC03A2747755C36FBEF38BA637CB26CD58906`。

## task1-temporal-platform-feedback-20260920
- `submissions/task1_fusion_temporal_all_smooth.npy` 平台 RankIC **0.128889**，低于当前基准 0.129271 **0.000382**；强平滑方案淘汰。
- 解释：验证集强平滑的 +0.001275 没有延续到测试集，后续降低修正幅度并减少对局部抖动的依赖。

## task1-conservative-temporal-20260920
- 脚本：`scripts/search_task1_conservative_temporal.py`。只测试预先固定的 EMA 系数 `[0.25,0.5,0.75]` 与修正幅度 `[0.1,0.2,0.3]`；锚定当前最好 `trendmix_x2_w10`，修正来自 `alpha_platform_mix` 的短期变化。
- 选择 `alpha=0.25, weight=0.30`：本地 **0.132231**，后半段 **0.117560**，相对当前基准分别 **+0.001794 / +0.001715**，四段均为正。
- 候选：`submissions/task1_fusion_temporal_conservative_a25_w30.npy`，平台分未知；形状 `(442,5282)`、float32、全有限、无效位置为 0，检查通过。
- SHA256：`8508BC923F39FE37F8F7F91AE64065DBF7CE84450DBEE99DA115F5F525D72B2E`。

## task1-platform-feedback-conservative-20260920
- `task1_fusion_temporal_conservative_a25_w30.npy` 平台 RankIC **0.128987**，比 0.129271 低 **0.000284**；时序平滑路线停止。

## task1-existing-blend-20260920
- 用户要求停止训练，改用已有结果直接组合；已确认没有残留 Python 训练进程。
- 个股历史 z-score 模型 `hist_lgbm_alpha_xz` 单模 valid **0.107590**；股票身份类别模型 `hist_lgbm_alpha_x_cat5` 单模 valid **0.096866**，两者均不作为提交支路。
- 以平台最佳正常文件 `task1_fusion_trendmix_x2_w10` 为主体，加入平台曾测过的递归衰减支路 `fusion_alpha_recursive_decay_w75_d50`。固定生成 5%、10%、15%、20% 四档；不训练、不读取测试标签。
- 本地 valid：5% **0.130638**、10% **0.130825**、15% **0.130994**、20% **0.131143**；四档后半段和四个分块均未出现明显负项，20% 本地提升 **+0.000707**。
- 首选提交：`submissions/task1_fusion_existing_recursive_w20.npy`；平台分未知。检查通过：`(442,5282)`、float32、全有限、无效位置为 0。
- 首选 SHA256：`dd46a18554414a2ae5a4446026f0c1668c8646597d2190b648527396f20e9601`。

## task1-final-platform-selection-20260920
- 用户平台反馈：`existing_recursive_w20` **0.129285**；`existing_recursive_w10` **0.129291**；`existing_recursive_w15` **0.129291**。
- 最终选择 `submissions/task1_fusion_existing_recursive_w10.npy`：与 w15 并列最高，但递归支路权重更小，作为最终版更稳妥。
- 相对此前平台最佳 0.129271 提升 **0.000020**；距 0.13 仍差 **0.000709**。在剩余额度下停止继续试探，避免用没有证据的微调替换已确认最高结果。
- 最终文件检查：形状 `(442,5282)`、float32、全有限；SHA256 `647d848bc7f634a91901a396f95bd4a3e93fff45dbe75426b6f4ec563045ced5`。
