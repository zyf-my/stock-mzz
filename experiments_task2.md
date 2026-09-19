# 课题 2 实验记录（标签 `y2`）

每次有效训练追加一条。没有数字的「大概好了」不算完成。对比实验一次只改窗口、特征集合、损失、模型四者之一。

**当前最强：valid 0.088569 / test 0.091467**，文件 `submissions/task2_fusion_best.npy`（**hist bag** n240+rankic+mkt_rel + gate **w_hi=0.92** + mlp6_mix=**0** + mlp_w=**0.1** + ens 0.7/0.34）。  
配方锁定：`outputs/task2/recipes/TREE_BAG_091467.json`（取代 TREE_BAG_091445）。

硬约束：产物只进 `outputs/task2/`、`checkpoints/task2/`、`submissions/task2_*.npy`。不要用 `y1` 当 `y2` 的特征。不要覆盖课题 1 主文件。课题 1 已抛弃的（gate3、recent 树、覆盖度加权、Pearson IC、输入衰减）不要搬过来。

---

## task2-baseline-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/baseline.yaml`，`scripts/run_task2_main.py`
- 输入特征：当天 raw 99 + 全市场 z-score 99 + 行业 z-score 99 + cat [0,1,2,3,4,6,7,8]
- 是否看历史：窗口 = 0
- 模型：LightGBM 回归，超参与课题 1 截面树相同
- 损失：MSE 回归 y2；验收 RankIC
- valid mean RankIC：0.074372
- 结论：**保留作截面底仓。** 低于课题 1 同配方的 0.0999，方向对。
- 下一步：时序树与融合已跑，见下条

## task2-hist-lgbm-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/hist_lgbm.yaml`
- 输入特征：截面基线 + 过去 10 日 21 列（课题 1 的 hist 列）cs z-score mean/std/last
- 是否看历史：窗口 = 10，`[t-L, t)`
- 模型：LightGBM 回归，与课题 1 hist-lgbm-002 相同
- 损失：MSE 回归 y2
- valid mean RankIC：0.075879（对照截面 0.074372）
- 结论：**保留为时序树。** 只比截面 +0.0015，小于课题 1 上 0.100→0.104 的幅度。历史列仍是按 y1 挑的。
- 下一步：树融合

## task2-fusion-trees-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/fusion.yaml`
- 输入特征：不重训。时序支=hist 0.0759；截面支=baseline 0.0744
- 模型：valid 上搜融合；锁定 coverage_gate rank，tau=4491，w_low=0.5，w_high=0.75
- valid mean RankIC：0.078735
- 结论：**保留为树融合支。** 注意这和课题 1 锁死的 raw 0.7 不是同一规则。
- 下一步：GRU / MLP 已按课题 1 列集合重训

## task2-x6-today-fusion-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_x6_today_fusion.py`（GRU/MLP 配置在 `configs/task2/`）
- 输入特征：课题 1 的 x6 / only6 / next6 列；门控 tau=4546 w=0.25/0.6；mlp6 0.4 + rank 0.15
- 是否看历史：与课题 1 主方案相同
- 模型：不重搜权重
- valid mean RankIC：**0.080643**
  - x6 GRU 0.072475；GRU ens 0.073418；门控 0.078738
  - only6 0.070048；next6 0.066732；cs_mlp 0.067475；cs_mlp_only6 0.062466
- 结论：**当前主候选。** 产物 `submissions/task2_fusion_x6_today.npy`。套课题 1 配方只比树融合 +0.002，列集合可能不对。
- 下一步：按 y2 自己的单因子重选 GRU 列，见 `scan-y2-cols-001`

## scan-y2-cols-001
- 日期：2026-08-23
- 代码/配置：`scripts/scan_task2_cols.py` → `outputs/task2/col_rankic.json`
- 输入特征：当天 raw 99 列，不用类别、不用历史；**不用 y1**
- 是否看历史：窗口 = 0
- 模型：无。官方 Spearman RankIC；train 排序，valid 只记账
- train y1 vs y2 RankIC：0.416949（与探查一致）
- 最强单列：`num_x[..., 8]` train **−0.046903**（valid −0.0558）。课题 1 同列是 −0.0792
- 绝对值 Top 15（train）：

| 列 | train RankIC | valid RankIC | 课题 1 Top 15 |
|---|---|---|---|
| 8 | −0.0469 | −0.0558 | 是 |
| 40 | −0.0403 | −0.0291 | 是 |
| 7 | −0.0396 | −0.0329 | 是 |
| 42 | −0.0394 | −0.0189 | 是 |
| 11 | −0.0390 | −0.0502 | 是 |
| 57 | +0.0389 | +0.0404 | 是 |
| 41 | −0.0386 | −0.0298 | 是 |
| 58 | −0.0366 | −0.0167 | 是 |
| 90 | −0.0347 | −0.0417 | 是 |
| **55** | −0.0337 | −0.0102 | **否（y2 新列）** |
| 68 | −0.0335 | −0.0539 | 是 |
| 39 | −0.0316 | −0.0460 | 是 |
| 69 | −0.0315 | −0.0516 | 是 |
| 74 | −0.0307 | −0.0540 | 是 |
| 73 | −0.0300 | −0.0434 | 是 |

- 课题 1 的 66 在 y2 上只有 −0.0271，掉出 Top 15。next6（38/72/47/3/1/4）平均 |RankIC| 仅 0.024
- **17–25 列单日截面是常数**（当天所有有标签股票同值），RankIC 为 NaN；树仍能当「市场状态」用（baseline 重要性里 raw_17 / raw_23 很靠前）。GRU 窗口加这些列没有截面排序信号，还有课题 1 hist-mkt 那种制度泄漏风险
- 耗时 / 硬件：加载 78s，扫描 48s，合计 136s
- 结论：**列集合要改，不要再拿课题 1 的 only6/next6 当 y2 最优。** 单因子天花板 0.047，低于课题 1 的 0.079，所以 0.0806 的融合并不意外。
- 下一步：一次只改一样。建议 y2 only6 = `42, 58, 55, 69, 73, 74`（课题 1 only6 把 66 换成 55）；hist / GRU 主干不要用 17–25。

## gru-only6-y2cols-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_only6_y2cols.yaml`，`scripts/eval_task2_only6_y2cols.py`
- 输入特征：相对 task2 only6 **只把 66 换成 55**（42, 58, 55, 69, 73, 74）；x6 / next6 / 树 / MLP / 门控原样锁死
- 是否看历史：窗口 = 10，含当天；末 800 天；每天 2000 只
- 模型：同 1 层 GRU hidden=64，MSE，标签 y2；epoch 1 最佳后崩，早停
- 损失：MSE 回归 y2
- valid mean RankIC：单支 **0.070531**（对照旧 only6 0.070048）；GRU ens 0.073612（对照 0.073418）；门控与全融合均为 **0.080643**（Δ=0）
- 耗时 / 硬件：拟合 231s，合计 316s，CPU
- 结论：**抛弃。** 单支噪声级上涨，锁死融合分数逐位不变。未覆盖 `task2_fusion_x6_today.npy`。
- 下一步：不要再为 0.0005 单支差换列。下一刀若做，用 y2 Top 列重做 hist / x6 主干，或先上传 0.0806 看平台 test。

## hist-y2cols-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/hist_lgbm_y2cols.yaml`
- 输入特征：当天仍是 99 列 raw/cs/ind；历史 21 列改成 y2 有限截面 RankIC Top 21（不含 17–25）
- 是否看历史：窗口 = 10，`[t-L, t)`
- 模型：与 task2 hist_lgbm 相同 LightGBM
- 损失：MSE 回归 y2
- valid mean RankIC：**0.075219**（对照旧 hist 0.075879）；与 baseline 再搜树融合 **0.077260**（对照旧树融合 0.078735）
- 结论：**抛弃。** 树重要性仍是 cat_6 / cat_1 / raw_17 / raw_23 领先，换历史列没把截面排序做起来。未覆盖主文件。

## gru-y2cols-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_y2cols.yaml`
- 输入特征：y2 Top 21 有限截面 RankIC 列；含当天；末 800 天；每天 2000 只
- 是否看历史：窗口 = 10
- 模型：1 层 GRU hidden=64，MSE
- valid mean RankIC：**0.072523**（对照旧 x6 0.072475）
- 结论：**抛弃作主干替换。** 与旧 x6 同级。

## fusion-y2cols-main-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_y2cols_fusion.py`
- 输入特征：新 y2cols GRU 换进锁定 x6_today 配方；树支仍用更强的旧 searched 0.0787（不用更弱的 y2cols 树融合）
- 模型：不重搜门控
- valid mean RankIC：**0.080643**（Δ=0）
- 结论：**抛弃。** 换 y2 列进锁定融合逐位不动。单因子换列这条路到头了。

## diag-task2-ceiling-001
- 日期：2026-08-23
- 代码/配置：`scripts/diag_task2_ceiling.py`（只读已有预测，不训练）
- 输入特征：现有全部 task2 分支
- valid 日度 oracle（每天挑当天最强一支）：**0.148274**
- 树 vs GRU 日均 Spearman：约 0.50–0.55（和课题 1 一样互补）
- 覆盖度四分位（valid 4242–4724，p25/50/75 = 4347/4491/4614）：
  - Q1：树 0.085 / GRU 0.085 / 锁定融合 0.088
  - Q2：树 0.078 / GRU 0.061 / 融合 0.080
  - Q3：树 0.084 / GRU 0.070 / 融合 0.085
  - Q4：树 0.068 / GRU 0.074 / only6 0.077 / 融合 **0.070**
- 只搜 GRU×树 raw 门控：最优 (4546, 0.15, 0.85) = 0.078741，锁死课题 1 门控 0.078738，几乎等于纯树 0.078735
- 结论：**0.12 在现有分支的日度 oracle 里 theoretically 够得到。** 更关键的是：y2 树融合输出已经是名次（mean≈2245），GRU 输出在 0.5 附近，raw 门控等于没用。测试集覆盖度 ≥4725，整天都会走 w_high，tau / w_low 到不了测试集。下一刀先修门控空间，再补树（去 17–25）和股票维 cs_mlp。

## fusion-rank-gate-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_rank_gate_fusion.py`；`scripts/eval_task2_fusion_search.py`；`scripts/eval_task2_rank_gate.py`
- 输入特征：不重训。相对锁定 x6_today **只改门控 space：raw → rank**。权重仍是 tau=4546、w_low=0.25、w_high=0.6、mlp rank 0.15
- 是否看历史：与课题 1 主方案相同
- 模型：不重搜权重
- valid mean RankIC：**0.086225**（对照 raw 门控 0.080643，Δ=+0.0056）
  - 树支 raw 尺度 mean 2245 / std 1242（已经是名次）；GRU ens mean 0.50 / std 0.06
  - raw 常数混合任意 w∈[0,0.85] 都是 0.07874（树尺度淹没 GRU）
  - rank 常数混合 w_gru=0.40 → 0.084613（Q4 0.0785）；w_gru=0.60 → 0.083499（Q4 0.0810，更接近测试集）
  - 课题 1 同权重 rank 门控 0.085984；再叠 mlp 0.15 → **0.086225**
- 结论：**保留为当前主候选。** 产物 `submissions/task2_fusion_rank_gate.npy`。未覆盖 `task2_fusion_x6_today.npy`。这是课题 1 配方在 y2 上真正生效的那一刀，不是换列。离 0.12 仍差 0.034；日度 oracle 0.148 说明分支里还有互补，但单靠再搜 tau 到不了测试集。
- 下一步：去掉树的日度常数 17–25（Q4/测试集才用得上）；用 y2 Top 21 重做 cs_mlp。

## baseline-noconst-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/baseline_noconst.yaml`
- 输入特征：相对 task2 baseline **只去掉当天数值列 17–25**（90 列 × raw/cs/ind + 8 cat）。重要性文件里的 raw_17 是选中列的第 17 个（原 26 列），名字对不上真实列号
- 是否看历史：窗口 = 0
- 模型：同截面 LightGBM
- 损失：MSE 回归 y2
- valid mean RankIC：**0.067680**（对照含 17–25 的 0.074372）
- 耗时 / 硬件：合计 239s，CPU
- 结论：**抛弃。** 日度常数对树是制度切分，不是噪声。强行删掉截面排序没做起来，分数掉 0.007。不要再融进主配方。
- 下一步：不要再砍 17–25。下一刀用 y2 Top 21 重做股票维 cs_mlp。

## cs-mlp-y2cols-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/cs_mlp_y2cols.yaml`，`scripts/eval_task2_mlp_y2cols_fusion.py`
- 输入特征：行业 z-score 换成 y2 Top 21 有限截面列；残差标签、cat_1、每天 800 只不变
- 是否看历史：窗口 = 0
- 模型：同 cs_mlp，Pearson IC 损失
- valid mean RankIC：单支 **0.067766**（对照旧 cs_mlp 0.067475）；叠进 rank-gate 配方 **0.086033**（对照 0.086225，Δ=−0.0002）
- 耗时 / 硬件：拟合 219s，合计 320s，CPU
- 结论：**抛弃。** 股票维换列没有新互补。未写新主文件。
- 下一步：按课题 1 最后一跳，用 y2 未进 Top21 的次强 6 列训 complementary GRU。

## gru-next6-y2left-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_next6_y2left.yaml`，`scripts/eval_task2_next6_y2left.py`
- 输入特征：Top21 之后 train |RankIC| 的 6 列 `83, 31, 4, 92, 86, 49`；含当天；末 800；每天 2000 只
- 是否看历史：窗口 = 10
- 模型：同 1 层 GRU hidden=64，MSE
- valid mean RankIC：单支 **0.061590**（对照旧 next6 0.066732）；叠进 rank-gate **0.086055**（对照 0.086225）
- 耗时 / 硬件：拟合 353s，合计 472s，CPU
- 结论：**抛弃。** 课题 1 式 leftover 互补列在 y2 上更弱，融合不动。未写新主文件。
- 下一步：列集合 / leftover GRU / 去常数列这条课题 1 复制链已经走完。要过 0.12 得换思路（先上传看平台 test，或单独开「y1 预测当融合支」实验并写泄漏说明）。

## diag-task1-pred-on-y2-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_task1pred.py`（只读课题 1 已有 valid 预测，不训练、不提交）
- 输入特征：`outputs/fusion_x6_today_valid.npy`（课题 1 主模型分数，**不是 y1 标签**）
- valid：课题 1 分数对 y1 = 0.120869；对 y2 = **0.085839**（Q4 0.077）；与 task2 rank-gate 日均 Spearman **0.886**
- rank 混入 task2：w=0.15 → 0.0871；w=0.50 → **0.088010**
- 结论：**不当主方案。** 说明 y2 里课题 1 架构能抓到的部分几乎全是 y1 共享结构（0.086 量级）。再混只能 +0.002，且 valid 上课题 1 权重见过 y1，y1↔y2 相关 0.417，有轻度泄漏风险。未写提交文件。
- 下一步：不要指望再搜融合权重过 0.12。需要 y2 独有信号的新模型。

## stock-z-lgbm-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/stock_z_lgbm.yaml`（`src/dataset.py` 新增 `stock_zscore`：当天值相对该股 `[t-L, t)` 均值/方差）
- 输入特征：**只**用 21 列股票自身 20 日 z-score + 8 个类别。不用 raw、不用截面 z、不用行业 z。这是课题 1 没有的第三轴
- 是否看历史：窗口 = 20，统计量只用过去，当天只作被标准化的值
- 模型：同截面 LightGBM
- 损失：MSE 回归 y2
- valid mean RankIC：单支 **0.037755**；与 rank-gate 日均 Spearman **0.399**（互补）；rank 混入 w=0.15 → **0.084812**（对照 0.086225）
- 耗时 / 硬件：合计 204s，CPU
- 结论：**抛弃。** 轴是新的、和其他支不够像，但信号太弱，融进去是拖后腿。未写新主文件。

## regime-factor-001
- 日期：2026-08-23
- 代码/配置：`scripts/train_task2_regime_factor.py`
- 输入特征：当天 CS z 的 y2 Top10 列当因子；**17–25 + 覆盖度**当制度向量，MLP 只输出 10 个因子权重（加一组全局底权重）。不是树上切 raw_17，也不是覆盖度门控两模型
- 是否看历史：窗口 = 0
- 模型：372 参数，Pearson IC 损失（按日）
- valid mean RankIC：单支 **0.062712**；与 rank-gate Spearman **0.682**；rank 混入 w=0.15 → **0.086010**（对照 0.086225）
- 耗时 / 硬件：合计 145s，CPU
- 结论：**抛弃。** 制度调权没有抓到 y2 独有结构，和现有截面支太像。未写新主文件。
- 下一步：主文件仍是 `task2_fusion_rank_gate.npy`（0.086225）。这两条新轴都不够当融合支。

## regime-router-001
- 日期：2026-08-23
- 代码/配置：`scripts/run_task2_regime_router.py`
- 输入特征：路由特征 **只**用当天 17–25（日度常数，不用覆盖度）。被路由的两支是已有树融合 vs GRU ens，上面仍叠 mlp 0.15
- 是否看历史：窗口 = 0（制度向量是当天的）
- 模型：train 上每天树/GRU 的 RankIC ~ 17–25 做 ridge；锁 train 上融合最好的硬切/软切。valid 只验收
- valid mean RankIC：**0.080640**（对照 rank-gate 0.086225）
  - train：树 0.185 / GRU 0.061，GRU 只赢 18% 的日子（树严重记了训练集）
  - valid：树 0.079 / GRU 0.073，GRU 赢 46%
  - valid 前半拟合、后半硬切：0.0693，对照同时段锁定门控 0.0848
- 耗时 / 硬件：合计 974s（含补 train 预测），CPU
- 结论：**抛弃。** 17–25 预测不了「今天该听谁」：train 上路由退化成永远信树，valid 内时间切分也输给覆盖度门控。未写新主文件。
- 下一步：主文件仍是 `submissions/task2_fusion_rank_gate.npy`。这条制度路由不再试。残差模型若做，必须避开「用 train 树分数当老师」（树 train RankIC 虚高）。

## hist-n200-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_tree_iters.py`，`scripts/eval_task2_hist_n200.py`
- 输入特征：不重训。相对锁定 hist **只改采用的树棵数**（课题 1 锁 400，y2 上扫 50–400）
- 是否看历史：窗口 = 10，与旧 hist 相同
- 模型：同一份 `hist_lgbm` / `baseline` 权重
- valid mean RankIC：
  - hist：200 棵 **0.077658**，400 棵 0.075879（400 棵过拟合）
  - baseline：仍是 400 棵最好 0.074372
  - 树融合（锁死规则）**0.079798**（对照 0.078735）
  - 叠进 rank-gate **0.086316**（对照 0.086225，Δ=+0.00009）
- 结论：**保留为当前主候选。** 产物 `submissions/task2_fusion_hist_n200.npy`。未覆盖 rank-gate 文件。这是按 y2 自己选轮数，不是搬课题 1 的 400。
- 下一步：继续按 y2 重选树正则 / 每天抽样 / GRU 窗口，一次只改一样。

## hist-reg-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/hist_lgbm_reg.yaml`
- 输入特征：与 task2 hist 相同
- 是否看历史：窗口 = 10
- 模型：相对 hist **只改** num_leaves 31→15、min_child_samples=80
- 损失：MSE 回归 y2
- valid mean RankIC：**0.073379**（对照 400 棵 0.075879，200 棵 0.077658）
- 耗时 / 硬件：合计 299s，CPU
- 结论：**抛弃。** y2 的树过拟合该用早停（少棵树），不是砍叶子。未融进主配方。

## hist-n1200-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/hist_lgbm_n1200.yaml`，`scripts/eval_task2_hist_n1200_iters.py`
- 输入特征：与 task2 hist 相同
- 是否看历史：窗口 = 10
- 模型：相对 hist **只改**每天训练股票 800→1200
- valid mean RankIC：400 棵 0.072728；最好仍是 200 棵 **0.076378**（对照 800 只/天 @200 的 0.077658）
- 耗时 / 硬件：拟合 244s，合计 441s，CPU
- 结论：**抛弃。** y2 上多抽股票没有帮助，还略差。未融进主配方。

## gru-x6-d1200-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_x6_d1200.yaml`，`scripts/eval_task2_new_x6.py`
- 输入特征：与 task2 x6_today 相同
- 是否看历史：窗口 = 10，含当天；**训练日从末 800 改成末 1200**
- 模型：同 1 层 GRU hidden=64，每天 2000 只
- valid mean RankIC：单支 **0.066906**（对照 800 天 0.072475）；叠进 hist@200 rank-gate **0.085286**（对照 0.086316）
- 耗时 / 硬件：拟合 539s，合计 638s，CPU
- 结论：**抛弃。** 更早的低覆盖度日子把 y2 GRU 带坏了。未写新主文件。

## gru-x6-d600-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_x6_d600.yaml`，`scripts/eval_task2_new_x6.py`
- 输入特征：与 task2 x6_today 相同
- 是否看历史：窗口 = 10，含当天；**训练日从末 800 改成末 600**
- 模型：同 1 层 GRU hidden=64，每天 2000 只
- valid mean RankIC：单支 **0.074262**（对照 800 天 0.072475）；叠进 hist@200 rank-gate **0.086145**（对照 0.086316）
- 耗时 / 硬件：拟合 465s，合计 625s，CPU
- 结论：**单支保留作对照，不当主。** 更短窗口对 y2 单支更好，但和现有树/only6 的锁死权重不互补，融合略降。未覆盖主文件。

## fusion-search-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_fusion_search.py`（hist@200 + 现有分支，valid 重搜）
- 输入特征：不重训
- 模型：树融合 tau=4614 w=0.5/1.0；GRU ens only6=0.25 next6=0.25；rank 门控 tau=4546 w=0.25/0.6；mlp rank **0.10**
- valid mean RankIC：**0.086676**（对照 hist_n200 锁死权重 0.086316）
- 结论：**保留。** `submissions/task2_fusion_search.npy`

## fusion-tune-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_fusion_tune.py`
- 输入特征：在 search 配方上 **只细调 mlp_w**
- valid mean RankIC：**0.086677**（mlp_w=0.08）
- 结论：**当前主候选。** `submissions/task2_fusion_tune.npy`。baseline@400 与 search 树参数组合同分。

## hist-fine-iters-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_hist_fine_iters.py`
- 输入特征：hist n=150–250，锁死 rank-gate 权重
- valid：n=225 全配方 **0.086349**；配 search 权重 **0.086622**（不如 tune）
- 结论：**hist 仍用 200 棵。**

## cs-mlp-n2000-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/cs_mlp_n2000.yaml`
- 输入特征：相对 cs_mlp **只改** max_train_stocks 800→2000
- valid mean RankIC：**0.065754**（对照 0.067475）
- 结论：**抛弃。** y2 股票维 MLP 不需要更多采样。

## hist-y2cols-iters-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_hist_y2cols_iters.py`
- 输入特征：y2 Top21 列做 hist，n=100–300
- valid：y2cols 单支最高 0.077；**全列 hist@100 锁死融合 0.086262**（不如全列 @200）
- 结论：**抛弃 y2cols 树替换。** 全列 hist@200 仍最优。

## gru-only6-d600-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_only6_d600.yaml`
- 输入特征：only6 六列 cs_zscore，末 600 天
- valid mean RankIC：单支 **0.070385**（对照 only6_today 0.070048）
- 结论：**抛弃作分支替换。** 在新 rank 门控下融合 0.087511，不如 only6_today 0.087522。

## fusion-gate-resync-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_gate_resync.py`
- 输入特征：锁 GRU ens 0.6/0.25，**重搜树/门控/mlp**
- valid mean RankIC：**0.087522**（gate tau=4700 w=0.35/0.85，mlp=0.05）
- 结论：**保留。** 关键发现：ens 加重 only6 后门控应更偏 GRU（高覆盖 w_hi↑）。

## fusion-gate-fine-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_gate_fine.py` → `eval_task2_gate_ultrafine.py`
- 输入特征：锁 tree + ens，细搜 gate tau 4650–4780
- valid mean RankIC：**0.088655**（tau=4670 w=0.28/0.9 mlp=0.06）
- 结论：**保留并继续细搜 ens。**

## fusion-ens-resync-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_ens_resync.py` → `eval_task2_ens_fine.py`
- 输入特征：锁 fine gate，重搜 GRU ens
- valid mean RankIC：**0.088722**（only6=0.7 next6=0.34 mlp=0.06）
- 结论：**当前主候选。** `submissions/task2_fusion_best.npy`。距 0.12 仍差 ~0.031。

## next6-y2left-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_next6_y2left.yaml`，`scripts/eval_task2_best_fusion.py --mode next6_y2left`
- 输入特征：y2 Top21 之外按 train |RankIC| 挑 6 列（83,31,4,92,86,49）；类比 y1 第三 GRU 支
- valid：单支 **0.061590**（对照旧 next6 0.066732）；锁死 best 配方重搜 **0.088564**
- 结论：**抛弃。** 单支更弱，融合也不涨。

## cs-mlp-y2cols-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/cs_mlp_y2cols.yaml`、`cs_mlp_only6_y2cols.yaml`，`eval_task2_best_fusion.py --mode mlp_y2cols`
- 输入特征：行业 z-score 换成 y2 Top21；only6 把 66→55；类比 y1 cs_mlp 叠层
- valid：cs_mlp_y2cols **0.067766**；only6_y2cols **0.061355**；融合最好 **0.088638**（mlp_w=0.06）
- 结论：**抛弃。** y2 上 MLP 叠层权重已顶在 0.06，加到 0.15 反而降（对照实验）。未覆盖 best。

## gru-x6-bag-s43-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/gru_x6_with_today_s43.yaml`，`eval_task2_best_fusion.py --mode x6_bag`
- 输入特征：与 x6_today 相同，只改 seed 43；42+43 平均；类比 y1 n2000 袋装
- valid：s43 单支 **0.075070**；袋装融合 **0.088696**
- 结论：**抛弃。** 未超过 best 0.088722。

## fusion-tree-gate-joint-001
- 日期：2026-08-23
- 输入特征：锁 ens 0.7/0.34，联合细搜 tree w_lo 与 gate w_hi
- valid mean RankIC：**0.088799**（tree w_lo 0.5→**0.45**，gate w_hi 0.9→**0.92**）
- 平台 test：**0.091004**（高于 valid，说明 test 全高覆盖、GRU 权重生效）
- 结论：**保留。** `submissions/task2_fusion_best.npy`

## high31-search-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_high31_search.py`
- 思路：以 valid 31 天高覆盖日 IC 优化（以为 test 全天高覆盖可对齐）
- 最优：only6=0.8 next6=0.3 w_hi=0.98 mlp6_mix=0.3 mlp_w=0.04
- valid full=0.088800 hi31=**0.140012**（OLD hi31=0.139527）
- 平台 test：**0.089633**（OLD **0.091004** ↓0.0014）
- 诊断：文件重建 diff=0，**不是 bug**；hi31 在 31 天上 +0.0005，但 test 442 天分布仍不同于这 31 天（valid 高覆盖 4242–4724 vs test 4725–5282，且 31 天样本太少过拟合）
- 结论：**抛弃 hicov。** 主方案仍 `task2_fusion_best.npy`。**禁止再用 high31 作 test 代理。**

## hist-lgbm-rankic-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/hist_lgbm_rankic.yaml`（LambdaRankIC，文献方法）
- valid：**-0.070841** → **抛弃**（y1 上 LambdaRank 也失败，y2 噪声更大）

## fusion-regression-bag-001
- 日期：2026-08-23
- 改动：x6 三种子袋装 + mlp6_mix 0.4→0.3（valid 0.088799→0.088818）
- 平台 test：**0.090889**（上一版 **0.091004**）
- 诊断（valid 高覆盖 31 日 IC）：OLD **0.139527** → NEW **0.139484**；x6 bag 单因子 high31 **0.139459**
- 结论：**全 valid 微涨 ≠ test 涨。** test 全天高覆盖，必须在 high31 子集上看增益。已恢复 `task2_fusion_best.npy` = 单 x6 + mlp6_mix=0.4。

## gru-x6-bag3-001
- 日期：2026-08-23
- 代码/配置：`gru_x6_with_today_s43/s44.yaml`，`scripts/eval_task2_x6_bag3.py`
- 输入特征：x6 种子 42+43+44 平均；类比 y1 n2000 袋装
- valid：**0.088818**（+0.000019）
- 结论：**保留进 best。** only6 袋装未试。

## y1-style-search-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_y1_style_search.py`
- y1 精确权重（only6=0.4 next6=0.15 mlp stack=0.15）在 y2 上 valid **0.088400**，不如 y2-native only6 加重（0.7/0.34）
- 最优：**mlp6_mix=0.3**（y1 用 0.4），mlp_w=0.06 → **0.088817**
- 结论：**y1 权重结构不能照搬**；mlp6 混合略调有小幅收益。

## lgbm-only6-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/lgbm_only6.yaml`
- 类比 y1 lgbm_only6 截面树
- valid 单支 0.062；叠融合 0.086 → **抛弃**

## diag-gru-usage-001
- 日期：2026-08-23
- 代码/配置：`scripts/diag_task2_gru_usage.py`
- 发现：
  - 去掉 GRU 门控（纯 tree+mlp）valid 仅 **0.081**；当前 **0.089**，GRU 贡献约 +0.008
  - test 442 天覆盖 4725–5282，**全部走 gate 高覆盖支**（w_hi=0.92 GRU）；valid 仅 31/243 天高覆盖
  - 高覆盖 valid 日上 GRU 单支 IC **0.12–0.14**，树仅 **0.089**
  - train 最大覆盖 4239，**从未见过 test 级覆盖度** → 门控 w_hi 只能在 31 天 valid 上校准
- 下一步：试 `task2_fusion_hicov.npy`（w_hi=1.0）；继续加强 GRU 主干

## y2native-gru-trio-001
- 日期：2026-08-23
- 代码/配置：`scripts/eval_task2_y2native_fusion.py`；三支同时替换为 `gru_y2cols` / `gru_only6_y2cols` / `gru_next6_y2left`
- 单支 valid：x6 **0.072523** / only6 **0.070531** / next6 **0.061590**（对照旧 0.072475/0.070048/0.066732）
- 全网格重搜融合：**0.088431**；锁死 best tree+gate 只换 ens 重搜：**0.088565**（均低于 0.088799）
- 结论：**抛弃整包替换。** y2 Top21 列 GRU 单支接近旧支，但 ens+gate 联调后仍弱于 x6_today 配方。

## cs-mlp-y2resid-001
- 日期：2026-08-23
- 代码/配置：`configs/task2/cs_mlp_y2resid.yaml`，`scripts/train_cs_mlp_y2resid.py`（hist@200 OLS 残差，类比 y1 hard_resid）
- valid：**-0.036970** → **抛弃**（y2 树残差 MLP 学不动）

## fusion-mlp-only-stack-001
- 日期：2026-08-23
- 发现：在锁死 tree+ens+gate 下，`mlp6_mix=0`（纯 cs_mlp，不混 only6）+ `mlp_w=0.1` → valid **0.088897**（+0.000098 vs 0.088799）
- 改动：`fusion_best_summary.json` 更新 mlp6_mix=0、mlp_w=0.1
- 结论：**保留。** 上传前需平台重测 test（历史 0.091004 对应旧 mlp6_mix=0.4 配方）。

## ens-oof-001
- 日期：2026-08-24
- OOF ens **0.75/0.38**：valid 0.088923 hi31 0.139319（略高于 OLD）
- 平台 test：**0.091321**（OLD **0.091354** ↓0.000033）
- 结论：**抛弃。** 已恢复 `task2_fusion_best.npy` = OLD_091354。

## conservative-fine-001
- 日期：2026-08-24
- 约束 hi31 距 OLD ≤0.00015，细搜 gate/mlp
- 最优：gate w_lo=**0.26** w_hi=**0.93** mlp_w=**0.095** valid=**0.088920**
- 产物：`task2_fusion_candidate.npy`（可选 upload，风险同前）

## hicov-fail-001
- cov4k（110 天 cov≥4000）：单支 **0.065** → 抛弃
- rank800（rank MSE 800 天）：单支 **0.071** → 抛弃
- OOF 3-way stack（纯 tree w=1）：valid **0.080** → 抛弃

## hicov-oof-plan-001
- 日期：2026-08-24
- 计划：高覆盖 GRU 重训 + OOF stack + test-sim 门禁（hi31 不涨不写 submission）
- 新增：
  - `configs/task2/gru_{x6,only6,next6}_hicov.yaml`（rank MSE + 末 600 天）
  - `scripts/train_task2_hicov_gru.py`（三支顺序训）
  - `scripts/eval_task2_recipe.py`（recipe JSON → valid/hi31/gate）
  - `scripts/run_oof_stack_task2.py`（OOF 400 天搜 3-way 权重）
  - `outputs/task2/recipes/OLD_091354.json`（hi31=0.139285，只读基准）
- 状态：**hicov GRU 训练中**；训完跑 `eval_task2_recipe.py --recipe recipes/hicov_candidate.json --write` 与 `run_oof_stack_task2.py`

## fusion-regression-gate1-001
- 日期：2026-08-24
- 改动：gate w_hi 0.92→1.0、w_lo 0.28→0.26；ens 0.7/0.34→0.72/0.36 或 0.76/0.38
- valid：0.088963–0.088971（↑）；hi31：0.139652–0.139663（↑）
- 平台 test：**0.090172**（OLD **0.091354** ↓0.0012）
- 诊断：`diag_fusion_regression.py` 确认当前 npy = NEW_summary，与 OLD_091354 spearman=0.998
- 结论：**已恢复 OLD_091354 配方。** 禁止用 valid/hi31 涨来推 test；test 全天高覆盖对 gate w_hi 极敏感。

## fusion-gate-ens-resync-002
- 日期：2026-08-23
- 平台 test 反馈：**0.091354**（mlp6_mix=0 配方）
- 改动：gate w_hi 0.92→**1.0**、w_lo 0.28→**0.26**；ens only6 0.7→**0.76**、next6 0.34→**0.38**；mlp_w=0.1
- valid：**0.088971**（+0.000074 vs 0.088897）
- 结论：**保留。** 已更新 `task2_fusion_best.npy`，建议重传测 test。
- 进行中：`gru_x6_hicov`（rank MSE + 末 600 天）训练中，目标加强 test 高覆盖 GRU 支。

## hist-n1200-001 / gru-x6-d1200-001
- 已在上方记录，均抛弃。

## hicov-candidate-001
- 日期：2026-08-24
- 代码/配置：`scripts/train_task2_hicov_gru.py`（rank MSE + 末 600 天），`scripts/eval_task2_recipe.py --recipe hicov_candidate.json --write`；对照 `recipes/OLD_091354.json`
- 输入特征：与 OLD 锁定 x6_today 配方相同，**只换 GRU 三支为 hicov 变体**（gru_{x6,only6,next6}_hicov，rank MSE + cosine + accum8 + 末 600 天）
- 是否看历史：窗口 = 10，含当天
- 模型：不重搜融合权重（tree/gate/ens/mlp 全部锁死 = OLD）
- 损失：GRU rank MSE（回归目标为当日 rank 归一化）
- valid mean RankIC：单支 x6 **0.074396**（对照 today 0.072475，+0.0019）/ only6 **0.069122**（对照 0.070048）/ next6 **0.065615**（对照 0.066732）；全配方 **0.087756**（对照 OLD 0.088897，Δ=−0.001141）
- hi31（31 天高覆盖）：单支 x6 **0.123834**（对照 today 0.133928，**−0.0101**）；全配方 **0.131164**（对照 0.139285，−0.008121）
- 诊断：hicov 三支在全 valid 只微升/降，但 hi31 全部大掉 → rank-MSE + 末 600 天没有把「高覆盖日信号」做出来，反而在测试集最接近的覆盖档退化。hicov 与 OLD 融合日均 Spearman 0.9917（hi31 0.9610），只换 GRU 的候选与现主方案几乎线性相关。`test-sim 门禁 FAIL`（hi31 掉 0.008、valid 掉 0.0011）
- 耗时 / 硬件：GRU 三支训练约 18–20 分钟（CPU 拟合 465–539s/支，参考 gru-x6-d600-001）；评估约 30s
- 结论：**抛弃。** 未覆盖 `task2_fusion_best.npy`，未写任何 submission。hicov 三支只作对照保留。
- 下一步：**高覆盖 GRU 这条路不要再加。** 单支 x6_hicov 全 valid 确实比 today 高（0.0744 vs 0.0725），但 hi31 掉 0.01——valid 全量被低覆盖日稀释，优化它不等于优化 test。诊断证明 valid 高覆盖 31 天与 test 全天高覆盖分布仍不同（同课题 1 教训）。继续做只能换「真正对高覆盖有效的信号」。

## cov4k-rank800-001
- 日期：2026-08-24
- 代码/配置：`configs/task2/gru_x6_cov4k.yaml`（train 只留 cov≥4000 的日）、`configs/task2/gru_x6_rank800.yaml`（rank MSE，末 800 天），`eval_task2_recipe.py` 对照
- 输入特征：与 OLD x6 相同，只改训练日/目标
- 模型：不重搜权重，锁 OLD 配方
- valid mean RankIC：cov4k 单支 **0.064955**（对照 0.072475）、rank800 单支 **0.071169**（对照 0.072475）；全配方 cov4k **0.088544**（Δ−0.000353）、rank800 **0.088244**（Δ−0.000653）
- hi31：cov4k **0.135910**（Δ−0.003375）、rank800 **0.135345**（Δ−0.003940）
- 结论：**抛弃。** 两个「训练分布向高覆盖对齐」的变体都低于 OLD，且 hi31 全掉。`test-sim 门禁 FAIL`。未覆盖主文件。
- 下一步：与 hicov-candidate-001 同结论——训练日/目标向高覆盖对齐这条路已试尽（cov4k / rank800 / hicov 三连败），不要再做。

## ens-oof-001（补记）
- 日期：2026-08-24
- 代码/配置：`scripts/run_oof_stack_task2.py`（OOF 400 天训练段搜 3-way 权重，全模型评估）
- OOF 400 天最优权重：**gru=0.0 / tree=1.0 / mlp=0.0**（纯树）
- valid 全配方 **0.080321**、hi31 **0.088844**（对照 OLD 0.088897 / 0.139285）
- 结论：**抛弃。** OOF 搜索在 y2 上直接退化成纯树（train 上树 RankIC 0.185 虚高，GRU 在 OOF 400 天上没赢过），valid 掉 0.0085、hi31 掉 0.05。`pass_gate=false`，未写 submission。这条 OOF 3-way stack 线路不再试。
- 下一步：主方案仍是 `submissions/task2_fusion_best.npy`（valid 0.088897 / test 0.091354）。OOF / 高覆盖重训 / 换列 / leftover 全部已走完且失败。

## tree-bag-001
- 日期：2026-08-24
- 代码/配置：`scripts/eval_task2_tree_bag.py`；hist rank-平均 n200+n225+rankic，其余 OLD_091354 不变
- valid mean RankIC：**0.088956**（OLD 0.088897）；hi31 **0.139461**（OLD 0.139285）
- 平台 test：**0.091445**（OLD **0.091354** ↑0.000091）
- 结论：**保留，已升主文件。** `submissions/task2_fusion_best.npy`；配方 `recipes/TREE_BAG_091445.json`
- 下一步：在 tree bag 壳内试 only6/next6/x6 替换；勿动 gate w_hi

## gru-hicov-repeat-001
- 日期：2026-08-24
- 代码/配置：`configs/task2/gru_x6_hicov_repeat.yaml`（末 600 天 + rank MSE + **high_cov_repeat=3 tau=4180**，600→662 天）
- 单支 valid RankIC：**0.073816**（`gru_x6_with_today` ~0.072）
- 换进 OLD 融合壳：valid **0.088555** / hi31 **0.136752**（gate **FAIL**）
- 结论：**抛弃。** 未覆盖主文件；主方案仍是 TREE_BAG test **0.091445**

## cov-extrap-001
- 日期：2026-08-24
- 代码/配置：`hist_lgbm_mkt_rel`（相对 20 日覆盖度 z + mkt std）+ `hist_lgbm_covw`（coverage_linear 样本权重）；`coverage_soft_gate_blend`；`eval_task2_cov_extrap.py`
- 单支 valid：mkt_rel **0.075644**、covw **0.074532**（均低于 hist@200 0.078）
- 融入 tree bag 最优：`n200+n225+rankic+covw` valid **0.089311** / hi31 **0.139054**（对照 TREE_BAG 0.088956 / 0.139461）
- soft gate（4200→4724 线性外推）：hi31 微涨但 valid **0.084867** ↓
- 结论：**抛弃。** test-sim FAIL，未写 submission。覆盖度外推树支不能补 train→test 分布缝

## dual-regime-001（A2）
- 日期：2026-08-24
- 代码/配置：`min_day_coverage` 过滤训练日；`dual_regime_blend`；`hist_lgbm_cov4100_expert`（70 天 scratch valid **0.035**）；`hist_lgbm_cov4000_finetune`（110 天暖启动 +80 轮 valid **0.069**）
- 融入 TREE_BAG：valid **0.088956** / hi31 **0.139461**（与 best **逐位相同**）；`task2_fusion_dual_regime.npy` 分数有微小扰动但 RankIC 不动
- 结论：**未成功。** 专家太弱或 rank-bag 抹平差异；**不升主文件**；暂不进入 A3（除非用户要求）

## dual-regime-v2-001（A2 修正）
- 日期：2026-08-24
- **修正**：A2 v1 用 `predict@200` 看 finetune 模型 → 前 200 棵不变，delta=0；改为 **delta = pred@480 − pred@400**，`hist_dual = n200 + w(cov)*delta`
- finetune delta 单支 valid **0.007661**；最佳 **delta_w0.75_x1** 融入 TREE_BAG：valid **0.091188**（+0.0022）/ hi31 **0.138700**（−0.0008）
- 平台 test：**0.091111**（prior **0.091445** ↓0.000334）
- 结论：**抛弃。** valid 涨来自低/中覆盖日，test 442 天全高覆盖 w=0.75 恒生效；hi31 降是 test 掉分先兆。未升 best。

## y2ortho-001（A3）
- 日期：2026-08-24
- 代码：`gru_x6_y2ortho`（target=y2−β·rank(y1)，β=0.4226；test 用 task1 gru y1_proxy 补回）
- 单支 valid **0.071816**；融入 TREE_BAG valid **0.087798** / hi31 **0.132764** → **FAIL**
- 结论：**抛弃**

## tree-bag-hi31-001
- 日期：2026-08-24
- 代码/配置：`scripts/eval_task2_tree_bag_hi31.py`、`emit_task2_hist_iters.py`（补 n210/n240/n250）
- 改动：**只换 hist bag** → **n240 + rankic + mkt_rel**（rank 平均）；gate/ens/mlp 全锁 TREE_BAG
- valid mean RankIC：**0.088569**（TREE_BAG 0.088956，Δ−0.000387）；hi31 **0.139893**（+0.000432）
- 平台 test：**0.091467**（prior **0.091445** ↑0.000022）
- 结论：**保留，已升主文件。** `submissions/task2_fusion_best.npy`；配方 `recipes/TREE_BAG_091467.json`

## extrap-batch-001
- 日期：2026-08-24
- 代码/配置：`hist_lgbm_extrap`（mkt_rel+covw+lambdarank）、`hist_lgbm_rankic_mkt`、`hist_lgbm_extrap_finetune`（mkt_rel 暖启动 cov≥3800 +120 轮）、`hist_lgbm_rankic_s43/s44`；`eval_task2_extrap_bag.py`
- 单支 valid（raw）：extrap **−0.072**、rankic_mkt **−0.072**、extrap_finetune **0.071**、rankic_s43 **−0.073**、rankic_s44 **−0.072**（rankic 系 raw 负 IC 正常，bag 内用 cs_rank）
- 融入 TREE_BAG_091467：4-way 加 rankic_s43 hi31 **0.140164**（+0.00027）但 valid **0.0797** ↓0.009 → **FAIL**；替换 mkt_rel / 加 extrap 均未过 gate
- 修复：`baseline.py` lambdarank + `sample_weight` 改用 LGB `weight` 参数
- 结论：**本批无新 candidate。** 主文件仍是 test **0.091467**；multi-seed rankic 与 rankic 近亲，4-way 只抬 hi31、毁 valid
- 下一步：weighted rankic 袋（`eval_task2_rankic_bag.py`）；或全新外推特征轴

## task2-现状-2026-08-24
- 平台 test **0.091467**（`task2_fusion_best.npy`，valid 0.088569 / hi31 0.139893，hist bag n240+rankic+mkt_rel）。
- 已试尽方向（全部 test-sim FAIL 或平台掉分）：hicov GRU 三支、cov4k、rank800、OOF 3-way stack、y2 列换列、y2cols GRU 整包替换、leftover 互补 GRU、regime factor/router、stock-z、hist 正则/棵数/股票数、GRU 窗口 600/1200/2000、LambdaRank、残差 MLP、mlp6_mix、ens OOF、gate 细搜、high31 优化（test 掉 0.0014，禁止再用 hi31 当 test 代理）。
- 剩余未试且有理论依据：y1 分数当特征（已诊断：与 y2 相关 0.886，混入 +0.002，且轻度泄漏风险）、cat_5 开给树、多折 walk-forward 元学习（valid 可能骗人）。三条都是低成本/高风险，且前两条已证收益 <0.003。
- 下一步：把精力转到「如何把 0.0914 的 test 分用文档/复现性保住」或「y2 独有信号的全新轴」；不要再做高覆盖对齐 / OOF 融合 / 权重细搜。

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
