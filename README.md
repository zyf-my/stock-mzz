# stock-mzz

靖戈企业命题一（课题 1，标签 `y1`）。用面板数据做横截面排序，指标是验证/测试集 mean RankIC，官方门槛 0.12。

**先读 [`计划.md`](计划.md) 和 [`experiments.md`](experiments.md)。**

当前最强：valid **0.122846**，平台 test **0.126487**（`task1_fusion_alpha_x.npy`）。相对上一版（valid 0.121627 / test 0.125795，`task1_fusion_alpha_tree.npy`）只把树历史列从 27 扩到 42，GRU/MLP/二档门控不变。

复现：先训 `configs/hist_lgbm_alpha_x.yaml`，再跑 `scripts/eval_alpha_fusion.py --stem hist_lgbm_alpha_x --tag alpha_x` → `submissions/task1_fusion_alpha_x.npy`。

三档门控（gate3）及输入时间衰减 valid 高但 test 差，已抛弃。测试分只记结果，不回灌调参。

## 队友接手

1. 数据放仓库外（官方 `data.z`）。不要解包第二份 8.5GB pickle，本机约 16GB 会爆。
2. 复制路径到 `configs/local.yaml`（已 gitignore），或设环境变量 `JINGGE_DATA`。参考：

```yaml
data:
  path: "C:/path/to/data.z"
  unpacked: false
```

3. 环境：

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. 预测文件、checkpoint、提交 npy **不在 git 里**。要复现 0.117，按 `计划.md` 第 10 节和 `experiments.md` 末条 `gru-no-today-recent-n2000-001`，不要并行两份面板。
5. 禁止提交：`data.z`、解包 pickle、`outputs/`、`checkpoints/`、`submissions/`、`*.npy`。

已抛弃的实验不要重做一遍，结论写在 `experiments.md`。
