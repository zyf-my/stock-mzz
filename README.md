# stock-mzz

靖戈企业命题一（课题 1，标签 `y1`）。用面板数据做横截面排序，指标是验证/测试集 mean RankIC，官方门槛 0.12。

**先读 [`计划.md`](计划.md) 和 [`experiments.md`](experiments.md)。**

当前最强：本地 valid **0.120542**，平台测试集 **0.125588**（同一文件：x6 + only6-today + next6 GRU，覆盖度门控 w_high=0.6，原行业 MLP 与 6 列 MLP 混合）。官方门槛 0.12 已过。测试分只记结果，不回灌调参。产物不在 git 里：`submissions/task1_fusion_next6_wt_mlp6.npy`。

不要把 0.117 / 0.120168 的旧融合当主方案。

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
