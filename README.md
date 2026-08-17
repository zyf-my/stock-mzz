# stock-mzz

靖戈企业命题一（课题 1，标签 `y1`）。用面板数据做横截面排序，指标是验证/测试集 mean RankIC，官方门槛 0.12。

**先读 [`计划.md`](计划.md) 和 [`experiments.md`](experiments.md)。** 本仓库是半成品：当前最强本地 valid **0.110**（`configs/fusion_gru_blend.yaml`），测试集未作为选模依据。

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

4. 预测文件、checkpoint、提交 npy **不在 git 里**。要复现 0.110，按 `计划.md` 第 10 节顺序重跑，不要并行两份面板。
5. 禁止提交：`data.z`、解包 pickle、`outputs/`、`checkpoints/`、`submissions/`、`*.npy`。

已抛弃的实验不要重做一遍，结论写在 `experiments.md`。
