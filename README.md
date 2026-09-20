# stock-mzz

靖戈企业命题一（课题 1，标签 `y1`）。用面板数据做横截面排序，指标是验证/测试集 mean RankIC，官方门槛 0.12。

**先读 [`计划.md`](计划.md) 和 [`experiments.md`](experiments.md)。**

锁定提交（平台已测）：`submissions/task1_fusion_existing_recursive_w10.npy`  
本地 valid **0.130825**，平台 test **0.129291**，形状 `(442, 5282)` float32。完整实验记录见 [`experiments.md`](experiments.md)，技术方案见 [`说明书.md`](说明书.md)。

## 队友打包提交

从本仓库拷一份即可。平台预测文件和锁定的 8 套模型已经入库，**不要把官方 `data.z` 拷进提交包**。

1. 平台打分文件：`submissions/task1_fusion_existing_recursive_w10.npy`
2. 已训练模型：`checkpoints/` 里已跟踪的 baseline / alpha-x / alpha-x2 / 三支 GRU / 两支 MLP（含 `.ens2.txt` 和 `.meta.json`）
3. 可运行代码：`src/`、`scripts/`、`configs/`（只要 yaml 示例，不要 `configs/local.yaml`）、`requirements.txt`
4. 文档：`说明书.md`、`README.md`、Word 实现方案、PPT 演示稿

数据仍放仓库外。本机路径写入已 gitignore 的 `configs/local.yaml`，或设 `JINGGE_DATA`：

```yaml
data:
  path: "C:/path/to/data.z"
  unpacked: false
```

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不要并行读两份整包面板。已抛弃的实验不要重做，结论在 `experiments.md`。这个仓库含赛题方法，不要公开。
