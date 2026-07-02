# 下一步实验规划

## 一、为什么主线收束为 MDR + FBG

经过 4 个候选创新点（FBG / MRG / MDR / GER）的初步实验和 36 组关键实验（baseline / FBG / MDR / FBG+MDR × 3 数据集 × 3 seed），结论如下：

| 创新点 | 评价 | 处理 |
|--------|------|------|
| **MDR**（模态 Dropout） | 3 数据集 Recall@10 全部提升，标准差小，最稳定 | **主创新点** |
| **FBG**（频段门控） | 单独边际正向，与 MDR 组合在 Sports 上最优 | **辅助模块** |
| MRG（模态可靠性门控） | 不稳定，Baby/Clothing 掉点，频谱幅度作可靠性信号不靠谱 | 暂时不进主线 |
| GER（图边重加权） | 实现复杂、大图 OOM、掉点明显 | 暂时不进主线 |

**主线方法**：
- 主方法 = MDR
- 辅助模块 = FBG
- 完整方法 = FBG + MDR

详见 `docs/results_keyexperiments.md`。

## 二、为什么暂时不继续 MRG / GER

- **MRG**：用频谱幅度作为模态可靠性信号，缺乏理论支撑且实验上拖累组合效果（Clothing 上 MRG 反而低于 baseline）。要救需要重新设计可靠性信号（如模态预测一致性、信息熵），成本高、收益不确定。
- **GER**：边权重学习在大图上反传 OOM，即使 detach 修复后 Baby 仍掉点 13%。图边重加权与本方法主线（频域 + 模态鲁棒）关联弱。

两者代码开关保留（`modality_reliability_gating` / `graph_edge_reweighting`），便于后续单独研究，但不在当前论文实验路径上。

## 三、鲁棒性实验的目的（第一优先级）

MDR 在干净数据上的提升为 1~2%，属于温和改进。要让 MDR 成为有说服力的论文贡献，必须证明其**核心卖点——鲁棒性**：

> 在模态缺失 / 模态噪声场景下，MDR 训练的模型掉点更少。

### 实验设置（`run_robustness.sh`）

- **方法**：baseline、MDR、FBG+MDR
- **数据集**：baby、sports、clothing
- **seed**：999、42、2024
- **推理扰动模式**：
  - `normal`：完整模态（对照）
  - `drop_image`：推理时图像模态置零
  - `drop_text`：推理时文本模态置零
  - `noise_image`：推理时图像特征加高斯噪声（std=0.1）
  - `noise_text`：推理时文本特征加高斯噪声
  - `noise_both`：两模态同时加噪
- **关键原则**：扰动**只在评估阶段**生效，训练始终用完整模态。所有方法共用同一套扰动逻辑，保证公平。

### 期望结果

- MDR / FBG+MDR 在 `drop_*` / `noise_*` 下相对 baseline 的**掉点幅度更小**（即更鲁棒）。
- 若成立，论文可写：“模态 Dropout 训练使模型学到不依赖单一模态的表示，从而在模态退化场景下显著优于基线。”

### 实现

- `src/models/smore.py`：`forward` 在 `train=False` 时按 `robust_eval_mode` 扰动 `image_feats` / `text_feats`。
- `src/configs/model/SMORE.yaml`：新增 `robust_eval_mode`（默认 `normal`）、`robust_noise_std`（默认 0.1）。
- 命令行：`robust_eval_mode=drop_image` 等直接传入。

## 四、dropout rate 搜索的目的（第二优先级）

当前 MDR 只用了 `modality_dropout_rate=0.1`。需要确认该值是否最优，以及是否存在更优区间。

### 实验设置（`run_mdr_rate_search.sh`）

- **方法**：MDR、FBG+MDR
- **数据集**：sports、clothing（MDR 提升最明显的两个数据集）
- **搜索范围**：0.05 / 0.1 / 0.2 / 0.3
- **seed**：999、42、2024

### 期望结果

- 找到每个数据集上 MDR / FBG+MDR 的最佳 dropout rate。
- 若 0.1 已最优，则现有结果直接可用；若更优值存在，则更新主结果。

## 五、实验完成后怎么看结果

### 1. 解析日志为 CSV

```bash
# 鲁棒性实验
python scripts/parse_smore_results.py logs_robustness_*/  -o results_robustness.csv

# rate 搜索
python scripts/parse_smore_results.py logs_mdr_rate_*/     -o results_mdr_rate.csv
```

CSV 字段：`dataset, method, seed, robust_mode, dropout_rate, recall@10, recall@20, recall@50, ndcg@10, ndcg@20, ndcg@50, precision@10, map@10, log_file, status`

### 2. 鲁棒性结果分析重点

- 对每个 (dataset, robust_mode)，计算 3 seed 平均的 recall@10 / ndcg@10。
- 对比 baseline vs MDR vs FBG+MDR 的**相对掉点**：
  - `drop_rate = (normal - perturbed) / normal`
  - MDR 的 drop_rate 应小于 baseline。

### 3. rate 搜索结果分析重点

- 对每个 (dataset, method)，画 dropout_rate vs recall@10 曲线（3 seed 平均）。
- 找峰值对应的 rate。

## 六、实验规模与时间预估

| 脚本 | 实验数 | 双 GPU 预估时间 |
|------|--------|----------------|
| `run_robustness.sh` | 162 | ~30 小时（较大，可先跑 sports+clothing） |
| `run_mdr_rate_search.sh` | 48 | ~8 小时 |

> 鲁棒性实验 162 个较大，建议先在 sports + clothing 上跑（108 个），或先单 seed 跑通验证流程再扩量。
