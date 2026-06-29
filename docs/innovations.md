# SMORE 创新点说明文档

> 本文档详细描述了在原始 SMORE (WSDM 2025) 基础上添加的四个创新点，包括动机、方法设计、数学公式、代码改动及配置说明。

---

## 目录

1. [创新点 1：频段门控 (Frequency Band Gating)](#创新点-1频段门控-frequency-band-gating)
2. [创新点 2：模态可靠性门控 (Modality Reliability Gating)](#创新点-2模态可靠性门控-modality-reliability-gating)
3. [创新点 3：模态 Dropout 鲁棒训练 (Modality Dropout Robust Training)](#创新点-3模态-dropout-鲁棒训练-modality-dropout-robust-training)
4. [创新点 4：图边重加权 (Graph Edge Reweighting)](#创新点-4图边重加权-graph-edge-reweighting)
5. [创新点交互关系](#创新点交互关系)
6. [配置与使用说明](#配置与使用说明)

---

## 创新点 1：频段门控 (Frequency Band Gating)

### 动机

原始 SMORE 的频谱卷积 (`spectrum_convolution`) 使用静态可学习的复数权重对频域信号进行滤波。每个频段的处理权重仅通过梯度更新全局学习，无法根据输入数据的特征动态调整。然而，不同 item 的多模态特征在频域中具有不同的噪声分布——某些 item 的图像特征可能在高频段存在大量噪声，而另一些 item 的文本特征可能在低频段包含冗余信息。静态滤波器无法自适应地处理这种差异性。

### 方法设计

在 FFT 变换后、复数权重相乘**之前**，引入基于频谱幅度的输入依赖门控机制：

$$
\mathbf{g}^{img} = \sigma(\mathbf{W}^{img}_g \cdot |\mathbf{F}^{img}|), \quad \mathbf{g}^{txt} = \sigma(\mathbf{W}^{txt}_g \cdot |\mathbf{F}^{txt}|)
$$

其中 $\mathbf{F}^{img}, \mathbf{F}^{txt} \in \mathbb{C}^{N \times D_f}$ 为 FFT 后的频谱表示，$|\cdot|$ 为复数幅度，$\mathbf{W}_g$ 为线性变换权重，$\sigma$ 为 Sigmoid 激活函数，$D_f = \lfloor D/2 \rfloor + 1$ 为频率维度。

门控应用于频谱信号：

$$
\hat{\mathbf{F}}^{img} = \mathbf{F}^{img} \odot \mathbf{g}^{img}, \quad \hat{\mathbf{F}}^{txt} = \mathbf{F}^{txt} \odot \mathbf{g}^{txt}
$$

融合频谱同样使用独立门控：

$$
\mathbf{g}^{fus} = \sigma(\mathbf{W}^{fus}_g \cdot |\hat{\mathbf{F}}^{img} \odot \hat{\mathbf{F}}^{txt} \odot \mathbf{W}^{fus}_c|), \quad \hat{\mathbf{F}}^{fus} = (\hat{\mathbf{F}}^{img} \odot \hat{\mathbf{F}}^{txt} \odot \mathbf{W}^{fus}_c) \odot \mathbf{g}^{fus}
$$

### 设计要点

- **门控位置**：门控在复数权重相乘之前应用，使模型能在滤波前选择有效频段，比滤波后缩放更具表达力
- **轻量设计**：每个门控网络仅为单层 Linear + Sigmoid，参数量极小（embedding_dim=64 时，每个门控仅 33×33+33=1122 个参数）
- **输入依赖**：门控信号由频谱幅度生成，能根据每个 item 的特征动态调整频段重要性

### 代码改动

**文件**：`src/models/smore.py`

| 位置 | 改动 |
|------|------|
| `__init__` | 新增 `freq_band_gating` 标志，以及 `image_band_gate`、`text_band_gate`、`fusion_band_gate` 三个门控网络 |
| `spectrum_convolution` | FFT 后计算幅度 → 门控网络 → 应用门控信号 |

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `freq_band_gating` | bool | False | 是否启用频段门控 |

---

## 创新点 2：模态可靠性门控 (Modality Reliability Gating)

### 动机

原始 SMORE 对三个模态视图（image/text/fusion）简单取平均融合，假设所有模态对所有 item 具有同等重要性。然而现实中，不同 item 的模态可靠性差异显著——服装类 item 的图像特征通常比文本更有区分力，而电子类 item 的文本描述往往比图像更可靠。均匀平均忽略了这种差异性，导致不可靠模态的噪声信号稀释了可靠模态的有效信息。

### 方法设计

基于频谱统计特征学习 per-item 的模态可靠性权重，用 softmax 加权求和替代简单平均：

**步骤 1**：在频谱卷积中提取频谱统计特征：

$$
\mathbf{s}_i = [|{\mathbf{F}^{img}_i}| \;\| \; |\mathbf{F}^{txt}_i| \;\| \; |\mathbf{F}^{img}_i \odot \mathbf{F}^{txt}_i|] \in \mathbb{R}^{3D_f}
$$

其中 $\|$ 表示拼接，$|\cdot|$ 为频谱幅度，$i$ 为 item 索引。

**步骤 2**：通过可靠性估计器生成权重：

$$
\mathbf{r}_i = \text{softmax}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{s}_i + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^3
$$

其中 $\mathbf{r}_i = [r_i^{img}, r_i^{txt}, r_i^{fus}]$ 为三个模态的可靠性权重，满足 $\sum r_i^{(\cdot)} = 1$。

**步骤 3**：加权融合替代均匀平均：

$$
\mathbf{e}^{side}_i = r_i^{img} \cdot \mathbf{e}^{img}_i + r_i^{txt} \cdot \mathbf{e}^{txt}_i + r_i^{fus} \cdot \mathbf{e}^{fus}_i
$$

对于用户，由于没有直接的频谱特征，使用均匀权重 $r_u = [\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]$。

### 设计要点

- **频谱驱动**：可靠性由频域统计特征驱动，而非简单地从嵌入空间学习，确保可靠性评估与信号质量直接关联
- **Per-item 个性化**：每个 item 拥有独立的模态可靠性权重，而非全局共享
- **Softmax 归一化**：权重归一化为概率分布，避免某一模态主导

### 代码改动

**文件**：`src/models/smore.py`

| 位置 | 改动 |
|------|------|
| `__init__` | 新增 `modality_reliability_gating` 标志和 `reliability_estimator` 网络 |
| `spectrum_convolution` | 返回值新增 `spectral_stats`（三组频谱幅度拼接） |
| `forward` | 将 `torch.mean(stacked)` 替换为 softmax 加权求和 |

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `modality_reliability_gating` | bool | False | 是否启用模态可靠性门控 |

---

## 创新点 3：模态 Dropout 鲁棒训练 (Modality Dropout Robust Training)

### 动机

多模态推荐模型容易过度依赖某一主导模态（如服装推荐中的图像模态），导致：(1) 当该模态缺失或噪声严重时性能急剧下降；(2) 次要模态的信号被忽略，模型无法充分利用多模态互补信息。原始 SMORE 的 Dropout 仅作用于行为偏好门控值，没有模态级别的正则化。

### 方法设计

训练时以概率 $p$ 独立随机丢弃整个模态视图，迫使模型学习不依赖任何单一模态的鲁棒表示：

$$
\text{drop}^{img}_t \sim \text{Bernoulli}(p), \quad \text{drop}^{txt}_t \sim \text{Bernoulli}(p), \quad \text{drop}^{fus}_t \sim \text{Bernoulli}(p)
$$

丢弃时将该模态视图嵌入置零：

$$
\mathbf{e}^{img}_t = \begin{cases} \mathbf{0} & \text{if } \text{drop}^{img}_t = 1 \\ \mathbf{e}^{img}_t & \text{otherwise} \end{cases}
$$

**安全约束**：保证至少保留一个模态视图。若三个模态均被丢弃，随机保留一个：

$$
\text{if } \text{drop}^{img} \land \text{drop}^{txt} \land \text{drop}^{fus}: \text{randomly keep one}
$$

**推理阶段**：不丢弃任何模态，使用全部三个模态视图（利用 `self.training` 标志自动控制）。

### 设计要点

- **模态级别丢弃**：与特征级 Dropout 互补，在更高的语义层次上进行正则化
- **独立丢弃**：三个模态独立决定是否丢弃，增加训练随机性
- **安全约束**：保证至少一个模态保留，避免信息完全丢失
- **与对比学习协同**：模态丢弃时 `side_embeds` 退化，InfoNCE 对比损失自然增大，起到额外正则化效果
- **丢弃位置**：在 item-item GCN 传播后、模态感知偏好模块前丢弃，不影响频谱卷积和图卷积的计算

### 代码改动

**文件**：`src/models/smore.py`

| 位置 | 改动 |
|------|------|
| `__init__` | 新增 `modality_dropout_rate` 参数 |
| `forward` | 在 item-item GCN 后、模态感知偏好模块前添加模态 Dropout 逻辑 |

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `modality_dropout_rate` | float | 0.0 | 每个模态被丢弃的概率（0.0 = 关闭），推荐搜索范围 [0.0, 0.1, 0.2, 0.3] |

---

## 创新点 4：图边重加权 (Graph Edge Reweighting)

### 动机

原始 SMORE 的用户-项目二部图中所有交互边权重相同（0/1），即所有历史交互被平等对待。然而，不同交互的重要性不同——用户与多模态特征高度一致的 item 的交互更值得关注，而与多模态特征不匹配的交互可能是随机的或噪声性的。均匀边权重无法区分交互质量差异。

### 方法设计

基于用户嵌入和项目模态嵌入学习交互边的权重，替换原始的均匀 0/1 权重：

**步骤 1**：提取每条交互边 $(u, i)$ 的特征表示：

$$
\mathbf{h}_{u,i} = [\mathbf{e}_u^{id} \;\| \; \frac{1}{3}(\mathbf{e}_i^{img} + \mathbf{e}_i^{txt} + \mathbf{e}_i^{fus})] \in \mathbb{R}^{2D}
$$

其中 $\mathbf{e}_u^{id}$ 为用户 ID 嵌入，$\mathbf{e}_i^{(\cdot)}$ 为经频谱门控和模态门控后的 item 模态嵌入。

**步骤 2**：通过 MLP 计算边权重：

$$
w_{u,i} = \text{Softplus}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h}_{u,i} + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^+
$$

Softplus 保证权重为正。

**步骤 3**：构建重加权邻接矩阵并进行对称归一化：

$$
\tilde{\mathbf{A}} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A}_w \mathbf{D}^{-\frac{1}{2}}
$$

其中 $\mathbf{A}_w$ 为使用学习权重 $w_{u,i}$ 构建的加权邻接矩阵，$\mathbf{D}$ 为对应的度矩阵。

**步骤 4**：用 $\tilde{\mathbf{A}}$ 替代原始 $\mathbf{A}$ 进行 User-Item GCN 传播。

### 设计要点

- **多模态信号驱动**：边权重由项目模态嵌入和用户嵌入共同决定，而非仅依赖交互频率
- **端到端可学习**：边权重 MLP 与模型其余部分联合优化，梯度通过稀疏矩阵操作回传
- **动态计算**：训练时每个 forward pass 重新计算边权重，适应嵌入的动态变化
- **Softplus 激活**：保证边权重为正值，维持邻接矩阵的物理意义
- **设备同步**：边索引使用 `register_buffer` 注册，确保模型 `.to(device)` 时自动迁移

### 代码改动

**文件**：`src/models/smore.py`

| 位置 | 改动 |
|------|------|
| `__init__` | 新增 `graph_edge_reweighting` 标志，`register_buffer` 注册边索引，`edge_weight_mlp` 网络 |
| `forward` | 训练时计算边权重，构建重加权邻接矩阵，替代原始 `adj` 进行 GCN |
| 新增方法 | `_build_reweighted_adj(edge_weights)` 构建对称归一化的重加权邻接矩阵 |

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `graph_edge_reweighting` | bool | False | 是否启用图边重加权 |

---

## 创新点交互关系

四个创新点作用于模型流水线的不同阶段，可独立开关且协同增强：

```
输入特征 → 频谱变换 → [创新点1: 频段门控] → 复数滤波 → IFFT
                                                      ↓
                                          [创新点2: 频谱统计 → 可靠性估计]
                                                      ↓
模态门控 → Item-Item GCN → [创新点3: 模态Dropout] → 模态感知偏好
                                                      ↓
用户ID嵌入 + Item模态嵌入 → [创新点4: 边权重MLP] → 重加权User-Item GCN
                                                      ↓
                                    [创新点2: 可靠性加权融合] → 最终嵌入
```

| 组合 | 协同效果 |
|------|---------|
| 创新点 1 + 2 | 门控后的频谱更干净，可靠性评估基于去噪后的频谱统计，更准确 |
| 创新点 1/2 + 3 | 频域操作与模态级 Dropout 互不冲突；Dropout 使可靠性估计更鲁棒 |
| 创新点 4 + 1 | 边权重基于门控后的 item 模态嵌入计算，信号质量更高 |
| 创新点 4 + 3 | 训练时模态 Dropout 微弱影响边权重计算（轻微正则化效果），推理时无影响 |
| 全部开启 | 频段门控提升频域信号质量 → 可靠性评估更准 → 模态 Dropout 增强鲁棒性 → 边重加权提升图结构质量 |

---

## 配置与使用说明

### YAML 配置

在 `src/configs/model/SMORE.yaml` 中控制各创新点的开关和参数：

```yaml
# Innovation 1: Frequency Band Gating
freq_band_gating: False          # True 开启

# Innovation 2: Modality Reliability Gating
modality_reliability_gating: False  # True 开启

# Innovation 3: Modality Dropout Robust Training
modality_dropout_rate: 0.0       # 设为 0.1~0.3 开启

# Innovation 4: Graph Edge Reweighting
graph_edge_reweighting: False    # True 开启
```

### 命令行使用

所有创新点关闭时，行为与原始 SMORE 完全一致：

```bash
# 原始 SMORE（默认）
python main.py -m SMORE -d baby

# 开启频段门控
python main.py -m SMORE -d baby freq_band_gating=True

# 开启模态可靠性门控
python main.py -m SMORE -d baby modality_reliability_gating=True

# 开启模态 Dropout（概率 0.2）
python main.py -m SMORE -d baby modality_dropout_rate=0.2

# 开启图边重加权
python main.py -m SMORE -d baby graph_edge_reweighting=True

# 组合使用多个创新点
python main.py -m SMORE -d baby freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1 graph_edge_reweighting=True
```

### 推荐消融实验方案

| 实验编号 | 频段门控 | 模态可靠性门控 | 模态Dropout | 图边重加权 | 说明 |
|----------|---------|--------------|------------|----------|------|
| Baseline | ✗ | ✗ | 0.0 | ✗ | 原始 SMORE |
| +FBG | ✓ | ✗ | 0.0 | ✗ | 仅频段门控 |
| +MRG | ✗ | ✓ | 0.0 | ✗ | 仅模态可靠性门控 |
| +MDR | ✗ | ✗ | 0.1 | ✗ | 仅模态Dropout |
| +GER | ✗ | ✗ | 0.0 | ✓ | 仅图边重加权 |
| +FBG+MRG | ✓ | ✓ | 0.0 | ✗ | 频段门控 + 可靠性门控 |
| +FBG+MRG+MDR | ✓ | ✓ | 0.1 | ✗ | 三个创新点组合 |
| Full | ✓ | ✓ | 0.1 | ✓ | 全部四个创新点 |

### 参数搜索建议

| 创新点 | 搜索参数 | 搜索范围 |
|--------|---------|---------|
| 模态 Dropout | `modality_dropout_rate` | [0.0, 0.1, 0.2, 0.3] |

其他创新点为布尔开关，建议先逐一开启验证效果，再组合使用。
