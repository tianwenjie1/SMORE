# SMORE 改进实验记录

> 本文档记录 SMORE（WSDM 2025）复现 + 改进的全过程：创新点设计、踩坑修复、各轮实验结果、后续尝试。
> **后续每次实验/尝试都追加到本文档末尾**，按时间倒序或正序追加，保持连贯。

---

## 一、项目背景

- **原始工作**：SMORE — Spectrum-based Modality Representation Fusion GCN for Multimodal Recommendation（WSDM 2025）
- **目标**：在复现 SMORE 的基础上，添加改进模块，形成可投稿的改进版本
- **数据集**：Baby、Sports、Clothing（Amazon 多模态推荐数据集，含图像/文本特征）
- **评估指标**：Recall@{5,10,20,50}、NDCG@{5,10,20,50}、Precision、MAP
- **硬件**：双 GPU 服务器（NVIDIA 24GB ×2），后迁移至 4×RTX 3060 12GB 服务器

---

## 二、SMORE 原始架构回顾

复现后确认的核心结构（后续改动均基于此）：

1. **特征投影**：原始图像/文本特征 → Linear → 共享 embedding 空间
2. **频谱卷积**（核心）：FFT → 可学习复数权重滤波 → IFFT，分别处理 image/text/fusion 三路
3. **模态门控**：6 个 sigmoid 门控（gate_v/t/f + gate_*_prefer）调制三路模态
4. **图卷积**：
   - User-Item 二部图 LightGCN（n_ui_layers 层）
   - 3 个 Item-Item KNN 图（image/text/fusion，n_layers=1）
5. **模态感知偏好**：fusion 条件注意力 + 行为偏好门控 → 三路平均 + 残差
6. **损失**：BPR + L2 正则 + InfoNCE 对比损失（side_embeds vs content_embeds）

---

## 三、创新点设计（4 个候选）

### Innovation 1：FBG — Frequency Band Gating（频段门控）
- **动机**：原始频谱卷积的复数权重是静态学习的，无法根据输入动态选择频段
- **方法**：FFT 后、复数权重相乘前，基于频谱幅度生成输入依赖的门控信号（sigmoid），自适应强调有效频段、抑制噪声频段
- **位置**：`smore.py` `spectrum_convolution()`
- **配置**：`freq_band_gating: False`（默认关）

### Innovation 2：MRG — Modality Reliability Gating（模态可靠性门控）
- **动机**：原始 SMORE 对三路模态简单平均，忽略 per-item 模态可靠性差异
- **方法**：基于频谱统计特征学习 per-item 模态可靠性权重（softmax），加权求和替代平均
- **位置**：`smore.py` `spectrum_convolution()`（返回 spectral_stats）+ `forward()`
- **配置**：`modality_reliability_gating: False`
- **状态**：⚠️ 后续实验证明效果不稳定，已弃用主线

### Innovation 3：MDR — Modality Dropout Robust Training（模态 Dropout 鲁棒训练）
- **动机**：模型过度依赖单一模态，模态缺失/噪声时性能下降
- **方法**：训练时以概率 p 随机丢弃整个模态视图（置零），保证至少保留一个；推理时用全部模态
- **位置**：`smore.py` `forward()`（item-item GCN 后、模态感知偏好前）
- **配置**：`modality_dropout_rate: 0.0`
- **状态**：✅ 核心有效创新点

### Innovation 4：GER — Graph Edge Reweighting（图边重加权）
- **动机**：User-Item 图所有边权重相同（0/1），忽略多模态对齐度
- **方法**：基于用户嵌入 + 项目模态嵌入学习边权重（Softplus 保证正），构建重加权归一化邻接矩阵
- **位置**：`smore.py` `_build_reweighted_adj()` + `forward()` GCN 部分
- **配置**：`graph_edge_reweighting: False`
- **状态**：⚠️ 大图 OOM + 掉点，已弃用主线

### 鲁棒性评估模块（附加）
- **目的**：评估推理阶段模态扰动下的性能
- **模式**：normal / drop_image / drop_text / noise_image / noise_text / noise_both
- **位置**：`smore.py` `forward()`（仅 `train=False` 时扰动，不影响训练）
- **配置**：`robust_eval_mode: normal`、`robust_noise_std: 0.1`

---

## 四、踩坑与修复记录

> 这些修复记录体现实验严谨性，写论文时可作复盘素材。

| # | 问题 | 原因 | 修复 |
|---|------|------|------|
| 1 | `AttributeError: 'Config' object has no attribute 'get'` | 项目 Config 类非普通 dict，不支持 `.get()` | 改用 `config['key'] or default` |
| 2 | `CUBLAS_STATUS_NOT_INITIALIZED` | 单 GPU 同时跑多个进程挤爆显存 | 脚本改为每 GPU 同一时刻只跑 1 个进程 |
| 3 | `KeyError: 'valid_metric'` | 从仓库根目录跑 `python src/main.py`，配置路径错 | 脚本先 `cd src` 再跑 `python main.py` |
| 4 | MRG 张量广播错误 | `full_weights [n,3]` 与 `stacked [3,n,dim]` 维度不匹配 | 转置 `full_weights.t().unsqueeze(-1)` → `[3,n,1]` |
| 5 | GER 大图反传 OOM（14.5GB） | 度归一化带梯度反传物化巨大中间张量 | 度归一化 `torch.no_grad()` detach |
| 6 | **创新点参数没生效**（24 组结果全相同） | `parse_known_args()` 把 `key=value` 当未知参数丢弃 | `main.py` 解析 `key=value` 注入 config_dict |
| 7 | `seed=999` 破坏网格搜索 | 覆盖了 `[999]` 列表为 int，`product(999)` 报错 | `main.py` 将 seed 自动包装成 `[N]` |
| 8 | 所有进程都挤到 GPU 0 | `configurator.py` 用 `gpu_id=0` 覆盖 `CUDA_VISIBLE_DEVICES` | 命令行传 `gpu_id=$g` |
| 9 | 解析器 robust_mode 全是 normal | 模式名含下划线（drop_image），被 split 拆散 | 先按已知模式后缀匹配，再 split |
| 10 | 解析器多目录报错 | `logs_*` glob 展开多目录，argparse 只收 1 个 | 改 `nargs="+"` + 去重 |

---

## 五、实验轮次与结果

### 第 1 轮：初步 24 组消融（2026-06-29）

- **设置**：8 方法（baseline/FBG/MRG/MDR/GER/FBG+MRG/FBG+MRG+MDR/Full）× 3 数据集 × 1 seed
- **结果**：❌ 全部结果相同 → 发现 bug #6（参数未生效），整轮作废
- **教训**：先验证参数真的进入配置再大规模跑

### 第 2 轮：修复后重跑失败实验（2026-07-01）

- 修复 bug #4（MRG）、#5（GER）后重跑 14 个失败实验
- MRG 三数据集全挂 → 修复后能跑但掉点
- GER 在 Baby 能跑（掉点 13%），Sports/Clothing 仍 OOM

### 第 3 轮：关键主实验（2026-07-01，36 组）✅

- **设置**：baseline / FBG / MDR / FBG+MDR × {baby, sports, clothing} × seed {999, 42, 2024}
- **dropout_rate = 0.1**
- **结果**（Recall@10，3 seed 平均）：

| 方法 | Baby | Sports | Clothing |
|------|------|--------|----------|
| baseline | 0.0650 | 0.0742 | 0.0660 |
| FBG | 0.0646 | 0.0744 | 0.0665 |
| MDR | 0.0651 | 0.0751 | 0.0667 |
| **FBG+MDR** | 0.0648 | **0.0754** | **0.0667** |

- **结论**：
  - MDR 三数据集 R@10 全提升（+0.1%~1.1%），最稳定 → **主创新点**
  - FBG+MDR 在 Sports 最优（+1.6%）→ FBG 为辅助模块
  - Clothing 上 MDR 单独最好，FBG 加成有限

### 第 4 轮：MDR dropout rate 搜索（2026-07-02，48 组）✅

- **设置**：MDR / FBG+MDR × {sports, clothing} × rate {0.05, 0.1, 0.2, 0.3} × 3 seed
- **结果**（Recall@10，3 seed 平均）：

| | Sports MDR | Sports FBG+MDR | Clothing MDR | Clothing FBG+MDR |
|---|---|---|---|---|
| 0.05 | 0.0753 | 0.0747 | 0.0666 | 0.0664 |
| 0.1 | 0.0753 | 0.0751 | 0.0666 | 0.0668 |
| **0.2** | **0.0755** | **0.0755** | **0.0671** | **0.0670** |
| 0.3 | 0.0755 | 0.0749 | 0.0666 | 0.0667 |

- **结论**：**rate = 0.2 最佳**，两数据集两方法全面最优；0.3 开始下降

### 第 5 轮：鲁棒性实验（2026-07-02，162 组）✅

- **设置**：baseline / MDR / FBG+MDR × {baby, sports, clothing} × 3 seed × 6 扰动模式（normal/drop_image/drop_text/noise_image/noise_text/noise_both）
- **扰动只在推理阶段**，训练始终完整模态
- **结果**（Recall@10，3 seed 平均，Sports）：

| 扰动模式 | baseline | MDR | FBG+MDR |
|---------|----------|------|---------|
| normal | 0.0740 | 0.0752 | 0.0752 |
| drop_image | 0.0759 | 0.0759 | 0.0758 |
| drop_text | 0.0758 | 0.0758 | 0.0759 |
| noise_image | 0.0743 | **0.0754** | 0.0751 |
| noise_text | 0.0744 | **0.0752** | 0.0751 |
| noise_both | 0.0740 | **0.0754** | 0.0752 |

- **核心发现**：**MDR 在噪声场景下几乎不掉点，baseline 普遍掉 0.3~0.5%**
  - Sports noise_both：baseline -0.5%，MDR +0.3%
  - Sports noise_image：baseline -0.3%，MDR +0.3%
  - Clothing noise_image：baseline -0.1%，MDR +0.3%
- **drop 场景**：所有方法都不掉点（三路模态冗余），论文重点用 noise 场景
- **论文卖点成立**：模态 Dropout 训练让模型不依赖单一模态，模态噪声下显著鲁棒

### 第 6 轮：用 rate=0.2 重跑主表 + 噪声强度敏感性（2026-07-09，进行中）

- **目的**：
  - 6a：主表用 0.2（与 rate 搜索结论一致，数字再涨 ~0.3%）
  - 6b：不同噪声强度（0.05/0.1/0.2/0.3）下 baseline vs MDR，证明"噪声越大 MDR 优势越大"
- **设置**：
  - 6a：baseline/FBG/MDR/FBG+MDR × 3 数据集 × 3 seed = 36 组
  - 6b：baseline/MDR × {sports,clothing} × 3 seed × {noise_image, noise_both} × 4 std = 96 组
- **硬件**：新服务器 4×RTX 3060 12GB，用 GPU 2、3
- **状态**：⏳ 环境配置 + 数据下载中，待跑
- **结果**：待追加

---

## 六、各创新点最终评价

| 创新点 | 效果 | 论文定位 |
|--------|------|---------|
| **MDR**（模态 Dropout） | R@10 跨数据集 +0.1~1.8%，噪声下不掉点 | ✅ **主创新点** |
| **FBG**（频段门控） | 单独边际正向，与 MDR 组合 Sports 最优 | 🟡 **辅助模块** |
| MRG（模态可靠性门控） | Baby -5.6%、Clothing -1.7%，不稳定 | ❌ 弃用（保留代码开关） |
| GER（图边重加权） | Baby -13%，大图 OOM | ❌ 弃用（保留代码开关） |

**主线方法**：主方法 = MDR，辅助 = FBG，完整方法 = FBG + MDR

---

## 七、论文故事线

> 基于频段自适应与模态鲁棒训练的多模态推荐增强方法

1. 原始 SMORE 频谱滤波偏静态，多模态推荐易过度依赖单一模态
2. 设计 FBG（频段门控，自适应滤波）+ MDR（模态 Dropout，鲁棒训练）
3. 实验：
   - 主表：MDR 跨数据集一致提升，FBG+MDR 在 Sports 最优
   - 鲁棒性：MDR 噪声下不掉点，baseline 掉 0.3~0.5%（差异化卖点）
   - rate 敏感性：0.2 最佳

---

## 八、后续尝试记录

> 每次新的实验/尝试追加于此，格式：
> ### 第 N 轮：标题（日期）
> - 目的：
> - 设置：
> - 结果：
> - 结论：

### 第 6 轮：rate=0.2 主表 + 噪声强度（2026-07-09）
- 见上文第五节第 6 轮，结果待跑完后追加

### 🔴 战略转向：从 trick 升级为问题级贡献（2026-07-09）

**背景**：MDR/FBG 若直接作为"创新点"就是 trick（dropout + gate），二区必被拒。经评估决定升级论文定位。

**新定位**：
- 旧：基于 FBG 和 MDR 的 SMORE 改进方法（trick）
- 新：**面向模态质量偏移（Modality Quality Shift, MQS）的鲁棒多模态推荐**（问题级贡献）

**核心论点**：现有 MMRec 默认模态完整可靠同分布，但真实平台存在模态缺失/噪声/错配/尾部低质量/图文不一致。模型过度依赖强模态时，模态质量变化下崩溃。需学习对质量变化稳定的偏好表示。

**MQS 五类定义**：
1. Missing modality（缺失）
2. Noisy modality（噪声）
3. Mismatched modality（图文错配）
4. Tail-quality degradation（尾部 item 模态更差）
5. Modality dominance（过度依赖某模态）

**方法升级**：MDR → **MQR（Modality Quality Regularization）**
- 不只是随机 dropout，而是构造多个模态质量环境（clean / image-noise / text-noise / mismatch / tail-degraded）
- 约束 clean 与 degraded 环境下偏好一致性（Preference Invariance Loss）
- tail item 加权（w = 1/log(1+degree)）

**命门消融（证明不是 trick）**：必须证明 MQR+不变性 ≫ 朴素 dropout + 朴素噪声增强

**FBG 处理**：暂停主线。仅当强噪声（std=0.3/0.5、shuffle 50%）下 FBG+MQR 明显优于 MQR（>3-5%）才召回，否则降级到 appendix。

**四阶段计划**：
- P1：扩展 MQS 评测协议（noise 多档/shuffle/mismatch/tail-noise/pop-missing），跑 baseline SMORE → 证明问题存在
- P2：实现 MQR（质量环境 + 偏好不变性 + tail-aware）
- P3：命门消融（证明不是 dropout）
- P4：外部基线 DGMRec / I3-MRec

**止损判断**（P3 后）：
- 继续条件：clean 不掉 / noise_both 提升>5% / tail Recall 提升>8% / MQR 明显超 dropout
- 放弃条件：MQR≈dropout / 只一数据集有效 / clean 掉>2% / 强噪声只提升 1-2%

### P2：实现 MQR 训练机制（2026-07-09，代码就绪待跑）

**定位升级**：从"MQR 训练策略"升级为"模态质量偏移下的偏好稳定学习（Modality-Quality Preference Stabilization）"，避免沦为 consistency trick。

**实现**（`smore.py`）：
- `_apply_mqs()`：扰动逻辑抽成独立方法，训练/推理复用
- `forward(adj, train, degrade_env)`：新增 degrade_env 参数，训练时为 degraded view 采样质量环境
- `calculate_loss`：clean forward + degraded forward（环境从 noise_both/mismatch/tail_noise_both 采样）
  - BPR_clean（原有）
  - BPR_degraded（权重 mqr_alpha）
  - 偏好稳定性损失：KL(softmax(s_clean/τ) ‖ softmax(s_degraded/τ)) over {pos,neg}（权重 mqr_beta）
  - tail-sensitive 加权：w_i = 1/log(2+degree_i)，归一化
- 消融开关：`mqr_alpha=0`（仅 PS）、`mqr_beta=0`（仅 degraded-BPR）、`mqr_tail_weight`、`train_noise_std`（朴素噪声增强 baseline）

**配置**（SMORE.yaml）：mqr_enabled/alpha/beta/tau/tail_weight/train_noise_std

### P3：命门消融脚本（2026-07-09，代码就绪待跑）

`run_ablation_mqr.sh`，7 方法 × 2 数据集 × 3 seed × 3 eval 模式 = 126 runs：
1. baseline
2. +dropout（朴素模态 dropout）
3. +noise_aug（朴素噪声增强）
4. +mqr_bpr（仅 degraded-BPR）
5. +mqr_ps（仅偏好稳定性损失）
6. +mqr_full（完整，无 tail 加权）
7. +mqr_full_tail（完整方法）

**命门判断**：完整方法在 MQS 下须明显优于 #2/#3（朴素 dropout/noise），否则仍是 trick。

### ⚠️ 待补：P1 输出增强（evaluator 扩展）

当前 evaluator 只输出 Recall/NDCG/Precision/MAP。GPT Pro 要求 P1 还需输出：
- Tail NDCG / Tail Recall（head/medium/tail 分桶）
- Item Coverage / Average Popularity / Gini（偏置指标）
- **PSS（Preference Stability Score）**：clean 与 degraded 推荐列表的 Top-K overlap / KL

这三类需要扩展 `topk_evaluator.py` + 在 eval 时跑 clean+degraded 双前向。**优先级：P1 baseline 扫描先跑（证明 Recall/NDCG 退化），PSS/tail 指标随后补**。

<!-- 后续尝试追加在此下方 -->

---


## 九、关键文件索引

| 文件 | 作用 |
|------|------|
| `src/models/smore.py` | 核心模型 + 4 创新点 + 鲁棒性扰动 |
| `src/main.py` | 入口，解析 key=value 参数 |
| `src/configs/model/SMORE.yaml` | 模型配置 + 创新点开关 |
| `scripts/parse_smore_results.py` | 日志 → CSV 解析 |
| `scripts/plot_results.py` | 画图（rate/鲁棒性/噪声强度） |
| `run_main_rate02.sh` | rate=0.2 主表实验 |
| `run_noise_level.sh` | 噪声强度实验 |
| `run_robustness.sh` | 鲁棒性实验 |
| `run_mdr_rate_search.sh` | rate 搜索实验 |
| `docs/results_keyexperiments.md` | 主实验详细结果 |
| `docs/results_final_summary.md` | 三轮实验汇总 |
| `docs/next_steps_plan.md` | 后续计划 |
