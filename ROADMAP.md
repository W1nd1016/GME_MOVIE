# ROADMAP：从随机 ID 到语义 ID —— 一个面向流式训练、可部署的物品冷启动方案

> 配套文档：[CODE_REVIEW.md](CODE_REVIEW.md)（问题清单与泄漏分析）。本文件只规划优化路线，不包含代码修改。
> 约束：单张 RTX 4090（24GB），学生项目，数据规模控制在十万级物品 / 百万级交互。

---

## 0. 定位与核心论点

**为什么不只做 GME + CoMeta + MWUF？**
三者分别发表于 2021–2023 年，单独叠加在面试官眼中是"老论文拼装"。本路线把它们作为**基线与模块**，而非卖点。

**项目主线（一句话）**：
新物品上线时，用**冻结的文本/多模态编码器 + RQ-VAE 语义 ID** 作为内容先验，让新物品与相似老物品共享 embedding；再用**基于早期交互用户的元学习 warm-up**（只用 support 数据，无标签泄漏）快速校准；全程在**按时间顺序的流式协议**下训练与评估，不依赖成对 batch，具备上线可行性。

**与业界趋势的对齐**：
- Semantic ID：Google/YouTube（arXiv 2306.08121）、快手 OneRec（arXiv 2502.18965）
- 流式冷启动元学习：PAM（arXiv 2411.11225，北大+快手）
- 冷启动 embedding 生成/校准：Meta-Embedding、GME、MWUF、CVAR、CoMeta

---

## 1. 阶段 0：干净的实验协议与基线（约 2 周，必做）

### 1.1 目标
修复 CODE_REVIEW.md 中 P0 级问题，建立可信、可复现的评估基准。

### 1.2 任务
| # | 任务 | 说明 |
|---|---|---|
| 0.1 | PyTorch 重写 | 替换 TF1 queue pipeline；数据读入改为 parquet/npz；固定随机种子 |
| 0.2 | 修复训练侧泄漏 | hist 只来自 support 集 A 的正反馈用户，作用于 r'（B 侧），见 CODE_REVIEW.md P0-1 正确流程 |
| 0.3 | 修复测试侧泄漏 | 冷启动测试不使用任何当前批次标签；warm-up 测试只使用已"发生过"的批次的用户 |
| 0.4 | 物品级 holdout | 基础模型训练数据（big_train）与 A/B/测试物品完全不相交 |
| 0.5 | 时间切分 | 每个新物品按时间切为 cold → warm-1 → warm-2 → test，补齐缺失的第三个 warm-up 批次 |
| 0.6 | 基线 | 随机初始化、Meta-Embedding、GME-P/G/A、MWUF、CVAR |
| 0.7 | 指标 | AUC、GAUC、logloss；3–5 个种子报告 mean±std；分头部/长尾物品报告 |

### 1.3 交付物
- 复现表：原始（泄漏）结果 vs 修复后结果，量化泄漏带来的虚高
- cold / warm-1 / warm-2 / warm-3 的 AUC 曲线

### 1.4 面试故事
"我发现原实验在训练与测试两侧都存在标签泄漏，重建了物品级、时间顺序的评估协议，修复后指标下降了 X，这才是真实基线。"

---

## 2. 阶段 1：语义 ID 作为内容先验（核心创新 1，约 2–3 周）

### 2.1 方法
1. **内容编码**：用冻结的文本编码器（如 bge-m3 / Qwen3-Embedding-0.6B）编码物品标题、简介、标签；MicroLens 可额外用冻结的 CLIP 编码封面。十万级物品在 4090 上数分钟完成。
2. **RQ-VAE 量化**：内容向量 → 3 层 × 256 码字的语义 ID（例如 `(17, 203, 88)`）；加冲突消解位。
3. **作为特征接入 CTR 模型**：每一层语义 ID 是一个 embedding 槽位，与（老物品的）原 ID embedding 融合：
   - 前缀共享：新物品与同前缀老物品共享前几层 embedding → 自动继承协同信号（GME "借邻居" 的思想，邻居由语义量化自动定义）
   - 融合方式消融：相加 / 拼接 / 门控
4. **与生成器结合**：Meta-Embedding 生成器输入改为 `内容向量 + 语义 ID embedding`，输出新物品 ID embedding 的初值。

### 2.2 消融实验
| 组别 | 新物品 embedding 来源 |
|---|---|
| A | 随机初始化 |
| B | Meta-Embedding 生成器（属性） |
| C | GME（属性 + 邻居） |
| D | 语义 ID（仅前缀共享） |
| E | 语义 ID + 生成器 |
| F | E + 内容向量直接输入 |

额外分析：码本大小（64/256/1024）、层数（2/3/4）、冲突率、各层前缀共享物品数分布。

### 2.3 可讨论点
- 层级共享带来泛化 vs 区分度下降的取舍
- 语义 ID 与随机 ID 在头部物品上的表现差异（YouTube 论文结论：头部不掉、长尾提升）
- 码本坍缩与利用率问题
- 和 OneRec / TIGER 中语义 ID 用途的区别（这里用于排序特征，不做生成）

---

## 3. 阶段 2：流式元学习 warm-up（核心创新 2，约 2–3 周）

### 3.1 方法
1. **早期用户聚合（正确版 hist）**：对每个新物品，只聚合"已发生"批次中的正反馈用户 embedding（CoMeta 思路），用注意力而非 `/20` 平均。
2. **MWUF 式校准**：以 `内容向量 + 早期用户聚合` 为条件，生成 scaling / shifting：
   `e_warm = e_cold ⊙ γ(cond) + β(cond)`
3. **流式元训练（借鉴 PAM）**：
   - 按时间顺序重放数据，不再构造成对 A/B batch
   - 按物品累计曝光数切分任务：0 次（纯冷）、1–10、10–50、50+，每个热度区间是一个任务
   - 元学习目标：让校准网络在每个热度区间都能快速提升
4. **冻结策略**：基础 DNN 冻结，只训练生成器 / 校准网络 / 语义 ID embedding，对比"同时微调底座"。

### 3.2 消融实验
- 去掉早期用户聚合 / 去掉 MWUF 校准 / 去掉热度分桶
- 成对 batch（原 Meta-Embedding 训练方式）vs 流式训练
- inner step 步长：原 cold_eta=1e-4 几乎无效，对比 1e-2、1e-1 与多步

### 3.3 可讨论点（面试重点："能不能上线"）
- 线上流式训练框架里，成对 batch 为什么难以实现
- 冷启动模块的推理开销：生成器只在物品入库时跑一次，校准网络开销可忽略
- 新物品从冷 → 温 → 热的切换策略（何时退出冷启动分支）

---

## 4. 阶段 3（可选加分）：轻量 LLM 增强（约 1 周）

| 方案 | 推荐度 | 说明 |
|---|---|---|
| LLM 生成物品描述 | ✅ 推荐 | 7B 级模型 4bit 量化（Qwen2.5-7B-Instruct 等），离线为物品补全描述/标签，再喂给阶段 1 编码器；对比原始文本 vs 增强文本的语义 ID 质量与 AUC |
| ColdLLM-lite 交互模拟 | ⚠️ 仅小规模 | 在 1k 物品子集上让 LLM 预测可能感兴趣的用户画像，作为伪早期交互；作为讨论素材，不作为主结果 |
| 训练/微调大模型 | ❌ 不做 | 超出单卡算力与项目周期 |

---

## 5. 数据集

| 数据集 | 用途 | 说明 |
|---|---|---|
| MovieLens-1M | 复现与健全性检查 | 与原项目、论文对齐；文本信息少，不适合展示语义 ID |
| **MicroLens-100K（主）** | 主实验 | 短视频场景，含原始标题、封面图、时间戳，最接近抖音/快手业务；github.com/westlake-repl/microlens |
| Amazon Reviews 单品类子集（备选） | 主实验备选 | 文本丰富、时间戳完整 |

---

## 6. 时间线（约 8–10 周）

| 周 | 内容 |
|---|---|
| 1–2 | 阶段 0：PyTorch 重写、协议修复、基线复现 |
| 3 | 数据迁移到 MicroLens，基线重跑 |
| 4–6 | 阶段 1：内容编码、RQ-VAE、语义 ID 接入与消融 |
| 7–8 | 阶段 2：流式元学习 warm-up 与消融 |
| 9 | 阶段 3（可选）：LLM 描述增强 |
| 10 | 结果整理、README、面试讲稿 |

---

## 7. 面试讲述结构（20–30 分钟）

| 时长 | 话题 |
|---|---|
| 5 分钟 | 问题定义：冷启动三阶段；原实验泄漏的发现与修复 |
| 8 分钟 | 为什么用语义 ID 而不只是生成器；量化设计取舍；与 YouTube Semantic ID、OneRec 的关系 |
| 7 分钟 | warm-up 元学习设计；流式训练与部署成本；为什么不用成对 batch |
| 5 分钟 | 消融结果、失败案例、长尾物品表现 |
| 5 分钟 | 延伸：LLM 增强、生成式推荐、线上会看哪些指标 |

**预判追问**：
- 为什么不直接用内容向量，而要量化成语义 ID？（可学习的共享 embedding、存储与线上特征一致性、协同信号可在 ID 空间更新）
- 语义 ID 冲突怎么处理？码本坍缩怎么办？
- MAML 一步内循环的意义？你的 cold_eta 设置合理吗？
- 和 ALDI（蒸馏）、CVAR（生成式）方法比，各自优劣？
- 线上如何 A/B 测试冷启动策略？（新物品曝光占比、新物品 CTR/完播、作者侧指标、大盘无损）

---

## 8. 与项目 2 的衔接

项目 2（MUSE 多模态长序列 CTR → 生成式推荐）复用本项目的 RQ-VAE 语义 ID 组件：
- 项目 1：语义 ID 作为**冷启动内容先验**
- 项目 2：语义 ID 用于**长序列 GSU 前缀检索**与**小规模 TIGER 生成式召回**

两个项目构成一条完整技术线：内容表征 → 语义 ID → 冷启动 → 序列建模 → 生成式推荐。

---

## 参考文献
- Pan et al. *Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings.* SIGIR 2019
- Ouyang et al. *Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction.* SIGIR 2021
- Zhu et al. *Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks (MWUF).* SIGIR 2021
- Zhao et al. *Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder (CVAR).* SIGIR 2022
- Huang et al. *Aligning Distillation For Cold-start Item Recommendation (ALDI).* SIGIR 2023
- Singh et al. *Better Generalization with Semantic IDs.* arXiv 2306.08121
- Rajput et al. *Recommender Systems with Generative Retrieval (TIGER).* NeurIPS 2023
- Deng et al. *OneRec.* arXiv 2502.18965
- *PAM: Popularity-Aware Meta-learning for Online Item Cold-start Recommendation.* arXiv 2411.11225
- *ColdLLM: Large Language Model Simulator for Cold-Start Recommendation.* WSDM 2025
- Ni et al. *A Content-Driven Micro-Video Recommendation Dataset at Scale (MicroLens).* arXiv 2309.15379
