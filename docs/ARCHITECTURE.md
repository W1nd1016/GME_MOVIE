# 新代码架构

本文件描述新 PyTorch 代码的**结构**：模块分工、数据流、接口约定和不变量的测试归属。旧 TF1 代码见 [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)，阶段与任务编号见 [ROADMAP.md](../ROADMAP.md)。

截至 2026-11-29，0.1 和 0.4 已实现数据转换、物品级留出和共用 CTR 底座的预训练。0.5 已为 1127 部测试电影生成三轮独立的反馈批次，供 0.3 按轮读取并更新电影表示。

---

## 1. 模块地图

| 路径 | 职责 | 引入任务 | 状态 |
|---|---|---|---|
| `scripts/convert_movielens.py` | 读取旧 TFRecord，输出按用途分开的样本 NPZ、物品元数据表和清单 | 0.1 | 已实现 |
| `src/coldstart/data/schema.py` | 统一定义特征槽位、样本划分（split）及各划分的允许用途（purpose），包括三轮反馈的更新权限 | 0.1、0.4、0.5 | 已实现 |
| `src/coldstart/data/movielens.py` | 加载 NPZ 和物品表，为底座训练与元训练提供按已验证电影清单筛选的 Dataset | 0.1、0.4 | 已实现 |
| `src/coldstart/models/base_dnn.py` | CTR 底座：按 schema 查表、槽位池化、MLP 输出 logit | 0.1 | 已实现 |
| `scripts/train_base.py`、`configs/*.yaml` | 运行合成冒烟或真实底座预训练，保存权重、配置和重载记录 | 0.1、0.4 | 已实现 |
| `src/coldstart/data/splits.py`、`scripts/prepare_holdout.py` | 生成和核验物品级留出清单，统计伪新与测试电影的邻居覆盖，通过 CLI 发布两份派生产物 | 0.4 | 已实现 |
| `src/coldstart/data/splits.py`、`scripts/prepare_warm_arrivals.py` | 按测试电影核对 A/B 的交互键，固定 B 的两组成员，通过 CLI 发布三轮反馈和清单 | 0.5 | 已实现 |
| `src/coldstart/data/visible_hist.py` | 按"已开放批次"重建正反馈用户集合，训练侧与测试侧共用 | 0.2、0.3 | 规划 |
| `src/coldstart/eval/` | AUC、GAUC、logloss；四个测点评估；多种子汇总 | 0.7 | 规划 |
| `src/coldstart/models/generators/` | 0.2 以最小属性生成器打通元训练与评分；0.6 扩充 Meta-Embedding、GME-P/G/A、MWUF、CVAR 等基线 | 0.2、0.6 | 规划 |
| `src/coldstart/semantic_id/` | 冻结内容编码、RQ-VAE 训练与语义 ID 分配 | 阶段 1 | 规划 |
| `src/coldstart/models/calibration.py` | warm-up 校准：早期用户聚合 + scaling / shifting | 阶段 2 | 规划 |

## 2. 数据流

```
旧 TFRecord（只读）
  └─ convert_movielens.py ──▶ artifacts/movielens/（gitignore）
                                ├─ 六份样本 NPZ（每种用途一份，见 §3.1）
                                ├─ items.npz（物品元数据表，见 §3.2）
                                └─ manifest.json（源文件、用途、行数）
合成冒烟数据 ─▶ MovieLensDataset ─▶ train_base.py / BaseDNN ─▶ runs/0.1-smoke/

base_candidate + meta_a/b 的电影 ID + items.npz
  └─ prepare_holdout.py ─▶ artifacts/holdouts/0.4/{holdout.json, neighbors.json}
base_candidate + holdout.json ─▶ base_train ─▶ train_base.py / BaseDNN ─▶ runs/0.4-base/
meta_a/b + holdout.json ─▶ 伪新电影 A/B
warm_a/b + holdout.json 中的 test_movies
  └─ prepare_warm_arrivals.py ─▶ artifacts/arrivals/0.5/{warm_1.npz, warm_2.npz, warm_3.npz, manifest.json}

后续任务
伪新电影 A/B ─▶ 0.2 属性生成、A 上适配与校准 ─▶ 冻结底座对 B 评分
warm_1 → warm_2 → warm_3 ─▶ 0.3 用累计可见的反馈更新历史与电影表示
各测点电影表示 + final_eval ─▶ 共用底座评分 ─▶ 0.7 指标 ─▶ 0.6 基线对比
```

## 3. 关键接口与约定

### 3.1 数据用途（split）

| 用途 | 来源文件 | 行数 | 允许的用法 |
|---|---|---:|---|
| `base_candidate` | `data/big_train_main.tfrecord` | 765,669 | 经 0.4 物品留出后才能训练底座 |
| `meta_a`、`meta_b` | `data_with_hist/train_oneshot_{a,b}_w_ngb_hist.tfrecord` | 21,160 ×2 | 0.4 按伪新电影筛选后，分别用于元训练 support / query |
| `warm_a`、`warm_b` | `data_with_hist/test_oneshot_{a,b}_hist.tfrecord` | 22,540 ×2 | 仅 `partition`，供 0.5 划分到达批次 |
| `warm_1`、`warm_2`、`warm_3` | `artifacts/arrivals/0.5/warm_{1,2,3}.npz` | 22,540 / 11,270 / 11,270 | 仅 `warm_update`，每次读取一轮新增反馈（见 §3.5） |
| `final_eval` | `data_with_hist/test_test_w_ngb_hist.tfrecord` | 123,787 | **只用于评分**。任何训练、更新、早停入口收到它都要报错 |

转换器按源文件行序写出 `<split>.npz`，含 `label`（uint8，形状 `[N]`，值为 0/1）和 `features`（int32，形状 `[N, 16]`，0 为填充）。`manifest.json` 的 `splits[split]` 记录 `source` 和 `rows`。旧 `hist` 不转换，邻居 ID 进入物品表。

`MovieLensDataset(root, split, purpose, *, holdout_path=None)` 按用途校验访问权限、行数、字段和取值，每条返回 `(features: torch.long, label: torch.float32)`。以下三个建模入口必须传入 `holdout.json` 路径。Dataset 用当前源文件的电影 ID、物品表和清单中的种子重算分组，核对清单后按原始行序返回该组的全部交互。[数据读取实现](../src/coldstart/data/movielens.py)

| split / purpose | 样本来源 | 返回的电影 |
|---|---|---|
| `base_train / base_train` | `base_candidate.npz` | 底座组 |
| `meta_a / meta_support` | `meta_a.npz` | 伪新组 |
| `meta_b / meta_query` | `meta_b.npz` | 伪新组 |

划分入口以 `partition` 用途读取完整候选集、元训练 A/B 或测试侧 A/B。建模入口只允许以 `final_eval / score` 读取最终集。物品划分、底座预训练和三轮反馈生成均独立于最终集。[用途映射](../src/coldstart/data/schema.py)

### 3.2 物品元数据表 `items.npz`

`load_items(root)` 返回由以下 int32 数组组成的字典。按 `movie_id` 排序，每部电影一行，覆盖全部 2185 部电影（1058 老 + 1127 新）：

- 属性：year、title ×5、genres ×5（取自样本中的物品侧槽位，每部电影恒定）
- `neighbors`：10 个邻居电影 ID，只有物品侧信息，不含邻居行里的用户槽位

已核实的数据事实（2026-11-28，全量解析 TFRecord）：邻居在每部电影内恒定，全部是老电影；11,270 条测试电影邻居行均为 `big_train` 中真实存在的（用户，老电影）交互，其用户均不是该测试电影在最终评估集中的用户。物品表只保留邻居电影 ID。原始选邻规则及其上架时可用性未经核实。测试电影的邻居只存在于 `final_eval` 源文件中，由转换脚本抽取到物品表。后续模型从物品表查邻居。

### 3.3 特征槽位与模型

- 槽位由 `schema.py` 统一定义：6 个单值槽位（user、movie、gender、age、occupation、year），2 个 multi-hot 槽位（title、genres，各 5 位，0 为填充）。共享 `11135 × 10` embedding 表。
- 底座接口：`forward(features) -> logits`，每条交互输出一个 logit。输入维度由 schema 计算，不在模型里写死。
- 0.2 将提供可微的电影向量评分路径，兼容现有 `forward(features)`。输入的生成或适配向量替换 movie 槽位，底座权重冻结。0.4 返回的 106 部伪新电影将在 0.2 按电影分组。85 部元训练电影的 B 损失经电影向量回传给生成器，21 部元验证电影的 B 仅用于计算验证损失。0.3 复用该路径做逐轮评分，0.6 的各方法共用相同的批次与评分条件。[元训练与验证约定](DECISIONS.md#02-元训练电影数量与验证)
- 阶段 1 将语义 ID 作为新增槽位加入 schema，并带新槽位重新预训练底座（见 ROADMAP 2.1 第 5 点）。阶段 2 的流式校准网络使用已到达的反馈修正电影向量。

### 3.4 配置与运行产物

在仓库根目录执行下列命令生成可复用的物品划分及邻居统计。CLI 核对候选集与元训练 A/B 的老电影 ID 集合一致，再一次发布两份 JSON。同样输入可重复执行。

```bash
.venv/bin/python scripts/prepare_holdout.py --root artifacts/movielens --output artifacts/holdouts/0.4 --split-seed 42
```

| 派生产物 | 内容与使用者 |
|---|---|
| `artifacts/holdouts/0.4/holdout.json` | `version`、`algorithm`、`split_seed` 及升序的 `base_movies`、`pseudo_movies`、`test_movies`。固定种子 42 对排序后的老电影 ID 做 PCG64 置换，得到 952 部底座电影和 106 部伪新电影。测试组为物品表中其余 1127 部电影。三个建模入口共用此清单。 |
| `artifacts/holdouts/0.4/neighbors.json` | 分 `pseudo`、`test` 两组记录每部目标电影的可用邻居位置数、0 至 10 个可用邻居的电影数分布，以及落入伪新组的位置数、总位置数和占比。0.6 的 GME 基线可据此解释邻居覆盖。原始邻居名单保存在物品表中。 |

`scripts/train_base.py --config <yaml> --run-dir <目录>` 使用同一 Dataset、DataLoader 和底座完成训练与权重重载。两种配置共用 `seed`、`batch_size`、`learning_rate`。合成冒烟配置指定 `data.split: synthetic_smoke`、`data.rows` 和 `steps`。真实底座配置指定 `data.split: base_train`、`data.root`、`data.holdout_path` 和 `epochs`，每轮完整遍历底座组交互并保留尾批。[冒烟配置](../configs/base_smoke.yaml) · [真实底座配置](../configs/base_holdout.yaml)

训练目录含 `config.yaml`、`weights.pt`（可直接载入 `BaseDNN` 的 state dict）及 `metrics.json`。指标记录代码提交、训练种子、实际行数、训练样本数、更新步数、训练 BCE 和重载检查。真实底座另存已验证的 `holdout.json` 副本，将其 SHA-256 记入指标。后续方法从 `runs/0.4-base/weights.pt` 加载共用底座。[实验登记表](EXPERIMENTS.md) · [训练命令](plans/0.4-item-holdout.md#验收标准)

### 3.5 测试电影的三轮反馈

在仓库根目录运行以下命令生成 `artifacts/arrivals/0.5/`。CLI 从 0.4 清单取得 `test_movies`，核对 A、B 各自恰好覆盖这些电影，每部各有 20 个互异的用户，且两批按（用户，电影）互斥。三轮共用以下划分，不依赖标签或训练种子。[生成入口](../scripts/prepare_warm_arrivals.py) · [划分约定](DECISIONS.md#05-测试电影的反馈到达批次)

```bash
.venv/bin/python scripts/prepare_warm_arrivals.py --root artifacts/movielens --holdout-path artifacts/holdouts/0.4/holdout.json --output artifacts/arrivals/0.5
```

| 测点 | 本轮读取 | 每部电影新增 / 累计交互 |
|---|---|---:|
| cold | 无反馈文件，历史为空 | 0 / 0 |
| warm-1 | `warm_1.npz`，A 的全部交互 | 20 / 20 |
| warm-2 | `warm_2.npz`，按固定哈希顺序排在前十位的 B 用户 | 10 / 30 |
| warm-3 | `warm_3.npz`，B 的其余十位用户 | 10 / 40 |

三份 NPZ 与 §3.1 的转换样本格式相同，各轮保留来源文件的相对行序。目录内的 `manifest.json` 记录 `algorithm`（`gme-warm-v1`）、0.4 清单原始字节的 `holdout_sha256`，以及每个 `splits[split]` 的 `source`、`rows`、`rows_per_movie`。四个文件整体发布，相同输入可复跑。

例如，`MovieLensDataset("artifacts/arrivals/0.5", "warm_2", "warm_update")` 从清单和第二轮文件读取当轮新增的 11,270 条交互。0.3 负责依序开放这些批次，从各电影累计开放的正反馈用户重建 `hist`，再更新电影表示。四个测点都在同一完整的 `final_eval` 上评分。0.5 的独立验收只取最终集的交互键，核对三个批次和最终集共四组均无重复，六个两两交集均为空。[协议测试](../tests/test_warm_arrivals.py)

## 4. 不变量与测试归属

| 不变量（[AGENTS.md](../AGENTS.md) §6） | 守护测试 | 状态 |
|---|---|---|
| 旧 `hist` 不进入建模数组 | `test_legacy_hist_is_excluded` | 已实现（0.1） |
| 训练入口拒绝 `final_eval` 与未留出的 `base_candidate` | `test_train_rejects_ineligible_splits` | 已实现（0.1） |
| 物品表恒定、邻居只含老电影、与源文件一致 | `test_item_table_matches_sources` | 已实现（0.1） |
| 源文件行序、特征、尾批完整；加载时校验建模字段 | `test_conversion_preserves_rows_and_fields`、`test_rejects_malformed_samples` | 已实现（0.1） |
| 最终集只能以评分用途打开 | `test_final_eval_rejects_non_scoring_access_before_io`、`test_train_rejects_ineligible_splits` | 已实现（0.1） |
| 槽位求和、零填充掩码、输出维度与可扩展槽位 | `test_base_dnn_slot_pooling_and_logits`、`test_base_dnn_uses_extended_slot_schema` | 已实现（0.1） |
| 冒烟训练的批次、loss、权重可复现，重载推理一致 | `test_seeded_smoke_and_checkpoint_reload` | 已实现（0.1） |
| 三组电影两两互斥并覆盖物品表，划分独立于标签及行序，建模按清单保留同源样本（§6.1） | `test_holdout_is_reproducible_and_label_independent`、`test_holdout_views_match_source_rows`、`test_modeling_requires_verified_holdout` | 已实现（0.4） |
| 物品划分与底座预训练独立于最终评分集（§6.5） | `test_holdout_does_not_open_final_eval`、`test_real_base_epoch_and_checkpoint_reload` | 已实现（0.4） |
| 邻居覆盖只计入底座组的 ID 向量 | `test_neighbor_coverage_counts_only_base_movies` | 已实现（0.4） |
| 底座按 epoch 遍历已留出数据，重载权重可复现 logits，复核所用清单摘要 | `test_real_base_epoch_and_checkpoint_reload`、`test_train_rejects_mixed_steps_and_epochs` | 已实现（0.4） |
| 每条交互所属轮次独立于标签、源行序与训练种子。各轮保留源样本及相对行序，每部累计 0 / 20 / 30 / 40 条（§6.2–3） | `test_b_assignment_uses_only_interaction_keys`、`test_arrival_views_match_source_rows` | 已实现（0.5） |
| 更新入口仅读取指定轮次，原始 A/B 仅供划分，异常输入或输出冲突保全已有产物 | `test_warm_update_reads_only_requested_round`、`test_arrivals_reject_invalid_sources` | 已实现（0.5） |
| 三个到达批次与完整最终集按（用户，电影）互斥（§6.3） | `test_real_arrivals_are_disjoint_from_final_eval` | 已实现（0.5） |
| 预测时 `hist` 只含已开放批次的正反馈（§6.4） | 待 0.2、0.3 计划命名 | 规划（0.2、0.3） |
| 指标实现可由手算样例核对（§6.6） | 待 0.7 计划命名 | 规划（0.7） |
