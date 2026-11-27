# 新代码架构

本文件描述新 PyTorch 代码的**结构**：模块分工、数据流、接口约定和不变量的测试归属。旧 TF1 代码见 [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)，阶段与任务编号见 [ROADMAP.md](../ROADMAP.md)。

**当前状态（2026-11-28）**：0.1 已实现数据转换、按用途读取、CTR 底座和 CPU 冒烟训练。后续模块在表中标为“规划”，路径在对应任务的计划中确认。

---

## 1. 模块地图

| 路径 | 职责 | 引入任务 | 状态 |
|---|---|---|---|
| `scripts/convert_movielens.py` | 读取旧 TFRecord，输出按用途分开的样本 NPZ、物品元数据表和清单 | 0.1 | 已实现 |
| `src/coldstart/data/schema.py` | 特征槽位定义、数据用途（split）枚举与各用途的允许用法。全项目唯一的定义处 | 0.1 | 已实现 |
| `src/coldstart/data/movielens.py` | 加载 NPZ 和物品表，提供 Dataset；按用途做访问检查 | 0.1 | 已实现 |
| `src/coldstart/models/base_dnn.py` | CTR 底座：按 schema 查表、槽位池化、MLP 输出 logit | 0.1 | 已实现 |
| `scripts/train_base.py`、`configs/*.yaml` | 训练入口与实验配置 | 0.1 | 已实现 |
| `src/coldstart/data/splits.py` | 物品级留出（底座电影 / 伪新电影）；三轮到达批次划分 | 0.4、0.5 | 规划 |
| `src/coldstart/data/visible_hist.py` | 按"已开放批次"重建正反馈用户集合，训练侧与测试侧共用 | 0.2、0.3 | 规划 |
| `src/coldstart/eval/` | AUC、GAUC、logloss；四个测点评估；多种子汇总 | 0.7 | 规划 |
| `src/coldstart/models/generators/` | 冷启动 embedding 生成器与基线（Meta-Embedding、GME-P/G/A、MWUF、CVAR） | 0.6 | 规划 |
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

后续任务的数据流：
artifacts ─▶ data/movielens.py（Dataset + 用途检查）
          ─▶ splits.py（物品留出、到达批次）─▶ visible_hist.py（可见历史）
          ─▶ 训练入口：底座 / 生成器 / 校准网络 ─▶ runs/<实验ID>/
final_eval ─▶ 只进入 eval/ 的评分入口 ─▶ runs/<实验ID>/metrics.json
```

## 3. 关键接口与约定

### 3.1 数据用途（split）

| 用途 | 来源文件 | 行数 | 允许的用法 |
|---|---|---:|---|
| `base_candidate` | `data/big_train_main.tfrecord` | 765,669 | 经 0.4 物品留出后才能训练底座 |
| `meta_a`、`meta_b` | `data_with_hist/train_oneshot_{a,b}_w_ngb_hist.tfrecord` | 21,160 ×2 | 元训练 support / query |
| `warm_a`、`warm_b` | `data_with_hist/test_oneshot_{a,b}_hist.tfrecord` | 22,540 ×2 | 测试电影的 warm-up 更新批次 |
| `final_eval` | `data_with_hist/test_test_w_ngb_hist.tfrecord` | 123,787 | **只用于评分**。任何训练、更新、早停入口收到它都要报错 |

转换器按源文件行序写出 `<split>.npz`，含 `label`（uint8，形状 `[N]`，值为 0/1）和 `features`（int32，形状 `[N, 16]`，0 为填充）。`manifest.json` 的 `splits[split]` 记录 `source` 和 `rows`。旧 `hist` 不转换，邻居 ID 进入物品表。

`MovieLensDataset(root, split, purpose)` 按 schema 检查用途，再加载并校验行数、字段和取值；每条返回 `(features: torch.long, label: torch.float32)`。例如，读取最终集需指定 `Split.FINAL_EVAL` 和 `Purpose.SCORE`。0.1 的底座训练入口仅生成和消费 `synthetic_smoke`；0.4 将接入通过物品级留出检查的 `base_train`。

### 3.2 物品元数据表 `items.npz`

`load_items(root)` 返回由以下 int32 数组组成的字典。按 `movie_id` 排序，每部电影一行，覆盖全部 2185 部电影（1058 老 + 1127 新）：

- 属性：year、title ×5、genres ×5（取自样本中的物品侧槽位，每部电影恒定）
- `neighbors`：10 个邻居电影 ID，只有物品侧信息，不含邻居行里的用户槽位

已核实的数据事实（2026-11-28，全量解析 TFRecord）：邻居在每部电影内恒定，全部是老电影；11,270 条测试电影邻居行均为 `big_train` 中真实存在的（用户，老电影）交互，其用户均不是该测试电影在最终评估集中的用户。物品表只保留邻居电影 ID。原始选邻规则及其上架时可用性未经核实。测试电影的邻居只存在于 `final_eval` 源文件中，由转换脚本抽取到物品表，下游一律从物品表读取，不再读取 `final_eval` 样本。

### 3.3 特征槽位与模型

- 槽位由 `schema.py` 统一定义：6 个单值槽位（user、movie、gender、age、occupation、year），2 个 multi-hot 槽位（title、genres，各 5 位，0 为填充）。共享 `11135 × 10` embedding 表。
- 底座接口：`forward(features) -> logits`，每条交互输出一个 logit。输入维度由 schema 计算，不在模型里写死。
- 扩展点：阶段 1 的语义 ID 作为新增槽位加入 schema，底座带新槽位重新预训练（见 ROADMAP 2.1 第 5 点）；生成器和校准网络只替换或修正 movie 槽位的 embedding，不改底座接口。

### 3.4 配置与运行产物

- `configs/base_smoke.yaml` 指定 `data.split`、合成样本数 `data.rows`、`seed`、`batch_size`、`steps` 和 Adam 的 `learning_rate`。
- `scripts/train_base.py --config <yaml> --run-dir <目录>` 生成合成 NPZ，经 Dataset 和 DataLoader 训练并重载权重。运行目录含 `config.yaml`、`weights.pt` 和 `metrics.json`，其中记录代码提交、种子、训练步数、逐步 loss 和重载检查。复现命令见 [0.1 计划](plans/0.1-pytorch-port.md#拟执行的验收)。

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
| 底座电影、伪新电影、测试电影两两不相交（§6.1） | 待 0.4 计划命名 | 规划（0.4） |
| 三个到达批次与最终评估集按（用户，电影）互斥（§6.3） | 待 0.5 计划命名 | 规划（0.5） |
| 预测时 `hist` 只含已开放批次的正反馈（§6.4） | 待 0.2、0.3 计划命名 | 规划（0.2、0.3） |
| 指标实现可由手算样例核对（§6.6） | 待 0.7 计划命名 | 规划（0.7） |
