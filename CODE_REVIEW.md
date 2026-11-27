# GME_MOVIE 代码与数据审查报告

> 目标：为推荐算法岗面试深挖、优化这个冷启动项目做准备。
> 方法：逐行阅读全部源码，用纯 Python 脚本解析所有 CSV 和 TFRecord（包括没有 CSV 的 `test_test_w_ngb_hist.tfrecord`，共 123,787 行），交叉核对 ID、样本配对和 `hist` 字段的含义。
> 限制：本机没有 TensorFlow，**没有重新训练**。文中的数字都来自数据统计和已有日志，不是复现出来的 AUC。

---

## 0. 一句话结论

这个项目的骨架是 **MovieLens-1M 上的 GME / Meta-Embedding 风格冷启动实验**（基础 DNN + ID embedding 生成器 + MAML 式一步更新），整体可以讲。但是：

1. **`hist` 字段在训练侧和测试侧都造成了标签泄漏**（见 P0-1）。各数据集之间的样本互不重叠，但 hist 把"谁点了这部电影"带进了模型输入，所以目前 "meta auc" 的提升不可信。这是面试时最致命、也最容易被追问的点；
2. 代码注释自己写着"实现是简易版，面试时应该说……"。**代码实际做的和简历上"说的"有好几处对不上**（邻居未使用、SENET 不参与推理、对比学习没有负样本、MAML 步长几乎为 0）。

先把这两类问题修掉、重新跑实验，项目才经得起深挖。

---

## 1. 数据的真实面貌（实测）

### 1.1 特征 ID 空间（所有特征共用一张 `11135 × 10` 的 embedding 表）

| 槽位 | 特征 | ID 范围 | 取值数 |
|---|---|---|---|
| 0 | user_id | 5020–11059 | 6040（= ML-1M 用户数） |
| 1 | movie_id | 29–2210 | 老电影 1058 / 新电影 1127 |
| 2 | gender | 8–9 | 2 |
| 3 | age | 1–7 | 7 |
| 4 | occupation | 2214–2234 | 21 |
| 5 | year | 11061–11133 | 67 |
| 6 | title（multi-hot ×5） | 2235–5013 | 1547 个词 |
| 7 | genres（multi-hot ×5） | 10–28 | 19 |

### 1.2 数据划分（典型的 GME 论文划分方式）

| 文件 | 行数 | 电影集合 | 每部电影样本数 | 正样本率 | 作用 |
|---|---|---|---|---|---|
| `big_train_main` | 765,669 | 老电影 1058 部（每部 ≥300 次评分） | 300–3428 | **61.8%** | 训练基础 DNN |
| `train_oneshot_a/b_w_ngb_hist` | 21,160 ×2 | 同上 1058 部老电影 | 每部 20 条 | 57%/58% | 元训练 A/B |
| `test_oneshot_a/b(_hist)` | 22,540 ×2 | 新电影 1127 部 | 每部 20 条 | 43.8%/42.8% | warm-up 第 1/2 轮 |
| `test_test_w_ngb_hist` | 123,787 | 同上 1127 部新电影 | 20–239（中位 102） | **45.1%** | 最终测试 |

已核对的事实：

- 老电影和新电影的 movie_id **完全不重叠**，所以新电影的 ID embedding 在基础训练里从没收到过梯度，一直是初始化值（N(0, 0.01)）。这就是"冷启动"本身；
- A 和 B 的**每一行电影都一一对应**（逐行 movie_id 100% 相同，352 个 meta batch 全部对齐），所以 "A 上更新、B 上评估" 的配对是对的；
- A/B 里的 (user, movie) 对 **100% 也出现在 `big_train_main` 中**。也就是说，元训练用的是基础模型见过的样本，这和论文做法一致，但值得在面试时主动说明；
- `test_oneshot_a.csv` 等于 `test_oneshot_a_hist.csv` 的前 17 列，`data/` 和 `data_with_hist/` 是同一批数据；
- 新电影标题里有 **41% 的 title token 在老电影中从没出现过**，它们的 embedding 也是随机初始化的；year 覆盖 98%，genre 覆盖 100%；
- 测试集用户 100% 在训练集中出现过，所以这是**纯 item 冷启动**，没有 user 冷启动；
- 训练集和测试集正样本率差距明显（62% vs 45%），而模型**没有任何 bias 项**（见 P1-1）。

### 1.3 `hist` 字段的真实含义（最重要的发现）

文档和代码注释把它称作"用户历史电影序列"，**实际并不是**：

- `hist` 的取值全部落在 **user_id 区间**（5020–11059），它是 **"对这部电影给出正反馈的 20 个用户"**；
- 同一部电影的所有样本，`hist` 完全相同（每部电影只有 1 种 hist），所以它是 item 级特征；
- 用这 20 个用户查 embedding、求和以后放进 movie 槽位，本质上是 **"用点过它的用户 embedding 来表示 item"**，类似 item-CF 或 two-tower 的思路；
- 思路本身合理，但目前取值来源造成了训练侧和测试侧的标签泄漏，详见 P0-1。

---

## 2. P0：必须先修，否则结论站不住

### P0-1 `hist` 导致的标签泄漏（训练侧 + 测试侧）

#### 前提：样本层面没有泄漏

6 个数据集的 (用户, 电影) 样本对两两求交集，结果如下：

| | big_train | A | B | test_oneshot_a | test_oneshot_b | test_test |
|---|---|---|---|---|---|---|
| big_train | — | 21,160（A 全部在内） | 21,160（B 全部在内） | 0 | 0 | 0 |
| A | | — | 0 | 0 | 0 | 0 |
| B | | | — | 0 | 0 | 0 |
| test_oneshot_a | | | | — | 0 | 0 |
| test_oneshot_b | | | | | — | 0 |

- 测试样本没有进入任何训练；三个测试集只含新电影，电影 ID 和训练集完全不重叠；
- 测试用户 100% 在训练集中出现过，这是 item 冷启动的正常设定；
- A/B 是 big_train 的子集，不算测试泄漏，但会让元训练的模拟失真，见本节末尾"训练侧补充"。

**问题出在 `hist` 的取值来源。** 它存的是"对这部电影给过正反馈的 20 个用户"，同一部电影的所有行都相同。生成器把这些用户的 embedding 聚合进新电影的 ID embedding，于是"谁点了这部电影"这种由标签导出的信息就成了模型的输入。

#### 训练侧（元训练 A → B）

现在的流程（[warm_up_dnn_base.py:445-467](warm_up_dnn_base.py:445)）：

```
r      = Denoise( G(A 的属性), hist )    # hist 在适配之前就已经混进 r
loss_a = CE(DNN(A, r), y_A)
r'     = r - cold_eta · ∇_r loss_a
loss_b = CE(DNN(B, r'), y_B)
```

实测：

| 检查项 | 结果 |
|---|---|
| hist 用户的构成 | 约 57% 是 A 的正样本用户，33% 是 B 的正样本用户，10% 是 big_train 中该电影的其他正样本用户；**100% 都是正样本** |
| A 中当前用户在 hist 里的行 | **12,091 行，正好是 A 的全部正样本**，标签 100% 为 1 |
| B 中当前用户在 hist 里的行 | **6,905 行**（约占 B 正样本的 56%），标签 100% 为 1 |

问题有两个：

1. **数据来源错误，A 和 B 的标签都泄漏了。**
   - A：所有正样本用户都在 hist 里，生成器只要学会"当前用户在不在 hist 里"，就能把 `loss_a` 压得很低；由 `loss_a` 算出的梯度 ∇ 和 r' 也被污染了；
   - B：hist 里含有 B 的正样本用户，query 集的标签被提前放进了 support 集的输入。MAML "在 A 上适配、在 B 上检验泛化"的前提因此不成立。
2. **位置错误：hist 放在了适配之前（放进了 r）。** hist 属于"看过 support 集之后"才有的信息，和梯度一样，应该作用在适配之后的 r' 上，再拿去 B 上评估。放在 r 里，`loss_a` 就不再衡量纯属性生成的冷启动质量。

正确做法：

```
r      = G(A 的属性)                     # 零交互：只用属性
loss_a = CE(DNN(A, r), y_A)              # 不含 hist，无泄漏
r'     = r - cold_eta · ∇_r loss_a       # support 信息 1：梯度
r''    = Denoise(r', hist_A)             # support 信息 2：只取 A 的正样本用户
loss_b = CE(DNN(B, r''), y_B)            # query：只看 B 的标签
meta_loss = α·loss_a + (1-α)·loss_b (+ 对比损失)
```

- `hist_A` 只能由 A 中标签为 1 的用户构成（平均每部电影约 11.5 个），不能包含 B 或 big_train 的数据；
- 不足 20 个的部分补 0，聚合时按有效个数做 masked mean；
- `cold_eta` 要调大（现在是 1e-4，适配这一步几乎不起作用，见 P0-3）。

**训练侧补充：基础 DNN 已经见过 A/B 的样本。** 这不会让测试指标虚高，但冻结的 MLP 和用户 embedding 已经拟合过 A/B 这些具体样本，所以 `loss_a`、`loss_b` 比真实冷启动时偏低，给生成器的训练信号偏乐观，造成元训练和元测试分布不一致。这是 MetaEmb/GME 论文的原始设定，不算错误。如果要更严格，有两种做法：
- 样本留出：从 big_train 中删掉 A/B 的样本，再训练基础 DNN；
- item 留出（最严格）：把老电影分成两组，一组训练基础 DNN，另一组作为"伪新电影"构造 A/B，它们的所有样本都不进 big_train。

#### 测试侧

各数据集的角色应该按时间顺序固定下来：

```
时间 →  [上架：冷启动] → [第1批交互 test_oneshot_a] → [第2批交互 test_oneshot_b] → [评估集 test_test]
可用信息：  只有属性          + a 的梯度 / hist              + b 的梯度 / hist（a∪b）
评估：   test_test            test_test                      test_test
```

`test_test` 在所有阶段都只做评估。从 `test_test` 的行里读电影**属性**没有问题（同一部电影每行的属性都相同，等同于查电影信息表），不能从中拿的是由标签导出的信息（hist）。

`test_test` 中 hist 的实测情况：

| 检查项 | 结果 |
|---|---|
| hist 用户来自 `test_test` 中同一电影的样本 | **100%，且全部是正样本** |
| hist 用户来自 `test_oneshot_a/b` | 0% |
| 当前用户本人在 hist 里的行 | **19,770 行（16%），标签 100% 为 1**；其余行正样本率为 34.7% |

**1）完全冷启动测试（默认 `warm_up_bool=False`）**

| 步骤 | 数据 | 现在的做法 | 输出 |
|---|---|---|---|
| A：直接测试 | `test_test` | 新电影使用从没被训练过的 ID embedding（相当于随机初始化） | `auc`，干净 |
| C2：元测试 | `test_test` | 生成器输入是 `test_test` 的属性 + **`test_test` 自带的 hist**，得到 r，不做适配 | `meta auc`，**有泄漏** |

问题：
- 标签泄漏：hist 来自测试集自身的正样本用户，16% 的行能直接"认出自己"；
- 概念错误：零交互阶段本来就不应该有新电影的 hist。

正确做法：
- 只用属性生成 r，在 `test_test` 上评估；
- 如果想在冷启动阶段利用用户侧信息，可以用 **10 个邻居老电影在 big_train 中的正样本用户**作为"代理 hist"。这些都是历史数据，没有泄漏。训练侧要同样处理：A 中电影的代理 hist 也取它邻居的正样本用户。

**2）warm-up 测试（`warm_up_bool=True`）**

| 步骤 | 数据 | 现在的做法 | 输出 |
|---|---|---|---|
| B1 → B2 | `test_oneshot_a` 训练 → `test_test` 评估 | 不用生成器，用 Adam 对整张 `emb_mat` 跑一个 epoch | `warm1 auc`，干净 |
| B3 → B4 | `test_oneshot_b` 训练 → `test_test` 评估 | 接着继续训练 | `warm2 auc`，干净 |
| D0 | `test_test` | 用 `test_test` 的属性 + **`test_test` 的 hist** 生成新电影 embedding，写回 `emb_mat` | — |
| D1 → D2 | `test_oneshot_a` 训练 → `test_test` 评估 | 从 D0 写回的 embedding 出发微调 | `warm1 meta auc`，**有泄漏** |
| D3 → D4 | `test_oneshot_b` 训练 → `test_test` 评估 | 同上 | `warm2 meta auc`，**有泄漏** |

问题：
1. **D0 泄漏**：D0 虽然没有训练，但把 `test_test` 的标签信息（谁点了这部电影）编码进了 `emb_mat`，之后又在 `test_test` 上评估。判断标准是测试标签导出的信息有没有进入预测时的模型状态，与是否训练无关；
2. **warm-up 没有用 hist**：真实交互到来后，本该用这批交互的正样本用户做去噪，现在完全没用；
3. **更新方式和训练不一致**：元训练学的是"一步、步长为 `cold_eta`"的适配，warm-up 却是 Adam 跑一整个 epoch；
4. **更新范围过大**：Adam 更新整张 `emb_mat`，连用户 embedding 和老电影 embedding 也被改了，应该只更新新电影那几行；第二次 `saver.restore` 也没有重置 Adam 状态。

正确做法：

| 阶段 | 用到的数据 | 做法 | 评估 |
|---|---|---|---|
| 冷启动 | 只用属性（可选：邻居的代理 hist） | r = G(属性) | `test_test` → cold auc |
| warm-up 1 | `test_oneshot_a` | 梯度更新 r 得到 r'；用 `test_oneshot_a` 的正样本用户做 hist 去噪，得到 r'' | `test_test` → warm1 auc |
| warm-up 2 | `test_oneshot_b`，hist 取 a∪b | 在 r'' 上再更新一次，hist 用累计的正样本用户 | `test_test` → warm2 auc |

- 全程冻结：基础 DNN 的 MLP、用户 embedding 和属性 embedding、元训练好的生成器和去噪模块（`W_meta`、`hist_vars_1`）；只更新新电影的 ID embedding；
- 更新规则和元训练保持一致（一步，步长 `cold_eta`）；
- warm-up 2 超出了训练时 A→B 一次适配的模拟范围。可以把元训练扩展成 A→B→C 两步（原始 GME 有 a/b/c 三份数据，`config_gme.py` 里还留着 `train_oneshot_c` 的路径），和两个 warm-up 阶段对齐；
- 基线（不用生成器）必须走完全相同的数据、更新规则和冻结范围，唯一的区别只能是初始 embedding 和是否使用去噪模块；
- `test_oneshot_a` 中每部新电影平均只有约 8.8 个正样本，hist 必须用 masked mean。

> 面试时：主动讲出"我发现了 hist 的标签泄漏，并按 support/query 的语义重新设计了训练和测试流程"，比报一个虚高的 AUC 加分得多。

### P0-2 基线不公平

`auc`（基线）的做法是：新电影直接用随机初始化的 ID embedding（约等于 0 向量）。`meta auc` 则额外用到了属性和（泄漏的）用户信息。至少需要补上下面这些基线，才能说明生成器有用：

| 基线 | 说明 |
|---|---|
| Random / Zero init | 已有 |
| 老电影 ID embedding 的均值 | 最简单的强基线 |
| 同 genre 老电影的均值 | 利用属性的无参基线 |
| 只用属性的生成器（MetaEmb） | 去掉 hist |
| GME-P / G / A（用邻居） | 代码中有 `meta_mode` 占位，但没有实现 |
| DropoutNet | 经典对比方法 |
| 多次随机种子 | 见 P1-4，基础 AUC 自身的波动就有 ±0.004 |

### P0-3 代码实现与"宣称的方法"不一致（面试追问必中）

| 宣称的方法 | 代码实际行为 | 位置 |
|---|---|---|
| 用 k 个相似老电影（邻居）的 ID embedding 生成 | `partition_input_w_ngb` 解析了 10 个邻居（已核对，100% 是老电影），但**结果没有被任何计算使用** | [warm_up_dnn_base.py:388](warm_up_dnn_base.py:388) |
| SENET 对属性做加权 | 主生成路径里 SENET 那一行**被注释掉**了，改成"每个槽位 embedding 的均值 /10"作为权重；`W_SENET` 只在对比损失分支里训练，**不影响推理时生成的 embedding** | [warm_up_dnn_base.py:304-309](warm_up_dnn_base.py:304) |
| 对比学习优化 embedding 空间 | 两个"视图"分别是 year-only 和 title+genre，经同一个 `W_meta` 映射；把**整个 batch 展平成一个长向量**算一次 cosine，loss 为 `exp(-cos)`；**没有负样本**，有塌缩风险，而且不是逐样本的对比学习 | [warm_up_dnn_base.py:319-358](warm_up_dnn_base.py:319) |
| MAML 一步梯度适配 | `cold_eta = 1e-4`，损失又是对 batch 取 `reduce_mean`，每个样本的梯度还要再乘 1/60，所以 `pred_emb_a_new ≈ pred_emb_a`，这一步**几乎什么都没做**；`cold_loss_b` 实际上只是"同一个生成 embedding 在另一批样本上的 loss" | [warm_up_dnn_base.py:455-456](warm_up_dnn_base.py:455) |
| 用户序列去噪 | 实际是 item 的正反馈用户聚合（见 1.3），并且有泄漏 | [warm_up_dnn_base.py:361](warm_up_dnn_base.py:361) |
| `meta_mode = 'GME-A'` | 只被 print 出来，不控制任何分支 | [config_gme.py](config_gme.py) |

---

## 3. P1：影响效果和结论可信度

1. **全网络没有 bias**：三层 matmul 都去掉了 bias，模型无法直接学全局 CTR。训练集正样本率 62%、测试集 45%，校准偏差会直接体现在 logloss 和 RMSE 上。建议至少给输出层加 bias。[base_dnn.py:95](base_dnn.py:95)
2. **生成 embedding 的值域和预训练 embedding 不一致**：最终输出经过 `ReLU(hist·W)`，结果全部 ≥ 0，而且没有约束尺度；预训练的 ID embedding 是有正有负的高斯分布。DNN 在没见过的输入分布上做推理。建议去掉 ReLU，或者加一个与老 item embedding 分布对齐的正则（例如 MSE 或 KL 到老 ID embedding）。[warm_up_dnn_base.py:368](warm_up_dnn_base.py:368)
3. **`get_hist_mean` 的分母是 20**，而实际参与求和的有 21 项（20 个 hist + 1 个生成 embedding）；padding 为 0 的位置（29 部新电影 hist 全为 0）也被算进了分母。应改成按有效个数做 masked mean。
4. **结果有不确定性**：同样 `seed=123` 运行 9 次，基础 AUC 在 0.7062–0.7144 之间波动（极差约 0.008），原因是 `shuffle_batch` 是多线程的，不是确定性的。任何小于约 0.005 的提升都要用多种子的均值±标准差来报。
5. **`tf.train.batch` 默认丢弃最后一个不满的 batch**（`allow_smaller_final_batch=False`），测试集最后约 100 行没有被评估。影响不大，但需要知道。
6. **新电影有 41% 的 title token 是随机 embedding**，会给生成器输入引入噪声。可以考虑：只对出现频次够高的 token 建 embedding，或者用预训练文本向量（标题、简介做 BERT/word2vec）替代。
7. **评估指标单一**：只看全局 AUC。`ctr_funcs.cal_group_auc` 已经实现但没被调用。推荐场景建议补 **GAUC（按用户分组）**、logloss，以及按 warm-up 阶段（0/20/40 条交互）报告的曲线。
8. **对比损失和 CE 损失的权重都是 1**，尺度不匹配：`exp(-cos)` 的范围是 [e⁻¹, e]，需要单独的系数来调。

---

## 4. P2：工程问题（不影响结论，但面试可能会被看代码）

- TensorFlow 1.x 的 queue runner 写法，现在的环境跑不起来；没有 `requirements.txt`；
- 配置和代码脱节：`config_dnn.py` 导入了但没用；`config_gme.py` 里的数据路径、`model_loading_addr` 都被硬编码覆盖，`warm_up_bool` 也在脚本里被重新赋值；
- warm-up 阶段用 Adam 更新**整张 `emb_mat`**（包括用户和老电影），并没有只更新新 item；第二次 `saver.restore` 也没有重置 Adam 的状态；
- 结果表头有 bug：第 4 列应该是 `warm2 auc`，现在写成了重复的 `warm1 auc`（[warm_up_dnn_base.py:752](warm_up_dnn_base.py:752)）；
- 元学习的参数没有保存；在 `try` 里放 dummy 除法来"通过语法检查"的写法很奇怪（[warm_up_dnn_base.py:709](warm_up_dnn_base.py:709)）；
- 命名混乱（ad/item/movie、`xiangsi_SENET`、`get_new_emebdding`），而且有"面试时应该这么说"之类的注释。**发给面试官看之前一定要删掉**；
- `__pycache__`、`.DS_Store`、checkpoint 以及约 200 MB 的数据都提交进了仓库，建议加上 `.gitignore`，数据改用脚本生成或用 Git LFS。

---

## 5. 面试深挖：你应该能回答的问题

**问题定义**
- 为什么新 item 的 ID embedding 不好？（没有梯度，停在初始化附近；而 DNN 对 ID 槽位的输入分布是按训练好的 embedding 学出来的）
- item 冷启动和 user 冷启动有什么区别？这个项目属于哪一种？（纯 item 冷启动，测试用户 100% 是老用户）
- 冷启动有哪几个阶段？（零交互的 cold，少量交互的 warm-up）分别用什么方法？

**方法本身**
- MetaEmb 和 GME 的区别：GME 用邻居老 item 的 ID embedding，通过图注意力（P/G/A 三种变体）来生成。
- 为什么 meta loss 是 `α·loss_a + (1-α)·loss_b`？α 控制什么？（loss_a 管"冷启动时直接好用"，loss_b 管"更新一步后能快速适应"）
- 二阶梯度：`tf.gradients` 里 `pred_emb_a_new` 对 `W_meta` 求导，会经过 `cold_emb_grads` 产生二阶项。first-order MAML 有什么取舍？
- `cold_eta` 应该怎么选？现在的值为什么让 MAML 退化了？
- 为什么要冻结基础 DNN，只训练生成器？（避免破坏老 item，生成器只要学会"往已有空间里映射"）

**数据与评估**（最能体现能力的部分）
- hist 字段的泄漏是怎么发现的？如何修？修完以后指标变化多少？
- 如何保证 A/B 配对正确？（同一个 item，样本不相交）
- 为什么要用 GAUC？推荐场景中 AUC 的局限是什么？
- 离线提升怎么转化成线上收益？冷启动的线上指标看什么？（新 item 曝光占比、CTR、首日点击、进入正常流量池的速度）

**工程落地**
- 线上怎么用？（新 item 入库时调生成器写入 embedding 表；积累交互后切换成正常训练的 embedding，或者做平滑过渡）
- 生成器输入的 item 属性来自哪里？对实时性有什么要求？

---

## 6. 建议的优化路线（按优先级）

| 阶段 | 工作 | 预期产出 |
|---|---|---|
| 1. 修数据和流程 | 按 P0-1 的"正确做法"重做：训练侧 hist 只取 A 的正样本用户并作用在 r' 上；测试侧冷启动只用属性，warm-up 的 hist 只取 `test_oneshot_a`（以及 a∪b）的正样本用户 | 一个干净、可信的实验设定 |
| 2. 能跑 | 迁移到 PyTorch（或 TF2 + `tf.data`），统一配置，固定种子，多种子跑 5 次 | 可复现的基线：均值±标准差 |
| 3. 补基线 | 第 P0-2 节的表，加上 GAUC 和 logloss | 一张完整的对比表 |
| 4. 补实现 | 把邻居接进来（GME-P/G/A 至少做一个 attention 版）；SENET 真正进入生成路径；对比学习改成 InfoNCE（batch 内其他 item 作负样本，逐样本计算） | 方法和描述一致 |
| 5. 调关键超参 | `cold_eta`（例如 1e-2 到 1）、α、对比损失权重；输出层去掉 ReLU，加分布对齐正则 | 每个模块的消融结果 |
| 6. 加分项 | 标题用预训练文本 embedding；warm-up 曲线（0/1/2 轮）；分析生成 embedding 和真实 embedding 的 cosine 或 t-SNE | 面试时可以展示的图 |

**一张面试时能讲的消融表**应该长这样：

| 方法 | AUC | GAUC | logloss |
|---|---|---|---|
| Random init | | | |
| 老 item 均值 | | | |
| + 属性生成器 | | | |
| + 邻居（GME） | | | |
| + SENET | | | |
| + InfoNCE 对比 | | | |
| + 无泄漏的交互用户聚合（warm 阶段） | | | |

---

## 附：本次核查用到的关键数字

- A/B 逐行 movie_id 一致：100%；meta batch 对齐：352/352
- 老 / 新电影 ID 交集：0
- 测试 hist 用户来自 test_test 同一电影的正样本：100%
- 测试行中"当前用户在 hist 里"：16.0%，这些行的正样本率是 100%
- 新电影 title token 覆盖率：59.2%
- 基础 DNN 9 次运行的测试 AUC：0.7062–0.7144
