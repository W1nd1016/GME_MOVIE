# GME Movie 项目全面说明

## 1. 项目概述

本项目是一个面向电影推荐冷启动问题的离线实验工程。它先用历史交互训练一个基础 CTR（点击率）预测 DNN，再为交互不足的新电影生成更合适的 `movie_id` embedding，以改善冷启动电影的推荐效果。

项目采用 TensorFlow 1.x 静态计算图风格，核心实验分为两个阶段：

1. 运行 `base_dnn.py`，训练并保存基础推荐模型；
2. 运行 `warm_up_dnn_base.py`，加载基础模型，通过电影属性、用户历史和元学习式的一步梯度更新生成冷启动 embedding，并比较原始 embedding 与生成 embedding 的 AUC。

仓库中的 `ReadMe` 也只给出了上述运行顺序。项目没有 Web 服务、接口层或在线推理模块，本质上是一个研究/教学性质的单机训练实验。

## 2. 目录结构

```text
gme_movie/
├── base_dnn.py                 # 第一阶段：基础 CTR DNN 的训练、测试和保存
├── warm_up_dnn_base.py         # 第二阶段：冷启动 embedding 生成及元学习实验
├── ctr_funcs.py                # 数据输入、AUC/RMSE 等公共函数
├── config_dnn.py               # 基础 DNN 的配置（部分配置在主脚本中重复定义）
├── config_gme.py               # 冷启动/GME 实验配置
├── ReadMe                      # 极简运行说明
├── data/
│   ├── big_train_main.csv
│   ├── big_train_main.tfrecord
│   ├── test_oneshot_a.csv
│   └── test_oneshot_a.tfrecord
├── data_with_hist/
│   ├── test_oneshot_a_hist.{csv,tfrecord}
│   ├── test_oneshot_b_hist.{csv,tfrecord}
│   ├── train_oneshot_a_w_ngb_hist.{csv,tfrecord}
│   ├── train_oneshot_b_w_ngb_hist.{csv,tfrecord}
│   └── test_test_w_ngb_hist.tfrecord
├── tmp/
│   ├── dnn/                    # 已保存的 TensorFlow checkpoint
│   └── dnn_0801_0900.txt       # 多次基础模型实验结果
├── __pycache__/                # Python 3.6/3.7 遗留字节码
└── .vscode/settings.json       # VS Code Python 环境设置
```

`.codex` 当前为空文件，不参与项目运行。

## 3. 业务与特征含义

每条样本表示一次用户与电影之间的二分类交互，首列 `label` 为 `0/1`。代码注释给出的 8 个特征槽位如下：

| 槽位 | 特征 | 编码形式 | 实际占用列数 |
|---:|---|---|---:|
| 0 | `user_id` | one-hot ID | 1 |
| 1 | `movie_id` | one-hot ID | 1 |
| 2 | gender | one-hot ID | 1 |
| 3 | age | one-hot ID | 1 |
| 4 | occupation | one-hot ID | 1 |
| 5 | release year | one-hot ID | 1 |
| 6 | title | multi-hot | 5 |
| 7 | genres | multi-hot | 5 |

因此基础样本由 `1 个 label + 6 个单值特征 + 2×5 个多值特征 = 17 列`组成。所有特征已预先映射到同一整数 ID 空间；`0` 用作 padding，embedding 表大小为 `(11134 + 1) × 10`。

冷启动阶段默认把 `movie_id`（`tar_idx = [1]`）作为需要生成 embedding 的目标，使用电影的 `release year`、`title` 和 `genres`（`attr_idx = [5, 6, 7]`）作为内在属性。每条含邻居的样本还带有 10 个相似老电影，每个邻居同样占 16 个特征列。

## 4. 数据集与数据契约

### 4.1 仓库内数据统计

以下统计直接来自当前 CSV 文件：

| 数据集 | 行数 | 列数 | 正样本 | 负样本 | 用途 |
|---|---:|---:|---:|---:|---|
| `data/big_train_main.csv` | 765,669 | 17 | 473,137 | 292,532 | 基础 DNN 训练 |
| `data/test_oneshot_a.csv` | 22,540 | 17 | 9,867 | 12,673 | 基础 DNN 测试 |
| `test_oneshot_a_hist.csv` | 22,540 | 37 | 9,867 | 12,673 | 第一批 warm-up 样本及 20 个历史 ID |
| `test_oneshot_b_hist.csv` | 22,540 | 37 | 9,638 | 12,902 | 第二批 warm-up 样本及 20 个历史 ID |
| `train_oneshot_a_w_ngb_hist.csv` | 21,160 | 197 | 12,091 | 9,069 | 元学习数据集 A |
| `train_oneshot_b_w_ngb_hist.csv` | 21,160 | 197 | 12,321 | 8,839 | 元学习数据集 B |

`test_test_w_ngb_hist.tfrecord` 没有对应 CSV，仓库内无法直接核对其行数和标签分布。

### 4.2 三种记录布局

- 基础记录：17 列，即 `label + 16 个当前样本特征`。
- 带历史记录：37 列，即基础记录后追加 `20 个用户历史电影 ID`。
- 带邻居和历史记录：197 列，即 `label + 16 个当前样本特征 + 10×16 个邻居特征 + 20 个历史电影 ID`。

TFRecord 中并非把所有内容存进一个数组，而是使用：

```text
label: int64[1]
data:  int64[16] 或 int64[176]
hist:  int64[20]（仅历史版本）
```

训练管线使用 `shuffle_batch`，测试和第二阶段数据管线使用普通 `batch`。数据读取依赖 TensorFlow 1.x 的 `string_input_producer`、`TFRecordReader`、queue runner 和 `Coordinator`。

## 5. 第一阶段：基础 DNN

入口为 `base_dnn.py`。

### 5.1 网络结构

1. 6 个单值特征分别查找 10 维 embedding；
2. 两个 multi-hot 特征各查找最多 5 个 embedding，并在槽位内求和；
3. 得到 `8×10=80` 维向量；
4. 输入无 bias 的全连接网络 `80 → 256 → 128 → 1`；
5. 隐藏层使用 ReLU 和 dropout，输出为 logit；
6. 使用 sigmoid cross entropy 和 Adam 优化器。

默认 `keep_prob=1.0`，所以当前配置实际上没有启用 dropout。模型约有 164,726 个可训练参数，其中 embedding 参数为 111,350 个。

### 5.2 默认训练参数

| 参数 | 值 |
|---|---:|
| epoch | 1 |
| batch size | 128 |
| learning rate | 0.001 |
| embedding 维度 | 10 |
| 随机种子 | 123 |
| 日志间隔 | 2,000 step |
| 优化器 | Adam |
| GPU 显存上限 | 30%，按需增长 |

训练完成后，脚本只保存 `emb_mat` 和三层 `weight_dict`，不保存优化器状态。checkpoint 位于 `./tmp/dnn/`。随后脚本在测试集上计算 AUC、RMSE 和平均交叉熵。

现有实验日志 `tmp/dnn_0801_0900.txt` 记录了多次运行，测试 AUC 大致为 `0.7062～0.7144`；最后一次记录为 AUC `0.7096`、RMSE `0.4771`、loss `0.64638`。

## 6. 第二阶段：冷启动 embedding 生成

入口为 `warm_up_dnn_base.py`。该脚本先从 `./tmp/dnn/` 恢复基础 embedding 和 DNN 权重，基础模型参数不参与元学习更新。

### 6.1 embedding 生成器

代码中的生成流程由三个模块组成：

1. **属性生成模块**：取当前电影的年、标题、类型 embedding，形成 30 维属性表示，经简单的特征加权、`W_meta(30×10)` 和 `tanh` 生成 10 维候选 `movie_id` embedding。
2. **对比/SENET 模块**：用 `W_SENET(3×3)` 对属性表示加权，计算两个掩码视图生成结果的余弦相似度，并将 `exp(-cosine)` 加入总损失。
3. **用户历史去噪模块**：查找 20 个历史电影 embedding，将候选 embedding 一并聚合，再经 `hist_vars_1(10×10)` 和 ReLU 得到最终 embedding。

虽然代码实现了邻居拆分、邻居 embedding 和属性选择函数，但当前真正执行的生成路径没有调用邻居 embedding 函数。因此“10 个相似老电影”存在于输入数据中，却没有实际进入最终生成器计算。

### 6.2 元学习式训练流程

第二阶段使用数据集 A 和 B：

1. 在 A 上生成新 embedding，并计算 `cold_loss_a`；
2. 求损失对生成 embedding 的梯度，执行一次显式更新：`pred_emb_a_new = pred_emb_a - cold_eta × gradient`；
3. 把更新后的 embedding 用于 B 的预测，计算 `cold_loss_b`；
4. 优化总损失：

```text
meta_loss = alpha × cold_loss_a
          + (1 - alpha) × cold_loss_b
          + contrastive_loss
```

默认 `alpha=0.1`、`cold_eta=1e-4`、`meta_eta=5e-3`、meta batch size 为 60。实际更新的参数只有 `W_meta`、`W_SENET` 和 `hist_vars_1`，共 409 个参数。

训练前先用随机/基础 `movie_id` embedding 直接测试得到 `auc`，训练后用生成 embedding 测试得到 `meta auc`，最终输出两者作为冷启动效果对比。

### 6.3 可选 warm-up 流程

脚本内 `warm_up_bool=False`，默认关闭。开启后会：

1. 用 `test_oneshot_a_hist` 更新基础 embedding 并测试；
2. 再用 `test_oneshot_b_hist` 更新并测试；
3. 重新加载基础模型；
4. 完成元训练后，把生成的新电影 embedding 写回 `emb_mat`；
5. 重复两轮 warm-up，对比普通更新和 GME 初始化后的效果。

warm-up 仅更新 `emb_mat`，不更新 DNN 全连接层。

## 7. 公共工具 `ctr_funcs.py`

该模块提供：

- 全局 AUC、按用户分组 AUC；
- bias、RMSE、采样率校正 RMSE；
- 二维列表展平、文件行数统计；
- CSV 输入管线；
- 普通 TFRecord 训练/测试管线；
- 同时返回 `label`、`data`、`hist` 的 TFRecord 管线；
- 时间打印函数。

基础训练实际使用 TFRecord 管线，CSV 管线及部分指标函数当前没有被两个入口脚本调用。

## 8. 配置关系

- `base_dnn.py` 导入了 `config_dnn.py`，但核心超参数又在脚本顶部重新硬编码；实际以脚本内变量为准。
- `warm_up_dnn_base.py` 大量读取 `config_gme.py`，但数据路径和 checkpoint 恢复路径仍有多处直接硬编码。
- `config_gme.py` 中的 `meta_mode='GME-A'` 仅被打印，没有控制任何模型分支。
- `config_gme.py` 的 `model_loading_addr='./tmp/dnn_1011_1705/'` 未被实际使用；脚本固定从 `./tmp/dnn/` 恢复。
- 配置中若干 `./data/...w_ngb.tfrecord` 路径在仓库中不存在，但当前脚本改为硬编码读取 `data_with_hist/` 下的现有文件，所以默认路径下不触发这些缺失项。

## 9. 运行条件与运行方式

### 9.1 预期依赖

从源码可确定至少需要：

- Python（遗留 `pyc` 表明项目曾在 Python 3.6/3.7 下运行）；
- NumPy；
- scikit-learn；
- TensorFlow 1.x，或经过兼容改造的 TensorFlow 2.x `compat.v1` 环境；
- 可选 CUDA/GPU，代码也可在 CPU 上运行。

仓库没有 `requirements.txt`、Conda 环境文件或安装脚本。当前工作环境为 Python 3.12.3，且没有安装 NumPy、TensorFlow、scikit-learn；因此本次只完成了源码、数据和语法层面的核查，没有实际重新训练。五个 Python 文件均通过 Python 3.12 的语法编译检查。

### 9.2 原始运行顺序

在项目根目录、满足旧版依赖的环境中运行：

```bash
python base_dnn.py
python warm_up_dnn_base.py
```

第一条命令会覆盖/更新 `tmp/dnn/` checkpoint；第二条命令依赖该 checkpoint。基础训练需要读取约 54 MB 的 TFRecord，第二阶段还会读取约 60 MB 的历史/邻居 TFRecord，且使用 TensorFlow 队列机制，运行时应从项目根目录启动。

## 10. 当前实现中的重要注意事项

1. **TensorFlow 版本兼容性**：`tf.Session`、`tf.placeholder`、`tf.train.string_input_producer` 等接口属于 TensorFlow 1.x；现代 TensorFlow 2.x 不能原样运行。
2. **依赖不可复现**：没有锁定 Python 和库版本，也没有自动化测试。
3. **邻居特征未实际使用**：代码解析了 10 个邻居，也定义了相关函数，但生成器只使用当前电影属性和用户历史。
4. **配置与执行脱节**：多个配置字段被主脚本硬编码覆盖或完全未使用，修改配置文件不一定生效。
5. **A/B batch 强绑定**：A 上一步更新得到的 embedding 被直接用于 B，要求两个 batch 在行数和语义上正确配对；代码没有显式校验这种配对关系。
6. **历史均值分母可疑**：`get_hist_mean` 聚合了 20 个历史 embedding 和 1 个生成 embedding，却固定除以 20，而不是 21 或有效非零项数。
7. **padding embedding 可训练**：ID 0 虽经 mask 后不参与前向输出，但 embedding 第 0 行没有被显式固定为零。
8. **对比损失是 batch 级标量**：实现把整个 batch 展平成两个长向量后计算一次余弦值，并非逐样本对比学习。
9. **保存不完整**：第二阶段默认 `save_model_ind=0`，而且脚本没有实现元参数的保存逻辑；运行结果主要输出到终端。
10. **命名和注释存在遗留**：代码混用 ad/item/movie，`get_new_emebdding` 有拼写错误，部分注释描述的“完整方案”与实际简化实现不一致。
11. **异常处理依赖队列耗尽**：训练和测试均以 `OutOfRangeError` 作为正常结束条件，调试时需区分数据耗尽和真正异常。

## 11. 一次完整实验的数据流

```text
big_train_main.tfrecord
        │
        ▼
基础特征 embedding → 80维拼接 → DNN → CTR loss
        │
        └────────── 保存 emb_mat + DNN weights 到 tmp/dnn/
                                      │
                                      ▼
          冷启动样本属性 ──→ 属性生成器 ──┐
          用户历史电影 ────→ 历史聚合 ────┼→ 新 movie_id embedding
                                          │
数据集 A → cold loss → 一步 embedding 梯度更新
                                          │
数据集 B ────────────────────────────────→ meta loss
                                          │
                                          ▼
                         更新生成器参数并在测试集比较 AUC
```

## 12. 项目定位总结

这个仓库实现的是一个“基础 CTR 模型 + 冷启动 ID embedding 生成器”的实验原型。基础 DNN 部分结构清晰、已有 checkpoint 和历史指标；第二阶段包含属性建模、用户历史聚合、对比损失和 MAML 风格的一步更新，但邻居特征与多种 `meta_mode` 尚未真正接入执行路径，工程化和可复现性也较弱。

如果继续维护，优先级最高的工作应是：固定依赖环境、统一配置入口、把数据管线迁移到 `tf.data` 或 PyTorch、补齐第二阶段参数保存与测试，并验证邻居特征和 A/B 样本配对是否符合预期算法。
