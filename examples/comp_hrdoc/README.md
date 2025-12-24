# comp_hrdoc - Detect-Order-Construct 文档结构重构

基于论文 [Detect-Order-Construct](https://arxiv.org/html/2401.11874v2) 的实现，
使用 LayoutXLM 作为多模态基座模型。

## 论文与代码对应关系

### 架构总览 (论文 Figure 2)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        论文 Section 4.2                              │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   4.2.3 Detect   │   4.2.4 Order    │       4.2.5 Construct         │
│   (区域内顺序)    │   (区域间顺序)    │       (层级结构)               │
├──────────────────┼──────────────────┼───────────────────────────────┤
│ intra_region.py  │    order.py      │       construct.py            │
└──────────────────┴──────────────────┴───────────────────────────────┘
```

### 详细对应

| 论文章节 | 核心内容 | 对应代码文件 | 说明 |
|---------|---------|-------------|------|
| **4.2.3 Detect** | Intra-region Head | `models/intra_region.py` | 行级别后继预测 + Union-Find 分组 |
| **4.2.4 Order** | Order Head | `models/order.py` | 区域级别阅读顺序预测 |
| **4.2.5 Construct** | Tree Construction | `models/construct.py` | 层级结构构建 (父子/兄弟关系) |
| 4.2.1 Text Encoder | LayoutXLM | `models/backbone.py` | 多模态特征编码 |
| 4.2.2 Visual Encoder | ResNet + FPN | `models/backbone.py` | 视觉特征提取 (LayoutXLM 内置) |

---

## 模块详解

### 1. Detect 模块 (Section 4.2.3)

**论文描述**：Intra-region Head 预测行级别的后继关系，使用 Union-Find 算法将行分组为区域。

**代码实现**：`models/intra_region.py`

```python
# 核心类
- IntraRegionModule     # 行级别后继预测器 (Transformer + Biaffine)
- IntraRegionLoss       # 后继预测损失函数
- UnionFind             # 分组算法
- group_lines_to_regions()  # 基于预测将行分组为区域
- predict_successors()  # 从 logits 预测后继

# 训练脚本
scripts/train_intra.py  # 独立训练 Detect 模块

# 数据加载
data/line_level_loader.py  # 行级别数据加载器
```

**输入输出**：
- 输入：行级别特征 `[batch, num_lines, hidden_size]`
- 输出：后继预测 logits `[batch, num_lines, num_lines]`

### 2. Order 模块 (Section 4.2.4)

**论文描述**：Order Head 使用三层 Transformer 预测区域间的阅读顺序。

**代码实现**：`models/order.py`

```python
# 核心类
- OrderHead             # 区域级别阅读顺序预测
- OrderModule           # 完整 Order 模块 (含 LayoutXLM)

# 训练脚本
scripts/train.py --config configs/order.yaml

# 数据加载
data/comp_hrdoc_loader.py  # 区域级别数据加载器
```

**输入输出**：
- 输入：区域特征 `[batch, num_regions, hidden_size]`
- 输出：阅读顺序预测

### 3. Construct 模块 (Section 4.2.5)

**论文描述**：使用 Biaffine Attention 预测父子关系，通过树插入算法构建层级结构。

**代码实现**：`models/construct.py`

```python
# 核心类
- ConstructHead         # 父子/兄弟关系预测
- ConstructLoss         # 构建损失函数
- tree_insert()         # 树插入算法

# 融合于联合训练
```

**输入输出**：
- 输入：区域特征 `[batch, num_regions, hidden_size]`
- 输出：父节点预测、兄弟关系预测

---

## 目录结构

```
comp_hrdoc/
├── models/                     # ★ 核心模型 (对应论文 Section 4.2)
│   ├── intra_region.py         # [Detect] 4.2.3 Intra-region Head
│   ├── order.py                # [Order] 4.2.4 Reading Order
│   ├── construct.py            # [Construct] 4.2.5 Tree Construction
│   ├── backbone.py             # 4.2.1-4.2.2 LayoutXLM 基座封装
│   ├── embeddings.py           # 位置/语义嵌入
│   ├── heads.py                # 任务预测头封装
│   ├── doc_model.py            # 文档级别联合模型
│   ├── order_only.py           # 仅 Order 的简化模型
│   ├── build.py                # 模型构建工厂
│   └── modules/
│       └── pooling.py          # 特征池化
│
├── data/                       # 数据处理
│   ├── line_level_loader.py    # [Detect] 行级别数据加载
│   ├── comp_hrdoc_loader.py    # [Order/Construct] 区域级别数据加载
│   ├── dataset.py              # 数据集基类
│   └── collator.py             # Batch 组织
│
├── configs/                    # 配置文件
│   ├── order.yaml              # Order 模块配置
│   └── doc.yaml                # 文档模型配置
│
└── scripts/                    # 入口脚本
    ├── train_intra.py          # [Detect] 训练 Intra-region Head
    ├── train.py                # [Order] 训练 Order 模块
    ├── train_doc.py            # 联合训练
    ├── predict.py              # 推理入口
    └── evaluate.py             # 评估入口
```

---

## 训练流程

### 方式一：分阶段独立训练

```bash
# Step 1: 训练 Detect (Intra-region Head)
python examples/comp_hrdoc/scripts/train_intra.py --env test

# Step 2: 训练 Order
python examples/comp_hrdoc/scripts/train.py --config configs/order.yaml --env test

# Step 3: 训练 Construct (联合训练或独立)
python examples/comp_hrdoc/scripts/train_doc.py --env test
```

### 方式二：端到端联合训练

```bash
python examples/comp_hrdoc/scripts/train_doc.py --env test
```

---

## 评估指标

| 模块 | 指标 | 说明 |
|-----|------|-----|
| Detect | Region F1 | 区域分组的准确率 |
| Detect | Successor Accuracy | 后继预测准确率 |
| Order | Reading Order Accuracy | 阅读顺序准确率 |
| Construct | TEDS | 树编辑距离相似度 |
| Construct | Parent Accuracy | 父节点预测准确率 |

---

## 参考

- 论文: [Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis](https://arxiv.org/abs/2401.11874)
- 基座模型: [LayoutXLM](https://huggingface.co/microsoft/layoutxlm-base)
