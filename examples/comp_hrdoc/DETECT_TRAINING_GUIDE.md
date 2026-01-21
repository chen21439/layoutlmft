# Detect Stage Training Guide

## 📋 概述

本指南说明如何在 `train_doc.py` 中集成 Detect 阶段（4.2）进行完整的 DOC（Detect-Order-Construct）训练。

## 🎯 新增功能

### 1. FullDOCPipeline

完整的三阶段 Pipeline：
- **Detect (4.2)**: IntraRegionHead + LogicalRoleHead
- **Order (4.3)**: InterRegionOrderHead + RelationTypeHead
- **Construct (4.4)**: TreeRelationHead

### 2. 独立保存机制

每个模块可以独立保存/加载：
```
output_dir/
├── detect_module.pt          # 完整 DetectModule
├── intra_head.pt            # 独立的 IntraRegionHead
├── role_head.pt             # 独立的 LogicalRoleHead
├── order_module.pt          # OrderModule
├── construct_module.pt      # ConstructModule
└── full_doc_pipeline.pt     # 完整模型
```

---

## 🚀 训练命令

### 方案 1: 完整端到端训练（从头开始）

```bash
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --use-construct \
    --batch-size 1 \
    --num-epochs 20 \
    --detect-weight 1.0 \
    --order-weight 1.0 \
    --construct-weight 1.0 \
    --num-roles 10 \
    --learning-rate 5e-5
```

**说明**：
- `--use-detect`: 启用 Detect 阶段（4.2）
- `--num-roles`: 逻辑角色类别数（默认 10）
- `--detect-weight`: Detect 损失权重

---

### 方案 2: 使用预训练的 classify head，只训练 intra_head

```bash
# Step 1: 先独立训练 Stage1 的分类头（如果还没有）
python examples/comp_hrdoc/scripts/train_stage1.py \
    --env test \
    --num-epochs 20

# Step 2: 使用预训练分类，训练 Detect + Order + Construct
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --detect-checkpoint /path/to/stage1_cls_head.pt \
    --freeze-detect \
    --use-construct \
    --num-epochs 20
```

**说明**：
- `--detect-checkpoint`: 加载预训练的 DetectModule
- `--freeze-detect`: 冻结 DetectModule，只训练 Order + Construct

---

### 方案 3: 逐阶段训练（推荐）

```bash
# Step 1: 训练 Detect 阶段（intra_head + role_head）
python examples/comp_hrdoc/scripts/train_intra.py \
    --env test \
    --num-epochs 20

# Step 2: 使用训练好的 Detect，训练 Order + Construct
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --detect-checkpoint ./artifacts/exp_xxx/intra/detect_module.pt \
    --freeze-detect \
    --use-construct \
    --num-epochs 20

# Step 3: 可选 - 联合微调所有阶段
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --detect-checkpoint ./artifacts/exp_xxx/intra/detect_module.pt \
    --order-checkpoint ./artifacts/exp_xxx/doc/order_module.pt \
    --construct-checkpoint ./artifacts/exp_xxx/doc/construct_module.pt \
    --use-construct \
    --num-epochs 5 \
    --learning-rate 1e-5
```

---

### 方案 4: 只训练 Detect + Order（不含 Construct）

```bash
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --no-construct \
    --num-epochs 20
```

---

## 📝 完整参数说明

### Detect 阶段参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-detect` | flag | False | 启用 Detect 阶段（4.2） |
| `--detect-checkpoint` | str | None | 预训练 DetectModule 路径 |
| `--freeze-detect` | flag | False | 冻结 DetectModule |
| `--num-roles` | int | 10 | 逻辑角色类别数 |
| `--detect-weight` | float | 1.0 | Detect 损失权重 |
| `--detect-num-layers` | int | 1 | Detect Transformer 层数（论文：1） |

### Order 阶段参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--order-checkpoint` | str | None | 预训练 OrderModule 路径 |
| `--freeze-order` | flag | False | 冻结 OrderModule |
| `--order-weight` | float | 1.0 | Order 损失权重 |
| `--order-num-layers` | int | 3 | Order Transformer 层数（论文：3） |

### Construct 阶段参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-construct` | flag | True | 启用 Construct 阶段 |
| `--construct-checkpoint` | str | None | 预训练 ConstructModule 路径 |
| `--freeze-construct` | flag | False | 冻结 ConstructModule |
| `--construct-weight` | float | 1.0 | Construct 损失权重 |
| `--construct-num-layers` | int | 3 | Construct Transformer 层数（论文：3） |

---

## 💾 保存/加载示例

### 保存模型

训练后会自动保存：

```python
# 自动保存（train_doc.py 中）
save_full_doc_pipeline(
    model,
    save_path=output_dir,
    save_separately=True,  # 分别保存各模块
)

# 输出文件：
# output_dir/
# ├── detect_module.pt       ← 完整 DetectModule
# ├── intra_head.pt          ← 独立的 IntraRegionHead
# ├── role_head.pt           ← 独立的 LogicalRoleHead
# ├── order_module.pt        ← OrderModule
# ├── construct_module.pt    ← ConstructModule
# └── full_doc_pipeline.pt   ← 完整模型
```

### 手动保存单个模块

```python
from examples.comp_hrdoc.models import (
    save_detect_module,
    save_intra_region_head,
)

# 保存完整 DetectModule
save_detect_module(
    detect_module=model.doc_pipeline.detect,
    save_path="./checkpoints/detect/",
    save_heads_separately=True,
)

# 只保存 IntraRegionHead
save_intra_region_head(
    detect_module=model.doc_pipeline.detect,
    save_path="./checkpoints/intra_head.pt",
)
```

### 加载模型

```python
from examples.comp_hrdoc.models.order import build_full_doc_pipeline

# 加载完整 Pipeline
model = build_full_doc_pipeline(
    hidden_size=768,
    num_roles=10,
    detect_checkpoint="./checkpoints/detect/detect_module.pt",
    order_checkpoint="./checkpoints/order/order_module.pt",
    construct_checkpoint="./checkpoints/construct/construct_module.pt",
    device="cuda",
)

# 也可以只加载部分模块
model = build_full_doc_pipeline(
    hidden_size=768,
    num_roles=10,
    detect_checkpoint="./checkpoints/detect/detect_module.pt",  # 只加载 Detect
    device="cuda",
)
```

---

## 🔧 API 使用示例

### 在自定义脚本中使用

```python
import torch
from examples.comp_hrdoc.models import (
    FullDOCPipeline,
    build_full_doc_pipeline,
    save_full_doc_pipeline,
)

# 1. 构建模型
model = FullDOCPipeline(
    hidden_size=768,
    num_roles=10,
    detect_num_heads=12,
    detect_num_layers=1,
    order_num_heads=12,
    order_num_layers=3,
    construct_num_heads=12,
    construct_num_layers=3,
    lambda_detect=1.0,
    lambda_order=1.0,
    lambda_construct=1.0,
)

# 2. 前向传播
outputs = model(
    # Detect inputs
    line_features=line_features,          # [batch, num_lines, 768]
    line_bboxes=line_bboxes,              # [batch, num_lines, 4]
    line_mask=line_mask,                  # [batch, num_lines]
    successor_labels=successor_labels,    # [batch, num_lines]
    role_labels=role_labels,              # [batch, num_lines]

    # Order inputs
    graphical_bboxes=graphical_bboxes,    # [batch, num_graphical, 4]
    graphical_mask=graphical_mask,        # [batch, num_graphical]
    region_order_labels=region_order_labels,
    relation_labels=relation_labels,

    # Construct inputs
    parent_labels=parent_labels,
    sibling_labels=sibling_labels,
)

# 3. 获取损失
detect_loss = outputs['detect_loss']
order_loss = outputs['order_loss']
construct_loss = outputs['construct_loss']
total_loss = outputs['total_loss']

# 4. 冻结/解冻模块
model.freeze_detect()    # 冻结 Detect
model.unfreeze_order()   # 解冻 Order
model.freeze_construct() # 冻结 Construct

# 5. 保存
save_full_doc_pipeline(model, "./checkpoints/")
```

---

## 📊 模型架构

```
FullDOCPipeline
├── doc_pipeline (DOCPipeline)
│   ├── detect (DetectModule)  ← 4.2 Detect Stage
│   │   ├── feature_proj
│   │   ├── intra_head (IntraRegionHead)
│   │   │   ├── transformer (1-layer, 12 heads)
│   │   │   ├── succ_head_proj (768 → 2048)
│   │   │   ├── succ_dep_proj (768 → 2048)
│   │   │   └── spatial_features (18-dim)
│   │   └── role_head (LogicalRoleHead)
│   │       └── classifier (768 → num_roles)
│   │
│   └── order (OrderModule)    ← 4.3 Order Stage
│       ├── region_builder
│       │   ├── attention_fusion
│       │   └── type_embedding
│       ├── transformer (3-layer, 12 heads)
│       ├── order_head (2048 nodes)
│       └── relation_head (BiLinear)
│
└── construct (ConstructModule) ← 4.4 Construct Stage
    ├── transformer (3-layer, 12 heads, RoPE)
    ├── parent_head
    └── sibling_head
```

---

## ⚙️ 训练配置建议

### 小数据集（< 1000 样本）

```bash
python examples/comp_hrdoc/scripts/train_doc.py \
    --env dev \
    --use-detect \
    --use-construct \
    --batch-size 1 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.1 \
    --gradient-accumulation-steps 4
```

### 中等数据集（1000-10000 样本）

```bash
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --use-construct \
    --batch-size 2 \
    --num-epochs 30 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --gradient-accumulation-steps 2
```

### 大数据集（> 10000 样本）

```bash
python examples/comp_hrdoc/scripts/train_doc.py \
    --env test \
    --use-detect \
    --use-construct \
    --batch-size 4 \
    --num-epochs 20 \
    --learning-rate 3e-5 \
    --warmup-ratio 0.05 \
    --gradient-accumulation-steps 1
```

---

## 🐛 故障排除

### 1. CUDA Out of Memory

```bash
# 减小 batch size 或使用 gradient accumulation
python examples/comp_hrdoc/scripts/train_doc.py \
    --batch-size 1 \
    --gradient-accumulation-steps 8

# 或冻结部分模块
python examples/comp_hrdoc/scripts/train_doc.py \
    --freeze-detect \
    --freeze-order
```

### 2. 加载 checkpoint 失败

```python
# 确保模型架构参数匹配
model = build_full_doc_pipeline(
    hidden_size=768,  # ← 必须与 checkpoint 一致
    num_roles=10,     # ← 必须与 checkpoint 一致
    detect_checkpoint="...",
)
```

### 3. 损失不收敛

```bash
# 调整损失权重
python examples/comp_hrdoc/scripts/train_doc.py \
    --detect-weight 0.5 \
    --order-weight 1.0 \
    --construct-weight 1.5

# 或使用更小的学习率
python examples/comp_hrdoc/scripts/train_doc.py \
    --learning-rate 1e-5
```

---

## 📚 相关文件

- `models/order.py` - FullDOCPipeline 实现
- `models/intra_region.py` - DetectModule 实现
- `scripts/train_doc.py` - 训练脚本
- `scripts/train_intra.py` - Detect 独立训练脚本

---

## 🎓 论文参考

完整实现遵循论文 "Detect-Order-Construct: A Unified Framework for Hierarchical Document Structure Analysis"：

- **Section 4.2**: Detect Stage (Intra-region + Logical Role)
- **Section 4.3**: Order Stage (Inter-region + Relation Type)
- **Section 4.4**: Construct Stage (Hierarchical Tree)
