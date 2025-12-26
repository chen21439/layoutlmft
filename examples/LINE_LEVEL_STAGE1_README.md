# Stage 1 Line-Level Classification

## 概述

本文档说明如何使用 **line-level 分类**（mean pooling）进行 Stage 1 训练，与联合训练中 Stage 1 的逻辑完全对齐。

## 背景

### 原有方式（Token-Level）

- 模型: `LayoutXLMForTokenClassification`
- 训练脚本: `examples/stage/run_hrdoc.py`
- 机制: Token-level 分类 + 多数投票聚合到 line-level

### 新方式（Line-Level，与 JointModel 对齐）

- 模型: `LayoutXLMForLineLevelClassification`
- 训练脚本: `examples/stage/run_hrdoc_line_level.py`
- 机制: Token → Line mean pooling → Line-level 分类

## 架构对齐

### JointModel 的 Stage 1 逻辑

```python
# joint_model.py (use_line_level_cls=True)
hidden_states = backbone(input_ids, bbox, image)  # [B, seq, 768]
line_features, mask = line_pooling(hidden_states, line_ids)  # [L, 768]
logits = cls_head(line_features)  # [L, num_classes]
loss = F.cross_entropy(logits, line_labels)
```

### 新的 Stage 1 单独训练

```python
# stage1_line_level_model.py
hidden_states = backbone(input_ids, bbox, image)  # [B, seq, 768]
line_features, mask = line_pooling(hidden_states, line_ids)  # [L, 768]
logits = cls_head(line_features)  # [L, num_classes]
loss = F.cross_entropy(logits, line_labels)
```

**完全一致！**

## 文件结构

```
examples/
├── models/
│   └── stage1_line_level_model.py          # Line-level 分类模型
├── stage/
│   ├── run_hrdoc_line_level.py             # Line-level 训练脚本
│   ├── data/
│   │   └── line_level_collator.py          # Line-level 数据整理器
│   └── scripts/
│       └── train_stage1_line_level.py      # 便捷启动脚本
```

## 使用方法

### 1. 基本训练

```bash
# 使用默认配置训练
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds

# 快速测试模式
python examples/stage/scripts/train_stage1_line_level.py --env test --quick

# 指定数据集
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrdh
```

### 2. 恢复训练

```bash
# 自动检测最新 checkpoint 并恢复
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds

# 从头开始（覆盖现有 checkpoints）
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds --restart
```

### 3. 迁移学习

```bash
# 从 hrds 训练好的模型初始化，训练 hrdh
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrdh \
    --init_from hrds
```

### 4. 直接使用训练脚本

```bash
python examples/stage/run_hrdoc_line_level.py \
    --model_name_or_path /path/to/layoutxlm-base \
    --output_dir outputs/stage1_line_level \
    --do_train \
    --do_eval \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 100
```

## 核心组件

### 1. LayoutXLMForLineLevelClassification

**位置**: `examples/models/stage1_line_level_model.py`

**功能**:
- Backbone: 使用 LayoutXLM 获取 token-level hidden states
- LinePooling: 聚合到 line-level features（mean pooling）
- LineClassificationHead: 分类每一行
- Loss: Line-level cross entropy

**与 JointModel 的共享模块**:
- `LinePooling` (from `examples/stage/models/modules/line_pooling.py`)
- `LineClassificationHead` (from `examples/stage/models/heads/classification_head.py`)

### 2. LineLevelDataCollator

**位置**: `examples/stage/data/line_level_collator.py`

**功能**:
- 提供 `input_ids`, `bbox`, `attention_mask`, `image`
- 提供 `labels` (token-level，用于提取 line_labels)
- 提供 `line_ids` (token → line 映射)
- 提供 `line_labels` (line-level 标签，从 token labels 提取)

**与 HRDocJointDataCollator 的区别**:
- 不需要 `line_parent_ids` 和 `line_relations`（Stage 3/4 专用）
- 预先提取 `line_labels`，避免在每次 forward 中重复计算

### 3. run_hrdoc_line_level.py

**位置**: `examples/stage/run_hrdoc_line_level.py`

**功能**:
- 加载数据集和 tokenizer
- 创建 `LayoutXLMForLineLevelClassification` 模型
- 使用 `LineLevelDataCollator` 整理数据
- 训练、评估、保存模型

**与 run_hrdoc.py 的区别**:
- 使用 line-level 模型而非 token-level 模型
- 使用 line-level 数据整理器
- 评估指标直接在 line-level 计算（无需投票聚合）

## 优势

### 1. 与联合训练完全对齐

- **相同的聚合方式**: Mean pooling（而非多数投票）
- **相同的分类头**: `LineClassificationHead`
- **相同的损失函数**: Line-level cross entropy

**好处**: Stage 1 单独训练的模型可以无缝迁移到联合训练，无需重新适应。

### 2. 训练更稳定

- **直接优化 line-level 目标**: 不经过 token-level 中间步骤
- **损失更平滑**: Line-level 的 loss 比 token-level 更稳定
- **收敛更快**: 直接优化最终目标

### 3. 评估更准确

- **无需投票聚合**: 直接在 line-level 计算指标
- **指标一致性**: 训练和评估使用相同的粒度
- **更好的类别平衡**: Line-level 不会被高频 token 主导

## 与原有方式的对比

| 维度 | Token-Level (原) | Line-Level (新) |
|------|-----------------|----------------|
| 模型 | LayoutXLMForTokenClassification | LayoutXLMForLineLevelClassification |
| 聚合 | 多数投票 | Mean pooling |
| 损失 | Token-level CE | Line-level CE |
| 评估 | 需要投票聚合 | 直接计算 |
| 与 Joint 对齐 | 不对齐 | **完全对齐** |
| 训练稳定性 | 中等 | **更稳定** |
| 收敛速度 | 中等 | **更快** |

## 常见问题

### Q1: 是否需要重新预处理数据？

**A**: 不需要。数据加载器会自动提供 `line_ids` 和 `line_labels`。

### Q2: 与原有 Stage 1 模型兼容吗？

**A**: 不直接兼容。但可以通过以下方式迁移：

```python
# 方式 1: 从原有 Stage 1 checkpoint 初始化 backbone
backbone = LayoutXLMForTokenClassification.from_pretrained("old_checkpoint")
model = LayoutXLMForLineLevelClassification(backbone_model=backbone)

# 方式 2: 直接从预训练模型开始训练（推荐）
python scripts/train_stage1_line_level.py --env test
```

### Q3: 训练速度如何？

**A**: 与原有 token-level 训练速度相当，因为：
- Backbone 前向传播时间相同
- Line pooling 是高效的向量化操作
- 分类头计算量更小（line << token）

### Q4: 可以用于推理吗？

**A**: 可以！模型输出 line-level logits，可以直接用于预测：

```python
model = LayoutXLMForLineLevelClassification.from_pretrained("checkpoint")
outputs = model(input_ids=..., bbox=..., line_ids=...)
predictions = outputs.logits.argmax(dim=-1)  # [batch, max_lines]
```

### Q5: 如何与 Stage 3/4 联合训练？

**A**: 使用 `JointModel` 时，设置 `use_line_level_cls=True`：

```python
from examples.models.joint_model import JointModel

model = JointModel(
    stage1_model=stage1_backbone,
    stage3_model=parent_finder,
    stage4_model=relation_classifier,
    use_line_level_cls=True,  # 使用 line-level 分类
)
```

## 实现细节

### Line Pooling

```python
# 从 examples/stage/models/modules/line_pooling.py
class LinePooling(nn.Module):
    def forward(self, hidden_states, line_ids):
        # 1. 展平: [batch, seq, H] → [N, H]
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        flat_line_ids = line_ids.reshape(-1)

        # 2. 过滤: 只保留 line_id >= 0 的 token
        valid_mask = flat_line_ids >= 0
        valid_hidden = flat_hidden[valid_mask]
        valid_line_ids = flat_line_ids[valid_mask]

        # 3. 聚合: 使用 scatter_add 按 line_id 求和
        line_features = torch.zeros(num_lines, hidden_dim)
        line_features.scatter_add_(0, line_indices, valid_hidden)

        # 4. 平均: 除以每行的 token 数
        line_features = line_features / line_counts

        return line_features, line_mask
```

### Line Classification Head

```python
# 从 examples/stage/models/heads/classification_head.py
class LineClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, line_features):
        x = self.dropout(line_features)
        logits = self.classifier(x)
        return logits
```

### Loss 计算

```python
# Line-level cross entropy
for b in range(batch_size):
    logits = cls_head(line_features[b])  # [num_lines, num_classes]
    labels = line_labels[b]  # [num_lines]

    # 过滤 label=-100
    valid_mask = labels != -100
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]

    loss = F.cross_entropy(valid_logits, valid_labels)
```

## 性能指标

### 预期改进

基于联合训练的经验，line-level 分类相比 token-level 应该有以下改进：

1. **训练稳定性**: Loss 曲线更平滑
2. **收敛速度**: 更快达到最优性能
3. **少数类性能**: 更好的 tail class F1
4. **Macro F1**: 提升 2-5%（因为不被高频类主导）

### 实验建议

建议进行以下对比实验：

```bash
# Token-level baseline
python scripts/train_stage1.py --env test --dataset hrds

# Line-level (新方法)
python scripts/train_stage1_line_level.py --env test --dataset hrds
```

对比指标：
- Line-level Accuracy
- Line-level Macro F1
- Per-class F1（特别是少数类）
- 训练时间
- 内存占用

## 总结

**Line-level Stage 1 训练**是联合训练中 Stage 1 逻辑的独立版本，具有以下优势：

1. **完全对齐**: 与 JointModel 使用相同的模块和逻辑
2. **更稳定**: 直接优化 line-level 目标
3. **更准确**: 评估指标与训练目标一致
4. **易迁移**: 可无缝用于联合训练

**推荐使用场景**:
- 需要与联合训练对齐的 Stage 1 预训练
- 需要更稳定的训练过程
- 需要更准确的 line-level 评估
- 作为联合训练的 warm start

**不推荐场景**:
- 已有大量 token-level 训练的 checkpoints（迁移成本高）
- 对训练速度要求极高（虽然差异不大）
