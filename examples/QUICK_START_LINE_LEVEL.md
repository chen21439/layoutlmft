# Quick Start: Line-Level Stage 1 Training

## TL;DR

```bash
# 训练 Stage 1 (line-level, 与 JointModel 对齐)
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds
```

## 什么是 Line-Level 训练？

**原有方式** (Token-Level):
```
Token 分类 → 多数投票 → Line 标签
```

**新方式** (Line-Level, 与 JointModel 对齐):
```
Token → Mean Pooling → Line Features → Line 分类
```

## 为什么要用 Line-Level？

1. **与联合训练对齐**: 使用相同的 LinePooling 和 ClassificationHead
2. **训练更稳定**: 直接优化 line-level 目标
3. **评估更准确**: 无需投票聚合

## 使用方法

### 基本训练

```bash
# 训练 hrds 数据集
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds

# 训练 hrdh 数据集
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrdh

# 快速测试模式（小数据量，快速验证）
python examples/stage/scripts/train_stage1_line_level.py --env test --quick
```

### 恢复训练

```bash
# 自动检测最新 checkpoint 并恢复
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds

# 从头开始（覆盖现有 checkpoints）
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds --restart
```

### 迁移学习

```bash
# 从 hrds 训练好的模型初始化，训练 hrdh
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrdh \
    --init_from hrds
```

### 自定义参数

```bash
# 覆盖训练步数和 batch size
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --max_steps 500 \
    --batch_size 2

# 指定输出目录
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --output_dir /path/to/output

# Dry run（打印配置但不训练）
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --dry_run
```

## 输出位置

```
outputs/
└── experiments/
    └── [experiment_id]/
        └── stage1/
            ├── hrds_line_level/          # Line-level 模型 (hrds)
            ├── hrdh_line_level/          # Line-level 模型 (hrdh)
            ├── hrds/                     # Token-level 模型 (原有)
            └── hrdh/                     # Token-level 模型 (原有)
```

## 检查训练结果

```bash
# 查看训练日志
tail -f outputs/experiments/[exp_id]/stage1/hrds_line_level/trainer_state.json

# 查看评估指标
cat outputs/experiments/[exp_id]/stage1/hrds_line_level/eval_results.json

# TensorBoard (如果启用)
tensorboard --logdir outputs/experiments/[exp_id]/stage1/hrds_line_level/runs
```

## 与联合训练集成

```python
# train_joint.py
from examples.models.stage1_line_level_model import LayoutXLMForLineLevelClassification

# 加载 line-level checkpoint
backbone = LayoutXLMForTokenClassification.from_pretrained(
    "outputs/experiments/[exp_id]/stage1/hrds_line_level/checkpoint-XXX"
)

# 用于 JointModel
model = JointModel(
    stage1_model=backbone,
    stage3_model=parent_finder,
    stage4_model=relation_classifier,
    use_line_level_cls=True,  # 启用 line-level 分类
)
```

## 常见问题

### Q: 与原有 Stage 1 训练有什么区别？

**A**:
- **原有**: Token-level 分类 + 多数投票聚合
- **新的**: Mean pooling + Line-level 分类（与 JointModel 对齐）

### Q: 需要修改数据吗？

**A**: 不需要。数据加载器会自动提供 `line_ids` 和 `line_labels`。

### Q: 训练速度如何？

**A**: 与原有 token-level 速度相当，因为主要时间在 backbone 前向传播。

### Q: 可以用于推理吗？

**A**: 可以！输出是 line-level logits，可以直接用于预测。

### Q: 模型大小？

**A**: 与 LayoutXLM backbone 相同（约 560M），因为只是改变了分类头。

## 文件结构

```
examples/
├── models/
│   └── stage1_line_level_model.py          # Line-level 分类模型
├── stage/
│   ├── run_hrdoc_line_level.py             # 训练脚本
│   ├── data/
│   │   └── line_level_collator.py          # 数据整理器
│   └── scripts/
│       └── train_stage1_line_level.py      # 启动脚本 (推荐使用)
└── LINE_LEVEL_STAGE1_README.md             # 详细文档
```

## 进阶使用

### 直接调用训练脚本

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

### 加载训练好的模型

```python
from examples.models.stage1_line_level_model import LayoutXLMForLineLevelClassification

# 加载模型
model = LayoutXLMForLineLevelClassification.from_pretrained("checkpoint_path")

# 推理
outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    image=image,
    line_ids=line_ids,
)

predictions = outputs.logits.argmax(dim=-1)  # [batch, max_lines]
```

## 推荐工作流

```bash
# 1. 快速测试（验证代码和环境）
python examples/stage/scripts/train_stage1_line_level.py --env test --quick

# 2. 完整训练 hrds
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds

# 3. 迁移到 hrdh
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrdh --init_from hrds

# 4. 评估并选择最佳 checkpoint

# 5. 用于联合训练或推理
```

## 性能预期

基于联合训练的经验：

- **Accuracy**: Line-level accuracy 应与 token-level 相当或略高
- **Macro F1**: 提升 2-5%（因为不被高频类主导）
- **训练稳定性**: Loss 曲线更平滑
- **收敛速度**: 更快达到最优性能

## 获取帮助

- 详细文档: `examples/LINE_LEVEL_STAGE1_README.md`
- 实现总结: `IMPLEMENTATION_SUMMARY.md`
- JointModel 参考: `examples/models/joint_model.py`

---

**快速链接**:
- [详细文档](LINE_LEVEL_STAGE1_README.md)
- [实现总结](../IMPLEMENTATION_SUMMARY.md)
- [联合训练](train_joint.py)
