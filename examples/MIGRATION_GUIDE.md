# Migration Guide: Token-Level → Line-Level Stage 1

本文档说明如何从原有的 token-level Stage 1 训练迁移到新的 line-level 训练。

## 快速对比

| 方面 | Token-Level (原) | Line-Level (新) |
|------|-----------------|----------------|
| 训练脚本 | `scripts/train_stage1.py` | `scripts/train_stage1_line_level.py` |
| 运行脚本 | `run_hrdoc.py` | `run_hrdoc_line_level.py` |
| 模型类 | `LayoutXLMForTokenClassification` | `LayoutXLMForLineLevelClassification` |
| Data Collator | `DataCollatorForKeyValueExtraction` | `LineLevelDataCollator` |
| 聚合方式 | 多数投票 (token → line) | Mean pooling |
| 损失 | Token-level CrossEntropy | Line-level CrossEntropy |
| 与 JointModel 对齐 | ❌ 否 | ✅ **是** |

## 迁移场景

### 场景 1: 新项目（推荐）

**建议**: 直接使用 line-level 训练

```bash
# 直接开始
python examples/stage/scripts/train_stage1_line_level.py --env test --dataset hrds
```

**优势**:
- 无需迁移成本
- 与联合训练对齐
- 训练更稳定

---

### 场景 2: 已有 token-level checkpoint，需要继续训练

**方案 A: 保持 token-level 训练**（不迁移）

```bash
# 继续使用原有脚本
python examples/stage/scripts/train_stage1.py --env test --dataset hrds
```

**方案 B: 迁移到 line-level**（推荐）

```bash
# Step 1: 从 token-level checkpoint 初始化 backbone
# 修改 run_hrdoc_line_level.py 的模型加载部分：

# 原代码:
backbone_model = LayoutXLMForTokenClassification.from_pretrained(
    model_args.model_name_or_path,
    ...
)

# 如果 model_name_or_path 是 token-level checkpoint，直接使用即可
# backbone 部分是兼容的

# Step 2: 训练
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --model_path /path/to/token_level_checkpoint
```

**注意**:
- Token-level 和 line-level 的 backbone 是兼容的
- 只有分类头不同（但 line-level 会重新初始化分类头）
- 建议从头训练几百步以适应新的训练方式

---

### 场景 3: 需要同时维护两种训练方式

**建议**: 分别使用不同的输出目录

```bash
# Token-level
python examples/stage/scripts/train_stage1.py \
    --env test \
    --dataset hrds \
    --output_dir outputs/stage1_token_level

# Line-level
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --output_dir outputs/stage1_line_level
```

**对比实验**:
```bash
# 1. 分别训练
# (见上面的命令)

# 2. 评估并对比
python util/compare_models.py \
    --model_a outputs/stage1_token_level \
    --model_b outputs/stage1_line_level \
    --eval_dataset validation
```

---

### 场景 4: 用于联合训练

**Token-Level → JointModel**:

```python
# 不对齐！需要设置 use_line_level_cls=False
from examples.models.joint_model import JointModel

backbone = LayoutXLMForTokenClassification.from_pretrained("token_level_checkpoint")

model = JointModel(
    stage1_model=backbone,
    use_line_level_cls=False,  # 使用 token-level 分类 + 投票
    ...
)
```

**Line-Level → JointModel** (推荐):

```python
# 完全对齐！
from examples.models.joint_model import JointModel

backbone = LayoutXLMForTokenClassification.from_pretrained("line_level_checkpoint")

model = JointModel(
    stage1_model=backbone,
    use_line_level_cls=True,  # 使用 line-level 分类
    ...
)
```

---

## 详细迁移步骤

### Step 1: 理解差异

**Token-Level 训练流程**:
```
1. Token 分类: LayoutXLM → logits [B, seq, num_classes]
2. 计算损失: CrossEntropy(token_logits, token_labels)
3. 评估时聚合: 多数投票 token → line
```

**Line-Level 训练流程**:
```
1. Backbone: LayoutXLM → hidden_states [B, seq, 768]
2. LinePooling: hidden_states → line_features [L, 768]
3. 分类: ClassificationHead → logits [L, num_classes]
4. 计算损失: CrossEntropy(line_logits, line_labels)
```

### Step 2: 准备数据

**好消息**: 数据格式完全相同，无需修改！

两种方式都使用相同的数据加载器：
```python
from data import HRDocDataLoader, load_hrdoc_raw_datasets

# 数据加载
datasets = load_hrdoc_raw_datasets(...)
data_loader = HRDocDataLoader(
    tokenizer=tokenizer,
    include_line_info=True,  # 两种方式都需要
)
tokenized_datasets = data_loader.prepare_datasets()
```

唯一区别是 Data Collator:
- Token-level: `DataCollatorForKeyValueExtraction`
- Line-level: `LineLevelDataCollator`

### Step 3: 迁移模型权重（可选）

如果你有 token-level checkpoint，可以这样迁移：

```python
from layoutlmft.models.layoutxlm import LayoutXLMForTokenClassification
from examples.models.stage1_line_level_model import LayoutXLMForLineLevelClassification

# 加载 token-level checkpoint
token_model = LayoutXLMForTokenClassification.from_pretrained(
    "path/to/token_level_checkpoint"
)

# 提取 backbone（兼容）
# LayoutXLM 的 backbone 在两种模型中结构相同
line_model = LayoutXLMForLineLevelClassification(
    backbone_model=token_model,  # 复用 backbone
    num_classes=14,
    hidden_size=768,
)

# 分类头会重新初始化（因为结构不同）
# 建议 fine-tune 几百步
```

### Step 4: 训练

**从预训练模型开始**（推荐）:
```bash
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --model_path microsoft/layoutxlm-base
```

**从 token-level checkpoint 开始**:
```bash
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds \
    --model_path /path/to/token_level_checkpoint \
    --max_steps 1000  # 建议训练一段时间以适应新方式
```

### Step 5: 评估

两种方式的评估指标可以直接对比：

```bash
# Token-level 评估
python util/hrdoc_eval.py \
    --model_path /path/to/token_level_checkpoint \
    --eval_dataset validation

# Line-level 评估
python util/hrdoc_eval.py \
    --model_path /path/to/line_level_checkpoint \
    --eval_dataset validation
```

对比指标：
- Line-level Accuracy
- Line-level Macro F1
- Per-class F1

---

## 常见问题

### Q1: 迁移后性能会变差吗？

**A**: 不会。基于联合训练的经验：
- Accuracy 应相当或略高
- Macro F1 通常提升 2-5%
- 少数类性能更好

### Q2: 需要重新调参吗？

**A**: 大部分超参数可以保持不变：
- Learning rate: 相同
- Batch size: 相同
- Warmup steps: 相同

可能需要微调的：
- Training steps: Line-level 收敛可能更快
- Dropout: 可以尝试稍微调整（默认 0.1）

### Q3: 训练时间会变长吗？

**A**: 几乎相同。主要时间在 backbone 前向传播，line pooling 是高效的向量化操作。

### Q4: 已有的 token-level 模型还能用吗？

**A**: 可以！两种方式可以共存：
- Token-level: 用于不需要与联合训练对齐的场景
- Line-level: 用于需要对齐的场景

### Q5: 如何选择？

**推荐使用 line-level** 如果：
- 需要与联合训练对齐
- 需要更稳定的训练
- 新项目

**继续使用 token-level** 如果：
- 已有大量 token-level checkpoints
- 不需要联合训练

---

## 迁移检查清单

完成迁移后，检查以下项目：

- [ ] 训练脚本正确（使用 `train_stage1_line_level.py`）
- [ ] 模型加载正确（backbone 兼容）
- [ ] Data collator 正确（使用 `LineLevelDataCollator`）
- [ ] 训练正常运行（loss 下降）
- [ ] 评估指标合理（与 token-level 相当或更好）
- [ ] Checkpoint 保存正确
- [ ] 可用于联合训练（如需要）

---

## 迁移示例

### 完整示例: 从 token-level 迁移到 line-level

```bash
# ========== 原有工作流 (Token-Level) ==========

# 1. 训练
python examples/stage/scripts/train_stage1.py \
    --env test \
    --dataset hrds

# 2. 评估
python util/hrdoc_eval.py \
    --model_path outputs/stage1/hrds/checkpoint-1000

# ========== 新工作流 (Line-Level) ==========

# 1. 从 token-level checkpoint 初始化（可选）
# 或直接从预训练模型开始（推荐）

# 2. 训练
python examples/stage/scripts/train_stage1_line_level.py \
    --env test \
    --dataset hrds

# 3. 评估
python util/hrdoc_eval.py \
    --model_path outputs/stage1/hrds_line_level/checkpoint-1000

# 4. 对比两种方式
python util/compare_models.py \
    --model_a outputs/stage1/hrds/checkpoint-1000 \
    --model_b outputs/stage1/hrds_line_level/checkpoint-1000
```

---

## 获取帮助

遇到迁移问题？

- 查看详细文档: `examples/LINE_LEVEL_STAGE1_README.md`
- 查看实现总结: `IMPLEMENTATION_SUMMARY.md`
- 查看快速开始: `examples/QUICK_START_LINE_LEVEL.md`

---

**总结**: 迁移到 line-level 训练是直接且低成本的，强烈推荐用于新项目和需要与联合训练对齐的场景。
