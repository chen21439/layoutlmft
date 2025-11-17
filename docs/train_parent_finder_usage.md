# 任务2：父节点查找训练说明

## 两种模式

### 1. Simple模式（简化版）- 本地测试
- **适用场景**: 本地4GB显存测试
- **实现方式**: 样本对分类（参考任务3）
- **内存占用**: ~8MB per batch
- **预期性能**: 接近任务3的0.9+ F1

```bash
# 本地测试（1个chunk）
python examples/train_parent_finder.py \
    --mode simple \
    --batch_size 128 \
    --num_epochs 20 \
    --max_chunks 1

# 本地全量训练
python examples/train_parent_finder.py \
    --mode simple \
    --batch_size 128 \
    --num_epochs 20
```

### 2. Full模式（完整版）- 云服务器
- **适用场景**: 云服务器24GB显存
- **实现方式**: GRU + 注意力（论文方法）
- **内存占用**: ~10GB per batch（batch_size=2）
- **预期性能**: 论文级别性能

```bash
# 云服务器训练（不使用soft-mask）
python examples/train_parent_finder.py \
    --mode full \
    --batch_size 2 \
    --num_epochs 10

# 云服务器训练（使用soft-mask）
python examples/train_parent_finder.py \
    --mode full \
    --batch_size 2 \
    --num_epochs 10 \
    --use_soft_mask

# 云服务器+梯度检查点（节省显存）
python examples/train_parent_finder.py \
    --mode full \
    --batch_size 4 \
    --num_epochs 10 \
    --gradient_checkpointing
```

## 参数说明

| 参数 | 说明 | Simple默认 | Full默认 |
|------|------|-----------|---------|
| `--mode` | simple或full | simple | - |
| `--batch_size` | 批大小 | 128 | 2 |
| `--learning_rate` | 学习率 | 1e-3 | 1e-4 |
| `--num_epochs` | 训练轮数 | 20 | 20 |
| `--max_chunks` | chunk数量(-1=全部) | -1 | -1 |
| `--use_soft_mask` | 使用soft-mask | - | False |
| `--gradient_checkpointing` | 梯度检查点 | - | False |

## 内存对比

### Simple模式
```
每个样本：(child, 20个候选parents)
Batch: [128, 20个候选]
内存: ~8 MB
```

### Full模式
```
每个样本：一个完整页面（可能400行）
Batch: [2, 400行, 400行] 注意力矩阵
内存: ~10 GB
```

## 输出

训练完成后，模型保存在：
```
/mnt/e/models/train_data/layoutlmft/parent_finder_simple/  # simple模式
/mnt/e/models/train_data/layoutlmft/parent_finder_full/    # full模式
```

包含：
- `best_model.pt` - 最佳模型checkpoint
- 训练日志

## 数据要求

使用与任务3相同的数据：
```
/mnt/e/models/train_data/layoutlmft/line_features/
├── train_line_features_chunk_0000.pkl  (1000页)
├── train_line_features_chunk_0001.pkl  (1000页)
├── ...
└── validation_line_features_chunk_0000.pkl
```

每个页面包含：
- `line_features`: [1, max_lines, 768]
- `line_mask`: [1, max_lines]
- `line_parent_ids`: List[int]  # Ground Truth
- `line_relations`: List[str]
- `line_bboxes`: np.array

## 推荐流程

### 本地开发
```bash
# 1. 快速验证（1个chunk）
python examples/train_parent_finder.py --mode simple --max_chunks 1 --num_epochs 5

# 2. 完整训练
python examples/train_parent_finder.py --mode simple --num_epochs 20

# 3. 评估结果
```

### 云服务器训练
```bash
# 1. 先用simple模式验证数据和代码
python examples/train_parent_finder.py --mode simple --max_chunks 1

# 2. Full模式训练
python examples/train_parent_finder.py --mode full --batch_size 2 --num_epochs 10

# 3. 如果显存充足，可以加大batch_size或启用soft-mask
python examples/train_parent_finder.py --mode full --batch_size 4 --use_soft_mask
```

## 性能预期

### Simple模式
- **训练速度**: 快（128样本/batch）
- **准确率**: ~0.85-0.90（参考任务3的0.9+）
- **优点**: 内存友好，训练快
- **缺点**: 无全局上下文

### Full模式
- **训练速度**: 慢（2页/batch）
- **准确率**: ~0.90+（论文方法）
- **优点**: 全局上下文，序列建模
- **缺点**: 内存消耗大

## Troubleshooting

### OOM错误
```bash
# 方案1：减小batch_size
--batch_size 1

# 方案2：启用梯度检查点
--gradient_checkpointing

# 方案3：减少chunk数量
--max_chunks 5

# 方案4：使用simple模式
--mode simple
```

### 训练太慢
```bash
# 使用simple模式
--mode simple

# 增大batch_size（如果显存允许）
--batch_size 256
```

## 与任务3的关系

任务2和任务3可以协同工作：

```python
# 推理流程
# 1. 任务1：语义分类（run_hrdoc.py）
semantic_labels = task1_model.predict(document)

# 2. 任务2：父节点查找（本脚本）
parent_pairs = task2_model.predict(document)
# 输出：[(parent_idx, child_idx), ...]

# 3. 任务3：关系分类（train_multiclass_relation.py）
relation_types = task3_model.predict(parent_pairs)
# 输出：["contain", "connect", ...]

# 4. 构建文档树
tree = build_tree(semantic_labels, parent_pairs, relation_types)
```

## 论文对应关系

| 组件 | Simple模式 | Full模式 |
|------|-----------|---------|
| **SubTask 2定义** | ✓ | ✓✓ |
| **GRU Decoder** | ✗ | ✓ |
| **Attention** | ✗ | ✓ |
| **Soft-mask (M_cp)** | ✗ | ✓ (可选) |
| **内存友好** | ✓✓ | ✗ |
| **符合论文** | 部分 | 完全 |
