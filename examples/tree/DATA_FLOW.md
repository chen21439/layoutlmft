# 阶段4数据流详解

## 完整的数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: SubTask 1 (run_hrdoc.py)                             │
│ 输入: 原始文档图像 + OCR                                     │
│ 模型: LayoutLMv2                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ 生成并缓存
┌─────────────────────────────────────────────────────────────┐
│ line_features.pkl (缓存文件)                                │
│ ├── line_features: [N, 768] tensor                         │
│ ├── line_mask: [N] tensor                                  │
│ ├── line_labels: [N] int list  ← SubTask 1的预测结果        │
│ ├── line_bboxes: [N, 4] float list                         │
│ ├── line_parent_ids: [N] int list (GT, 仅用于评估)         │
│ └── line_relations: [N] str list (GT, 仅用于评估)          │
└─────────────────────────────────────────────────────────────┘
                            ↓ 被阶段2、3、4使用
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: SubTask 2 (train_parent_finder.py)                  │
│ 输入: line_features + line_bboxes                           │
│ 输出: parent_finder/best_model.pt                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 阶段3: SubTask 3 (train_multiclass_relation.py)            │
│ 输入: line_features + line_bboxes + GT parent_ids          │
│ 输出: multiclass_relation/best_model.pt                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 阶段4: Overall Task (inference_build_tree.py)              │
│ 输入:                                                        │
│   1. line_features.pkl (从阶段1)                           │
│   2. parent_finder/best_model.pt (从阶段2)                 │
│   3. multiclass_relation/best_model.pt (从阶段3)           │
│                                                              │
│ 处理流程:                                                    │
│   ├─ 读取 line_features, line_labels, line_bboxes          │
│   ├─ 用 SubTask2模型 → 预测 parent_indices                 │
│   ├─ 用 SubTask3模型 → 预测 relation_types                 │
│   └─ DocumentTree.from_predictions() → 构建树              │
│                                                              │
│ 输出:                                                        │
│   └─ tree_*.json / tree_*.md / tree_*_ascii.txt            │
└─────────────────────────────────────────────────────────────┘
```

## 关键数据文件位置

### 输入数据（阶段1生成）

```bash
/mnt/e/models/train_data/layoutlmft/line_features/
├── train_line_features_chunk_0000.pkl      # 训练集特征
├── validation_line_features_chunk_0000.pkl # 验证集特征
└── test_line_features_chunk_0000.pkl       # 测试集特征（如果有）
```

### 训练好的模型（阶段2、3生成）

```bash
/mnt/e/models/train_data/layoutlmft/
├── parent_finder_simple/
│   └── best_model.pt                       # 阶段2模型
└── multiclass_relation/
    └── best_model.pt                       # 阶段3模型
```

## 阶段4如何使用数据（详细代码）

### 1. 加载缓存的特征（来自阶段1）

```python
# 在 inference_build_tree.py 中
import pickle

# 读取validation数据
with open("validation_line_features_chunk_0000.pkl", "rb") as f:
    page_features = pickle.load(f)

# 每个page_data包含：
for page_data in page_features:
    line_features = page_data["line_features"]    # [N, 768] - 来自LayoutLMv2
    line_labels = page_data["line_labels"]        # [N] - SubTask 1预测的语义类别
    line_bboxes = page_data["line_bboxes"]        # [N, 4] - 边界框
    line_mask = page_data["line_mask"]            # [N] - 有效性mask
```

### 2. 使用SubTask 2模型预测父节点

```python
# 加载训练好的模型（来自阶段2）
subtask2_model = SimpleParentFinder(...)
checkpoint = torch.load("parent_finder_simple/best_model.pt")
subtask2_model.load_state_dict(checkpoint["model_state_dict"])

# 对每个child预测父节点
parent_indices = []
for child_idx in range(num_lines):
    # 候选父节点：0 到 child_idx-1
    for parent_idx in range(child_idx):
        # 使用line_features和line_bboxes
        score = subtask2_model(
            child_feat=line_features[child_idx],
            parent_feat=line_features[parent_idx],
            geom_feat=compute_geometry_features(
                line_bboxes[parent_idx],
                line_bboxes[child_idx]
            )
        )

    # 选择得分最高的
    best_parent = argmax(scores)
    parent_indices.append(best_parent)
```

### 3. 使用SubTask 3模型预测关系

```python
# 加载训练好的模型（来自阶段3）
subtask3_model = MultiClassRelationClassifier(...)
checkpoint = torch.load("multiclass_relation/best_model.pt")
subtask3_model.load_state_dict(checkpoint["model_state_dict"])

# 对每个(parent, child)对预测关系
relation_types = []
for child_idx in range(num_lines):
    parent_idx = parent_indices[child_idx]  # 使用阶段2预测的parent

    # 预测关系类型
    logits = subtask3_model(
        parent_feat=line_features[parent_idx],
        child_feat=line_features[child_idx],
        geom_feat=compute_geometry_features(
            line_bboxes[parent_idx],
            line_bboxes[child_idx]
        )
    )

    relation = argmax(logits)  # 0=none, 1=connect, 2=contain, 3=equality
    relation_types.append(relation)
```

### 4. 构建文档树

```python
# 组合三个子任务的输出
tree = DocumentTree.from_predictions(
    line_labels=line_labels,           # 来自阶段1（缓存）
    parent_indices=parent_indices,     # 来自阶段2（预测）
    relation_types=relation_types,     # 来自阶段3（预测）
    line_bboxes=line_bboxes,          # 来自阶段1（缓存）
)

# 输出
tree.to_json("tree_0000.json")
tree.to_markdown("tree_0000.md")
```

## 快速验证方案

### 方案1: 验证单个样本（最快，10秒）

```bash
cd /root/code/layoutlmft/examples/tree

# 只处理1个validation样本
python inference_build_tree.py \
    --subtask2_model /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt \
    --subtask3_model /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt \
    --features_dir /mnt/e/models/train_data/layoutlmft/line_features \
    --split validation \
    --max_samples 1 \
    --max_chunks 1 \
    --output_dir ./quick_test \
    --save_json \
    --save_markdown \
    --save_ascii

# 查看结果
cat ./quick_test/tree_0000.md
cat ./quick_test/tree_0000_ascii.txt
```

### 方案2: 验证10个样本（1分钟）

```bash
python inference_build_tree.py \
    --split validation \
    --max_samples 10 \
    --max_chunks 1 \
    --output_dir ./validation_10 \
    --save_json \
    --save_markdown

# 查看统计
cat ./validation_10/summary.json
```

### 方案3: 完整validation集（5-10分钟）

```bash
python inference_build_tree.py \
    --split validation \
    --max_samples -1 \
    --max_chunks -1 \
    --output_dir ./validation_full \
    --save_json

# 生成约8000个树文件
ls ./validation_full/tree_*.json | wc -l
```

### 方案4: 使用test集（如果有）

```bash
python inference_build_tree.py \
    --split test \
    --max_samples 10 \
    --max_chunks 1 \
    --output_dir ./test_10 \
    --save_json \
    --save_markdown
```

## 验证结果分析

### 检查单个树

```bash
# 1. JSON格式（详细信息）
cat ./quick_test/tree_0000.json | python -m json.tool | head -50

# 2. Markdown格式（人类可读）
cat ./quick_test/tree_0000.md

# 3. ASCII树（可视化）
cat ./quick_test/tree_0000_ascii.txt
```

### Python分析

```python
import json

# 读取树
with open("./quick_test/tree_0000.json") as f:
    tree = json.load(f)

# 查看统计
stats = tree["statistics"]
print(f"节点数: {stats['total_nodes']}")
print(f"最大深度: {stats['max_depth']}")
print(f"标签分布: {stats['label_distribution']}")
print(f"关系分布: {stats['relation_distribution']}")

# 检查是否合理
if stats['max_depth'] > 10:
    print("⚠️ 深度过大，可能有预测错误")

if stats['relation_distribution'].get('none', 0) > stats['total_nodes'] * 0.3:
    print("⚠️ none关系过多，SubTask预测可能不准确")
```

## 性能评估（带GT对比）

```bash
# End-to-End评估（包含预测 vs GT对比）
python evaluate_end_to_end.py

# 这会输出：
# - SubTask 2单独性能（父节点准确率）
# - SubTask 3单独性能（关系分类F1）
# - End-to-End性能（父节点+关系都正确的准确率）
```

## 数据依赖总结

| 阶段 | 输入 | 输出 | 被谁使用 |
|------|------|------|----------|
| **阶段1** | 原始文档 | line_features.pkl | 阶段2,3,4 |
| **阶段2** | line_features.pkl | parent_finder.pt | 阶段4 |
| **阶段3** | line_features.pkl | relation_classifier.pt | 阶段4 |
| **阶段4** | 以上所有 | DocumentTree | 最终应用 |

## 注意事项

### ✅ 阶段4可以做的

1. **使用任意split的数据**（train/validation/test）
2. **只需要line_features.pkl**（不需要原始图像）
3. **可以快速测试**（max_samples=1即可）
4. **支持批量处理**（处理整个数据集）

### ⚠️ 阶段4的限制

1. **依赖阶段1的缓存**（必须先运行extract_line_features.py）
2. **依赖阶段2、3的模型**（必须先训练完成）
3. **使用预测的line_labels**（如果缓存中没有，需要重新推理）
4. **错误会累积**（阶段2错误会影响阶段3）

## 完整示例脚本

```bash
#!/bin/bash
# quick_validate.sh - 快速验证阶段4

echo "=== 阶段4快速验证 ==="

# 1. 检查输入文件
echo "1. 检查输入数据..."
ls -lh /mnt/e/models/train_data/layoutlmft/line_features/validation_*.pkl
ls -lh /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt
ls -lh /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt

# 2. 快速推理（1个样本）
echo -e "\n2. 快速推理（1个样本）..."
python inference_build_tree.py \
    --max_samples 1 \
    --save_json --save_markdown --save_ascii \
    --output_dir ./quick_test

# 3. 查看结果
echo -e "\n3. 查看结果..."
echo "--- Markdown ---"
cat ./quick_test/tree_0000.md

echo -e "\n--- 统计 ---"
cat ./quick_test/summary.json | python -m json.tool

echo -e "\n✅ 验证完成！"
```

保存为`quick_validate.sh`并运行：
```bash
chmod +x quick_validate.sh
./quick_validate.sh
```
