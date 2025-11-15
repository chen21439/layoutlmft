# 快速开始指南

## 5分钟快速上手

### 步骤1: 测试树构建（30秒）

```bash
cd /root/code/layoutlmft/examples/tree
python document_tree.py
```

**预期输出**:
```
=== ASCII Tree ===
└── Title: Document Title...
    ├── Section: Section 1...
    │   └── Para-Line: This is a paragraph....
    ...

✓ Tree saved to demo_tree.json
```

### 步骤2: 运行示例程序（2分钟）

```bash
python example_usage.py
```

**你会看到**:
- 5个不同的使用示例
- 树的遍历、查询、导出
- 各种可视化格式

### 步骤3: 完整推理（2分钟）

```bash
# 快速测试（只处理2个样本）
python inference_build_tree.py \
    --max_samples 2 \
    --max_chunks 1 \
    --save_json \
    --save_markdown \
    --save_ascii \
    --output_dir ./test_outputs
```

**检查输出**:
```bash
ls -lh ./test_outputs/
# tree_0000.json
# tree_0000.md
# tree_0000_ascii.txt
# summary.json
```

---

## 10分钟深入了解

### 1. 理解数据流 (3分钟)

**输入数据**（已缓存在 `line_features.pkl`）:
```python
{
    "line_features": Tensor[num_lines, 768],  # LayoutLMv2特征
    "line_labels": List[int],                 # 语义类别
    "line_bboxes": List[[x1,y1,x2,y2]],      # 边界框
}
```

**三个子任务**:
1. SubTask 1: `line_labels` (已完成，使用缓存)
2. SubTask 2: `parent_indices` ← **本次实现的核心**
3. SubTask 3: `relation_types` ← **本次实现的核心**

**输出**: `DocumentTree` 对象

### 2. 使用Python API (5分钟)

**最简示例**:
```python
from document_tree import DocumentTree

# 模拟三个子任务的输出
line_labels = [0, 1, 3, 3]         # Title, Section, Para, Para
parent_indices = [-1, 0, 1, 2]     # ROOT, Title, Section, Para
relation_types = [0, 2, 2, 1]      # none, contain, contain, connect

# 构建树
tree = DocumentTree.from_predictions(
    line_labels=line_labels,
    parent_indices=parent_indices,
    relation_types=relation_types,
)

# 输出
print(tree.visualize_ascii())
tree.to_json("my_tree.json")
```

**复杂示例**:
```python
# 添加更多信息
tree = DocumentTree.from_predictions(
    line_labels=line_labels,
    parent_indices=parent_indices,
    relation_types=relation_types,
    line_bboxes=[[100, 100, 500, 150], ...],
    line_texts=["Title", "Section 1", ...],
    label_confidences=[0.99, 0.95, ...],
    relation_confidences=[0.98, 0.92, ...],
)

# 查询节点
node = tree.get_node_by_idx(2)
print(f"节点2: {node.label}, 父节点: {node.parent.label}")

# 获取路径
path = tree.get_path_to_root(node)
for n in path:
    print(f"  {n.label}")

# 统计
stats = tree.get_statistics()
print(f"总节点数: {stats['total_nodes']}")
print(f"最大深度: {stats['max_depth']}")
```

### 3. 运行End-to-End评估 (2分钟)

```bash
# 评估完整pipeline的性能
python evaluate_end_to_end.py
```

**输出示例**:
```
【SubTask 2: Parent Finding】
  F1: 0.9562

【SubTask 3: Relation Classification】
  F1 (macro): 0.9081

【End-to-End: 父节点 + 关系都正确】
  准确率: 0.8421
```

---

## 30分钟实战演练

### 任务: 处理真实验证集并分析结果

#### 1. 完整推理（10分钟）

```bash
# 处理所有验证集样本
python inference_build_tree.py \
    --subtask2_model /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt \
    --subtask3_model /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt \
    --features_dir /mnt/e/models/train_data/layoutlmft/line_features \
    --output_dir ./outputs/trees_full \
    --max_samples -1 \
    --max_chunks -1 \
    --save_json \
    --save_markdown

# 这会生成约8000个树文件，可能需要几分钟
```

#### 2. 分析结果（10分钟）

```python
import json
from pathlib import Path

# 读取summary
with open("./outputs/trees_full/summary.json") as f:
    summary = json.load(f)

print(f"处理了 {summary['total_pages']} 个页面")
print(f"平均每页 {summary['average_nodes_per_page']:.1f} 个节点")

# 读取单个树
with open("./outputs/trees_full/tree_0000.json") as f:
    tree_data = json.load(f)

# 分析标签分布
label_dist = tree_data['statistics']['label_distribution']
print("\n标签分布:")
for label, count in sorted(label_dist.items(), key=lambda x: -x[1])[:5]:
    print(f"  {label}: {count}")
```

#### 3. 可视化最佳案例（10分钟）

```bash
# 找出节点数最多的文档（结构最复杂）
python -c "
import json
from pathlib import Path

trees = []
for p in Path('./outputs/trees_full').glob('tree_*.json'):
    with open(p) as f:
        data = json.load(f)
        trees.append((p.stem, data['num_nodes']))

trees.sort(key=lambda x: -x[1])
for name, num_nodes in trees[:5]:
    print(f'{name}: {num_nodes} nodes')
"

# 查看最复杂的文档
cat ./outputs/trees_full/tree_XXXX.md
cat ./outputs/trees_full/tree_XXXX_ascii.txt
```

---

## 常见用例

### 用例1: 提取文档大纲

```python
from document_tree import DocumentTree
import json

# 加载树
with open("tree_0000.json") as f:
    data = json.load(f)

# 提取大纲（只保留Title和Section）
def extract_outline(node, level=0):
    if node['label'] in ['Title', 'Section', 'Para-Title']:
        print("  " * level + f"- {node['label']}: {node['text']}")

    for child in node.get('children', []):
        extract_outline(child, level + 1)

extract_outline(data['root'])
```

### 用例2: 统计文档元素

```python
def count_elements(tree_data):
    stats = tree_data['statistics']

    paragraphs = stats['label_distribution'].get('Para-Line', 0)
    sections = stats['label_distribution'].get('Section', 0)
    tables = stats['label_distribution'].get('Table-Title', 0)
    figures = stats['label_distribution'].get('Figure-Title', 0)

    print(f"段落数: {paragraphs}")
    print(f"章节数: {sections}")
    print(f"表格数: {tables}")
    print(f"图片数: {figures}")
```

### 用例3: 验证树结构合理性

```python
def validate_tree(tree_data):
    issues = []

    # 检查深度
    if tree_data['statistics']['max_depth'] > 10:
        issues.append(f"深度过深: {tree_data['statistics']['max_depth']}")

    # 检查关系分布
    relations = tree_data['statistics']['relation_distribution']
    if relations.get('none', 0) > relations.get('contain', 0):
        issues.append("none关系过多，可能有预测错误")

    return issues
```

---

## 下一步

### 优化方向

1. **提升性能**:
   - 批量推理
   - GPU加速
   - 缓存优化

2. **增强功能**:
   - 添加SubTask 1实时推理
   - 实现段落合并
   - HTML/SVG可视化

3. **改进质量**:
   - 基于规则的后处理
   - 不确定性估计
   - 错误检测和修正

### 学习资源

- **论文**: [HRDoc](https://ar5iv.labs.arxiv.org/html/2303.13839)
- **代码**: `examples/tree/` 目录
- **文档**: `README.md`, `IMPLEMENTATION.md`

---

## 故障排查

### 问题1: ModuleNotFoundError

```bash
# 确保在正确的目录
cd /root/code/layoutlmft/examples/tree

# 或添加路径
export PYTHONPATH=/root/code/layoutlmft:$PYTHONPATH
```

### 问题2: 找不到模型文件

```bash
# 检查模型路径
ls /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt
ls /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt

# 如果不存在，需要先训练
python examples/train_parent_finder.py --mode simple
python examples/train_multiclass_relation.py
```

### 问题3: GPU内存不足

```bash
# 使用CPU
export CUDA_VISIBLE_DEVICES=""

# 或减少batch size（在代码中修改）
```

---

## 总结

**你已经学会了**:
1. ✅ 理解HRDoc的Overall Task
2. ✅ 使用DocumentTree构建树
3. ✅ 运行完整推理pipeline
4. ✅ 分析和可视化结果

**下一步行动**:
- 在自己的数据上测试
- 根据需求定制后处理
- 集成到实际应用中

**需要帮助?**
- 查看示例: `example_usage.py`
- 阅读文档: `README.md`
- 运行测试: `quick_test.sh`
