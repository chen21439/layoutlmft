# HRDoc 文档结构树推理 (Overall Task)

这个目录实现了 HRDoc 论文的第4步：**Overall Task**，将三个子任务的输出组合成完整的文档层次结构树。

## 背景

根据 [HRDoc 论文](https://ar5iv.labs.arxiv.org/html/2303.13839)，完整的文档结构重建需要以下步骤：

### 三个子任务

1. **SubTask 1: 语义单元分类** (`run_hrdoc.py`)
   - 输入：文档图像 + OCR结果
   - 输出：每个语义单元的类别标签（Title, Section, Para-Line, etc.）
   - 模型：LayoutLMv2
   - 作用：提取行级特征 + 语义分类

2. **SubTask 2: 父节点查找** (`train_parent_finder.py`)
   - 输入：行级特征 + 几何信息
   - 输出：每个语义单元的父节点索引
   - 模型：SimpleParentFinder / ParentFinderGRU
   - 作用：建立父子关系

3. **SubTask 3: 关系分类** (`train_multiclass_relation.py`)
   - 输入：父子对的特征 + 几何信息
   - 输出：关系类型（Connect/Contain/Equality）
   - 模型：MultiClassRelationClassifier
   - 作用：细化父子关系类型

### Overall Task（本目录实现）

**将三个子任务的输出组合成树结构**：

```
SubTask 1 → line_labels (语义类别)
SubTask 2 → parent_indices (父节点索引)
SubTask 3 → relation_types (关系类型)
             ↓
      DocumentTree (文档结构树)
             ↓
    JSON / Markdown / 可视化输出
```

## 文件说明

### 核心文件

- **`document_tree.py`**: 树结构定义和构建工具
  - `DocumentNode`: 树节点类
  - `DocumentTree`: 文档树类
  - 提供JSON/Markdown/ASCII可视化

- **`inference_build_tree.py`**: 完整推理Pipeline
  - 加载三个子任务的模型
  - 串行执行推理
  - 构建并输出树结构

### 评估文件

- **`evaluate_end_to_end.py`**: End-to-End评估脚本
  - 评估SubTask 2和SubTask 3的联合性能
  - 分析误差累积效应

## 快速开始

### 1. 准备模型

确保已经训练好三个子任务的模型：

```bash
# SubTask 1: 语义分类（LayoutLMv2）
# 训练脚本：examples/run_hrdoc.py
# 输出：/mnt/e/models/train_data/layoutlmft/hrdc_layoutlmv2/

# SubTask 2: 父节点查找
# 训练脚本：examples/train_parent_finder.py
python examples/train_parent_finder.py --mode simple --num_epochs 20

# SubTask 3: 关系分类
# 训练脚本：examples/train_multiclass_relation.py
python examples/train_multiclass_relation.py --num_epochs 20
```

### 2. 测试树构建（Demo）

```bash
cd examples/tree
python document_tree.py
```

这会生成一个演示树并输出：
- ASCII树可视化
- Markdown格式
- JSON格式
- 统计信息

### 3. 完整推理

```bash
# 基本用法（处理前10个样本）
python examples/tree/inference_build_tree.py \
    --subtask2_model /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt \
    --subtask3_model /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt \
    --features_dir /mnt/e/models/train_data/layoutlmft/line_features \
    --output_dir ./outputs/trees \
    --max_samples 10 \
    --save_json \
    --save_markdown

# 完整推理（所有验证集）
python examples/tree/inference_build_tree.py \
    --max_samples -1 \
    --max_chunks -1 \
    --save_json \
    --save_markdown \
    --save_ascii
```

### 4. End-to-End评估

```bash
python examples/tree/evaluate_end_to_end.py
```

## 输出格式

### JSON 格式

```json
{
  "root": {
    "idx": -1,
    "label": "ROOT",
    "children": [
      {
        "idx": 0,
        "label": "Title",
        "bbox": [100, 100, 500, 150],
        "text": "Document Title",
        "relation_to_parent": "none",
        "children": [
          {
            "idx": 1,
            "label": "Section",
            "bbox": [100, 200, 500, 250],
            "relation_to_parent": "contain",
            "children": [...]
          }
        ]
      }
    ]
  },
  "num_nodes": 7,
  "statistics": {
    "total_nodes": 7,
    "max_depth": 3,
    "label_distribution": {
      "Title": 1,
      "Section": 2,
      "Para-Line": 4
    },
    "relation_distribution": {
      "contain": 5,
      "connect": 2
    }
  }
}
```

### Markdown 格式

```markdown
# Document Structure Tree

- [Title] `Document Title` (none)
  - [Section] `Section 1` (contain)
    - [Para-Line] `This is a paragraph.` (contain)
      - [Para-Line] `This is another sentence.` (connect)
  - [Section] `Section 2` (contain)
    - [Para-Line] `Another paragraph here.` (contain)
```

### ASCII 可视化

```
└── Title: Document Title
    ├── Section: Section 1
    │   └── Para-Line: This is a paragraph.
    │       └── Para-Line: This is another sentence.
    └── Section: Section 2
        └── Para-Line: Another paragraph here.
```

## 树结构说明

### 节点属性

- **idx**: 节点在文档中的行号/索引
- **label**: 语义类别（Title, Section, Para-Line, etc.）
- **bbox**: 边界框 [x1, y1, x2, y2]
- **text**: 文本内容（如果有）
- **confidence**: 分类置信度
- **relation_to_parent**: 与父节点的关系类型
  - `none`: 无关系（通常是ROOT的子节点）
  - `connect`: 连接关系（同级内容的延续）
  - `contain`: 包含关系（父节点包含子节点）
  - `equality`: 等价关系（同级内容）

### 树的语义

根据 HRDoc 论文，不同的关系类型有不同的语义：

- **Contain**: 父节点包含子节点
  - 例如：Section → Para-Title → Para-Line
  - 例如：Table-Title → Table-Column-Header

- **Connect**: 同级内容的延续
  - 例如：Para-Line → Para-Line（段落内的多行）
  - 例如：List-Item → List-Item（列表项之间）

- **Equality**: 等价关系（较少使用）
  - 例如：并列的标题

## API 使用

### Python API

```python
from document_tree import DocumentTree

# 从三个子任务的预测结果构建树
tree = DocumentTree.from_predictions(
    line_labels=[0, 1, 3, 3],           # SubTask 1输出
    parent_indices=[-1, 0, 1, 2],       # SubTask 2输出
    relation_types=[0, 2, 2, 1],        # SubTask 3输出
    line_bboxes=[[100, 100, 500, 150], ...],
    line_texts=["Title", "Section 1", ...]
)

# 输出
print(tree.visualize_ascii())
print(tree.to_markdown())
tree.to_json("output.json")

# 统计
stats = tree.get_statistics()
print(f"总节点数: {stats['total_nodes']}")
print(f"最大深度: {stats['max_depth']}")
```

## 性能评估

使用 `evaluate_end_to_end.py` 评估整体性能：

```bash
python examples/tree/evaluate_end_to_end.py
```

输出示例：

```
【SubTask 2: Parent Finding】
  F1: 0.9562

【SubTask 3: Relation Classification (使用GT父节点)】
  F1 (macro): 0.9081

【SubTask 3: Relation Classification (使用预测父节点)】
  F1 (macro): 0.8750
  性能下降: 0.0331

【End-to-End: 父节点 + 关系都正确】
  准确率: 0.8421

【误差累积分析】
  父节点准确率: 0.9562
  理论最大End-to-End: 0.8684 (0.9562 * 0.9081)
  实际End-to-End: 0.8421
```

## 后处理选项

### 剪枝 (Pruning)

移除 `relation="none"` 的边：

```python
tree.prune_none_relations()
```

这会将关系预测为"none"的节点提升到更高层级。

### 合并 (Merging)

根据 `connect` 关系合并同级内容：

```python
# TODO: 实现段落合并逻辑
# 将多个 Para-Line 通过 connect 关系连接的节点合并成一个段落
```

## 常见问题

### Q1: 为什么有些节点的父节点是ROOT？

A: 可能的原因：
1. SubTask 2预测错误（没有找到合适的父节点）
2. 该节点确实是顶层节点（如文档标题）

### Q2: 关系类型的分布不合理怎么办？

A: 检查：
1. SubTask 3模型是否训练充分
2. 训练数据中关系类型的分布是否平衡
3. 是否需要调整关系分类的阈值

### Q3: 如何可视化大型文档的树？

A: 使用限制深度的方式：
```python
print(tree.visualize_ascii(max_depth=5))
```

或者导出为JSON后使用专门的树可视化工具。

## 参考资料

- [HRDoc 论文](https://ar5iv.labs.arxiv.org/html/2303.13839)
- [LayoutLMv2 论文](https://arxiv.org/abs/2012.14740)
- [HRDoc 数据集](https://github.com/microsoft/HRDoc)

## TODO

- [ ] 添加SubTask 1的完整推理（目前使用缓存的line_labels）
- [ ] 实现段落合并功能（基于connect关系）
- [ ] 添加HTML/SVG可视化
- [ ] 支持批量推理加速
- [ ] 添加树的后处理优化（基于规则修正明显错误）
