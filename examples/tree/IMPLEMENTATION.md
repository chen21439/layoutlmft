# HRDoc Overall Task 实现说明

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    HRDoc 完整 Pipeline                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  SubTask 1   │     │  SubTask 2   │     │  SubTask 3   │
│              │     │              │     │              │
│   语义分类    │ →  │  父节点查找   │ →  │   关系分类    │
│ (LayoutLMv2) │     │(ParentFinder)│     │(RelationCls) │
└──────────────┘     └──────────────┘     └──────────────┘
      ↓                     ↓                     ↓
 line_labels          parent_indices        relation_types
      │                     │                     │
      └─────────────────────┼─────────────────────┘
                            ↓
                  ┌──────────────────┐
                  │  Overall Task    │
                  │   (本目录实现)    │
                  │                  │
                  │ DocumentTree     │
                  │   构建树结构      │
                  └──────────────────┘
                            ↓
                  ┌──────────────────┐
                  │  输出格式         │
                  │                  │
                  │ - JSON           │
                  │ - Markdown       │
                  │ - ASCII Tree     │
                  │ - 统计信息        │
                  └──────────────────┘
```

## 核心实现

### 1. DocumentNode (文档节点)

**文件**: `document_tree.py`

**功能**: 表示文档结构树中的一个节点

**属性**:
- `idx`: 节点索引（行号）
- `label`: 语义类别（Title, Section, Para-Line, etc.）
- `bbox`: 边界框 [x1, y1, x2, y2]
- `text`: 文本内容
- `parent`: 父节点引用
- `children`: 子节点列表
- `relation_to_parent`: 关系类型（connect/contain/equality）

**方法**:
- `add_child()`: 添加子节点
- `to_dict()`: 转换为字典（用于JSON序列化）

### 2. DocumentTree (文档树)

**文件**: `document_tree.py`

**功能**: 文档结构树，组合三个子任务的输出

**核心方法**:

```python
@classmethod
def from_predictions(
    cls,
    line_labels: List[int],      # SubTask 1 输出
    parent_indices: List[int],    # SubTask 2 输出
    relation_types: List[int],    # SubTask 3 输出
    ...
) -> 'DocumentTree':
    """从三个子任务的预测结果构建树"""
```

**可视化方法**:
- `to_json()`: 导出为JSON
- `to_markdown()`: 导出为Markdown
- `visualize_ascii()`: ASCII树可视化
- `get_statistics()`: 获取统计信息

**后处理方法**:
- `prune_none_relations()`: 剪枝无效关系
- `get_path_to_root()`: 获取节点路径
- `get_node_by_idx()`: 节点查询

### 3. 推理Pipeline

**文件**: `inference_build_tree.py`

**流程**:

```python
def inference_single_page(page_data, models, device):
    # 1. 提取特征（SubTask 1已完成，使用缓存）
    line_features = page_data["line_features"]
    line_labels = page_data["line_labels"]

    # 2. 预测父节点（SubTask 2）
    parent_indices = predict_parents(
        subtask2_model, line_features, ...
    )

    # 3. 预测关系（SubTask 3）
    relation_types = predict_relations(
        subtask3_model, line_features, parent_indices, ...
    )

    # 4. 构建树
    tree = DocumentTree.from_predictions(
        line_labels, parent_indices, relation_types, ...
    )

    return tree
```

## 文件清单

### 核心文件

1. **`document_tree.py`** (294行)
   - DocumentNode 类定义
   - DocumentTree 类定义
   - 树构建逻辑
   - 可视化方法
   - 包含完整的demo

2. **`inference_build_tree.py`** (429行)
   - 完整推理pipeline
   - 模型加载
   - 批量推理
   - 结果输出

3. **`evaluate_end_to_end.py`** (570行)
   - End-to-End评估
   - 误差累积分析
   - 性能报告

### 辅助文件

4. **`README.md`**
   - 使用文档
   - API说明
   - 常见问题

5. **`IMPLEMENTATION.md`** (本文件)
   - 实现说明
   - 架构设计

6. **`example_usage.py`** (359行)
   - 5个使用示例
   - 演示各种功能

7. **`quick_test.sh`**
   - 快速测试脚本
   - 端到端验证

## 使用流程

### 方式1: 使用命令行工具

```bash
# 1. 测试树构建
python examples/tree/document_tree.py

# 2. 完整推理（处理10个样本）
python examples/tree/inference_build_tree.py \
    --subtask2_model /path/to/parent_finder/best_model.pt \
    --subtask3_model /path/to/relation_classifier/best_model.pt \
    --features_dir /path/to/line_features \
    --output_dir ./outputs/trees \
    --max_samples 10 \
    --save_json --save_markdown --save_ascii

# 3. End-to-End评估
python examples/tree/evaluate_end_to_end.py
```

### 方式2: 使用Python API

```python
from document_tree import DocumentTree

# 假设已经有了三个子任务的输出
tree = DocumentTree.from_predictions(
    line_labels=[0, 1, 3, 3],
    parent_indices=[-1, 0, 1, 2],
    relation_types=[0, 2, 2, 1],
    line_texts=["Title", "Section", "Para 1", "Para 2"]
)

# 可视化
print(tree.visualize_ascii())
print(tree.to_markdown())

# 导出
tree.to_json("output.json")

# 统计
stats = tree.get_statistics()
print(f"总节点数: {stats['total_nodes']}")
```

## 数据流

### 输入

```python
# SubTask 1 输出 (缓存在 line_features.pkl)
{
    "line_features": torch.Tensor,  # [num_lines, 768]
    "line_mask": torch.Tensor,       # [num_lines]
    "line_labels": List[int],        # 语义类别
    "line_bboxes": List[List[float]], # 边界框
    "line_parent_ids": List[int],    # GT父节点（评估用）
    "line_relations": List[str],     # GT关系（评估用）
}
```

### 处理

```python
# SubTask 2: 父节点预测
for each child in lines:
    candidates = lines[0:child_idx]
    scores = model(child, candidates)
    parent = argmax(scores)

# SubTask 3: 关系预测
for each (parent, child) pair:
    relation = model(parent, child, geom_features)
```

### 输出

```python
# DocumentTree
{
    "root": {
        "idx": -1,
        "label": "ROOT",
        "children": [...]
    },
    "statistics": {
        "total_nodes": 42,
        "max_depth": 5,
        "label_distribution": {...},
        "relation_distribution": {...}
    }
}
```

## 关键设计决策

### 1. 为什么使用树结构？

- **符合论文**: HRDoc论文明确提出文档是层次结构
- **直观表达**: 树能清晰表达父子关系和嵌套结构
- **易于操作**: 支持遍历、查询、修剪等操作

### 2. 为什么分离SubTask 1？

- **复用性**: line_features可以被多次使用（SubTask 2和3都需要）
- **效率**: 避免重复运行LayoutLMv2模型
- **模块化**: 各个子任务可以独立开发和优化

### 3. 为什么支持多种输出格式？

- **JSON**: 机器可读，便于后续处理
- **Markdown**: 人类可读，便于文档和报告
- **ASCII**: 调试和快速查看
- **统计**: 分析和评估

### 4. 后处理的作用？

- **剪枝**: 移除错误预测的关系
- **合并**: 将connect关系的节点合并成段落
- **校正**: 基于规则修正明显错误

## 性能优化

### 当前实现

- **串行推理**: 逐个样本处理
- **CPU/GPU混合**: 特征在GPU，其他在CPU
- **无缓存**: 每次重新计算

### 可优化方向

1. **批量推理**: 一次处理多个样本
2. **预计算**: 缓存几何特征
3. **并行化**: 多进程处理
4. **剪枝策略**: 减少候选父节点数量

## 评估指标

### SubTask 级别

- **SubTask 2**: 父节点准确率
- **SubTask 3**: 关系分类F1（macro）

### End-to-End 级别

- **完全正确率**: 父节点和关系都正确
- **误差累积**: 分析SubTask 2错误对SubTask 3的影响

### 树级别

- **结构准确率**: 树的拓扑结构是否正确
- **深度分布**: 树的深度是否合理
- **关系分布**: 各种关系类型的比例

## 已知限制

1. **SubTask 1依赖缓存**: 当前未实现实时推理
2. **内存占用**: 大文档可能占用较多内存
3. **错误累积**: SubTask 2的错误会影响SubTask 3
4. **后处理简单**: 仅实现了基本的剪枝

## 扩展方向

### 短期

- [ ] 添加SubTask 1的实时推理
- [ ] 实现段落合并（基于connect关系）
- [ ] 添加HTML/SVG可视化
- [ ] 支持批量推理加速

### 长期

- [ ] 基于规则的后处理优化
- [ ] 联合训练三个子任务
- [ ] 添加不确定性估计
- [ ] 支持多模态输入（图像+文本）

## 测试

### 单元测试

```bash
# 测试树构建
python examples/tree/document_tree.py

# 测试示例
python examples/tree/example_usage.py
```

### 集成测试

```bash
# 完整流程测试
bash examples/tree/quick_test.sh
```

### 性能测试

```bash
# End-to-End评估
python examples/tree/evaluate_end_to_end.py
```

## 参考文献

1. HRDoc论文: https://ar5iv.labs.arxiv.org/html/2303.13839
2. LayoutLMv2: https://arxiv.org/abs/2012.14740
3. HRDoc数据集: https://github.com/microsoft/HRDoc

## 贡献者

- 实现日期: 2025-11-15
- 基于: HRDoc论文和LayoutLMv2

## 许可

遵循项目主许可证
