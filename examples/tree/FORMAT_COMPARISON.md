# 数据格式对比：HRDS原始 vs 我们生成的树

## 1. HRDS原始格式（平铺列表）

```json
[
    {
        "text": "Learning to Understand Child-directed...",
        "box": [95, 70, 502, 84],
        "class": "title",
        "page": 0,
        "is_meta": true,
        "line_id": 0,
        "parent_id": -1,
        "relation": "meta"
    },
    {
        "text": "Lieke Gelderloos",
        "box": [107, 119, 194, 131],
        "class": "author",
        "page": 0,
        "is_meta": true,
        "line_id": 1,
        "parent_id": -1,
        "relation": "meta"
    },
    ...
]
```

**特点**：
- ✅ **平铺列表** - 所有行在一个数组中
- ✅ **parent_id字段** - 通过parent_id引用父节点
- ✅ **relation字段** - 直接存储关系类型
- ✅ **包含文本** - 有text字段
- ✅ **简单直接** - 易于存储和查询

**结构**：
```
扁平化表示，需要通过parent_id重建树
```

---

## 2. 我们生成的格式（嵌套树）

```json
{
    "root": {
        "idx": -1,
        "label": "ROOT",
        "bbox": [0, 0, 0, 0],
        "text": null,
        "relation_to_parent": null,
        "children": [
            {
                "idx": 0,
                "label": "Title",
                "bbox": [127.0, 83.0, 872.0, 99.0],
                "text": null,
                "relation_to_parent": "contain",
                "relation_confidence": 0.95,
                "children": [
                    {
                        "idx": 1,
                        "label": "Section",
                        ...
                        "children": [...]
                    }
                ]
            }
        ]
    }
}
```

**特点**：
- ✅ **嵌套树结构** - 直接体现层次关系
- ✅ **children数组** - 父节点包含子节点
- ✅ **relation_to_parent** - 存储与父节点的关系
- ✅ **置信度** - 额外的relation_confidence
- ⚠️ **缺少文本** - 当前text为null（可以添加）

**结构**：
```
树形表示，直接可视化父子关系
```

---

## 3. 主要差异

| 维度 | HRDS原始 | 我们的树 | 说明 |
|------|----------|----------|------|
| **数据结构** | 平铺列表 | 嵌套树 | HRDS是数组，我们是树 |
| **父子关系** | parent_id引用 | children嵌套 | HRDS用ID，我们直接嵌套 |
| **文本内容** | 有text字段 | text=null | 需要添加 |
| **置信度** | 无 | 有confidence | 我们额外提供 |
| **可视化** | 需要重建 | 直接可视化 | 树结构更直观 |

---

## 4. 字段映射

| HRDS字段 | 我们的字段 | 说明 |
|----------|-----------|------|
| `line_id` | `idx` | 行索引 |
| `class` | `label` | 语义类别 |
| `box` | `bbox` | 边界框 |
| `text` | `text` | 文本内容（当前缺失） |
| `parent_id` | (隐含在children中) | 通过嵌套表示 |
| `relation` | `relation_to_parent` | 关系类型 |
| `page` | 无 | 页码（可添加） |
| `is_meta` | 无 | 元数据标记（可添加） |

---

## 5. 转换示例

### HRDS格式 → 树格式

**HRDS输入**:
```json
[
    {"line_id": 0, "text": "Title", "parent_id": -1, "relation": "meta"},
    {"line_id": 1, "text": "Section 1", "parent_id": 0, "relation": "contain"},
    {"line_id": 2, "text": "Paragraph", "parent_id": 1, "relation": "contain"}
]
```

**转换为我们的树**:
```json
{
    "root": {
        "children": [
            {
                "idx": 0,
                "text": "Title",
                "relation_to_parent": "none",
                "children": [
                    {
                        "idx": 1,
                        "text": "Section 1",
                        "relation_to_parent": "contain",
                        "children": [
                            {
                                "idx": 2,
                                "text": "Paragraph",
                                "relation_to_parent": "contain"
                            }
                        ]
                    }
                ]
            }
        ]
    }
}
```

---

## 6. 优缺点对比

### HRDS平铺格式

**优点**：
- ✅ 存储效率高
- ✅ 易于数据库存储
- ✅ 查询单个节点快
- ✅ 包含所有原始信息

**缺点**：
- ❌ 需要手动重建树
- ❌ 不直观
- ❌ 难以可视化

### 我们的嵌套树格式

**优点**：
- ✅ 直观展示层次结构
- ✅ 易于可视化
- ✅ 自然表达父子关系
- ✅ 包含置信度信息

**缺点**：
- ❌ 存储冗余（深度嵌套）
- ❌ 查询特定节点较慢
- ❌ 当前缺少文本内容

---

## 7. 如何添加缺失的字段？

### 添加文本内容

如果你有OCR结果或原始文本，可以在构建树时传入：

```python
tree = DocumentTree.from_predictions(
    line_labels=line_labels,
    parent_indices=parent_indices,
    relation_types=relation_types,
    line_bboxes=line_bboxes,
    line_texts=line_texts,  # ← 添加这个！
)
```

### 添加其他字段

修改`DocumentNode`类，添加新字段：

```python
class DocumentNode:
    def __init__(self, ..., page=None, is_meta=None):
        ...
        self.page = page
        self.is_meta = is_meta
```

---

## 8. 两种格式的使用场景

### 使用HRDS平铺格式
- 数据库存储
- 批量查询
- 数据交换
- 训练数据集

### 使用我们的树格式
- 可视化展示
- 层次分析
- 文档结构理解
- 前端渲染

---

## 9. 格式转换工具

### 树格式 → HRDS平铺格式

```python
def tree_to_flat(tree):
    """将嵌套树转换为HRDS平铺格式"""
    flat_list = []

    def traverse(node, parent_id=-1):
        if node.idx >= 0:  # 跳过ROOT
            flat_list.append({
                "line_id": node.idx,
                "text": node.text,
                "box": node.bbox,
                "class": node.label.lower(),
                "parent_id": parent_id,
                "relation": node.relation_to_parent or "none",
            })

        for child in node.children:
            traverse(child, node.idx)

    traverse(tree.root)
    return flat_list
```

### HRDS平铺格式 → 树格式

这就是我们`DocumentTree.from_predictions()`做的事情！

---

## 10. 结论

| 问题 | 答案 |
|------|------|
| **结构是否相同？** | ❌ 不同 - HRDS是平铺列表，我们是嵌套树 |
| **信息是否等价？** | ✅ 几乎等价 - 包含相同的父子关系和语义信息 |
| **缺少什么？** | ⚠️ 文本内容(text)、页码(page)、元数据标记(is_meta) |
| **可以转换吗？** | ✅ 可以双向转换 |

**建议**：
1. 如果需要HRDS格式，添加转换函数
2. 如果需要文本，从OCR结果中提取并传入
3. 两种格式各有用途，保持都可以互转即可
