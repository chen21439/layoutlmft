# 文档级别训练实现文档

## 概述

本文档说明如何使用文档级别（Document-level）训练代替页面级别（Page-level）训练。

### 页面级别 vs 文档级别

| 特性 | 页面级别 | 文档级别 |
|------|---------|---------|
| 数据单位 | 每个样本 = 一页 | 每个样本 = 整个文档（多页） |
| 父子关系 | 仅限页内 | 支持跨页关系 |
| line_id | 页内局部编号 (0-N) | 文档内全局编号 (0-M) |
| parent_id | 页内局部索引 | 文档内全局索引 |
| 序列长度 | 通常 < 100 行/页 | 可达 512 行/文档 |
| 显存占用 | 较小 | 较大 |

### 为什么需要文档级别？

根据 HRDoc 论文 (https://ar5iv.labs.arxiv.org/html/2303.13839)：

1. **跨页关系**: 文档的层次结构可能跨越多页（例如：第2页的标题是第1页某个章节的子节点）
2. **全局上下文**: 完整文档提供更丰富的上下文信息
3. **论文一致性**: 论文使用文档级别的 line_id 和 parent_id

## 实现的修改

### 1. 特征提取 (Feature Extraction)

#### 新增文件
- `examples/extract_line_features_document_level.py`
  - 替代: `examples/extract_line_features.py` (页面级别)
  - 功能: 按文档聚合多个页面，提取文档级别特征

#### 关键区别
```python
# 页面级别 - 每页独立处理
for page in pages:
    features = extract_page_features(page)  # [num_lines_on_page, H]
    save(features)

# 文档级别 - 整个文档聚合
all_features = []
for page in document.pages:
    page_features = extract_page_features(page)  # [lines_on_page, H]
    all_features.append(page_features[:num_valid_lines])  # 去除padding

document_features = concatenate(all_features)  # [total_lines_in_doc, H]
save(document_features)
```

#### 输出数据结构
```python
{
    "line_features": torch.Tensor,  # [1, total_lines, 768]
    "line_mask": torch.Tensor,      # [1, total_lines]
    "line_parent_ids": list,        # length = total_lines (全局索引)
    "line_relations": list,         # length = total_lines
    "line_bboxes": np.array,        # [total_lines, 4]
    "line_labels": list,            # length = total_lines
    "document_name": str,           # 文档名称
    "num_pages": int,               # 页数
    "num_lines": int                # 总行数
}
```

#### 使用脚本
```bash
# 测试脚本 (20个文档)
bash scripts/test_document_level_extraction.sh

# 完整提取 (全部训练集)
export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/path/to/checkpoint"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
export LAYOUTLMFT_NUM_SAMPLES="-1"  # -1 = 全部
export LAYOUTLMFT_DOCS_PER_CHUNK="100"

python examples/extract_line_features_document_level.py
```

### 2. ParentFinder 训练

#### 修改文件
- `examples/train_parent_finder.py`

#### 新增参数
```bash
python examples/train_parent_finder.py \
    --mode full \
    --level document \           # 新增: document 或 page
    --max_lines_limit 512 \      # 新增: 文档级别推荐512
    --batch_size 1 \
    --num_epochs 10 \
    --features_dir /path/to/line_features_doc
```

#### 关键修改
1. **max_lines_limit 默认值**
   - 页面级别: 256
   - 文档级别: 512

2. **collate_fn 函数**
   - 支持更长的序列 (最多512行)
   - 动态 padding 和 truncation

3. **Dataset 类**
   - `ParentFinderDataset`: 同时支持页面和文档
   - `ChunkIterableDataset`: 流式加载，内存友好

#### 使用脚本
```bash
# 文档级别训练 (小规模测试)
bash scripts/train_parent_finder_document.sh
```

### 3. 关系分类器训练

#### 修改文件
- `examples/train_multiclass_relation.py`

#### 关键修改
1. **文档说明更新**
   - 明确支持页面级别和文档级别
   - 通过 `LAYOUTLMFT_FEATURES_DIR` 环境变量区分

2. **无需代码修改**
   - 原有逻辑已支持可变长度序列
   - 自动适配文档级别数据

#### 使用脚本
```bash
# 文档级别训练
bash scripts/train_relation_document.sh
```

## 使用指南

### 完整流程 (文档级别)

```bash
# 1. 提取文档级别特征
export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
export LAYOUTLMFT_NUM_SAMPLES="-1"
export LAYOUTLMFT_DOCS_PER_CHUNK="100"

python examples/extract_line_features_document_level.py

# 2. 训练 ParentFinder (文档级别)
python examples/train_parent_finder.py \
    --mode full \
    --level document \
    --batch_size 1 \
    --num_epochs 20 \
    --features_dir "$LAYOUTLMFT_FEATURES_DIR" \
    --output_dir "/mnt/e/models/train_data/layoutlmft"

# 3. 训练关系分类器 (文档级别)
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
python examples/train_multiclass_relation.py
```

### 对比：页面级别流程

```bash
# 1. 提取页面级别特征
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features"
python examples/extract_line_features.py

# 2. 训练 ParentFinder (页面级别)
python examples/train_parent_finder.py \
    --mode full \
    --level page \              # 使用 page
    --max_lines_limit 256 \     # 更小的限制
    --batch_size 2 \
    --features_dir "$LAYOUTLMFT_FEATURES_DIR"

# 3. 训练关系分类器 (页面级别)
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features"
python examples/train_multiclass_relation.py
```

## 重要修复记录

### Bug #1: Token→Line 转换错误

**问题**: 最初误认为 `line_parent_ids` 是 token 级别，尝试从 token 映射到 line

**真相**: `line_parent_ids` 本身就是 LINE 级别！

**证据** (来自 `hrdoc.py:263`):
```python
# 每个 line 只添加一次
line_parent_ids.append(parent_id)  # 注意：不是为每个token添加
```

**修复**: 直接使用 `raw_sample["line_parent_ids"]`，无需转换

### Bug #2: 字段长度不一致

**问题**: 提取后发现 `features` 有 363 行，但 `parent_ids` 有 626 行

**原因**: 只对 `features` 应用了 `[:num_valid_lines]` 切片，其他字段遗漏

**修复** (`extract_line_features_document_level.py:303-306`):
```python
all_line_features.append(page_line_features[:num_valid_lines])
all_line_parent_ids.extend(page_line_parent_ids[:num_valid_lines])
all_line_relations.extend(page_line_relations[:num_valid_lines])
all_line_bboxes.append(line_bboxes[:num_valid_lines])
all_line_labels.extend(line_labels[:num_valid_lines])
```

## 性能和资源

### 显存占用对比

| 模式 | batch_size | max_lines | 显存占用 (估算) |
|------|-----------|-----------|----------------|
| 页面级别 (simple) | 128 | 256 | ~4 GB |
| 页面级别 (full) | 2 | 256 | ~8 GB |
| 文档级别 (full) | 1 | 512 | ~12 GB |

### 训练速度

- 文档级别训练速度较慢（序列更长）
- 建议使用 `--gradient_accumulation_steps` 模拟更大的 batch

## 后续任务

- [ ] 修改 `inference_build_tree.py`，支持文档级别推理
- [ ] 完整流程测试
- [ ] 性能对比（页面级别 vs 文档级别）

## 参考文档

- HRDoc 论文: https://ar5iv.labs.arxiv.org/html/2303.13839
- HRDS 数据集说明: `/mnt/e/models/data/Section/HRDS/README.md`
- 依赖管理: `docs/DEPENDENCIES.md`
- ParentFinder 实现说明: `docs/parent_finding_implementation.md`
