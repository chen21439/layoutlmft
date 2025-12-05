# 文档级别训练 - 修改摘要

## 修改概述

将页面级别训练改为文档级别训练，以支持跨页的父子关系。

## 修改的文件

### 1. 训练代码修改

#### `examples/train_parent_finder.py`
**修改内容**:
- 新增 `--level` 参数: 选择 `page` 或 `document` 模式
- 新增 `--max_lines_limit` 参数: 可自定义最大行数
- `collate_fn` 默认 `max_lines_limit` 从 256 提升到 512
- 根据 `level` 自动设置合理的 `max_lines_limit`:
  - `page`: 256 行
  - `document`: 512 行
- 更新文档字符串，说明支持文档级别

**使用方式**:
```bash
python examples/train_parent_finder.py \
    --mode full \
    --level document \    # 关键参数
    --batch_size 1 \
    --features_dir /path/to/line_features_doc
```

#### `examples/train_multiclass_relation.py`
**修改内容**:
- 更新文档字符串，说明支持文档级别
- 添加环境变量使用说明

**使用方式**:
```bash
export LAYOUTLMFT_FEATURES_DIR="/path/to/line_features_doc"
python examples/train_multiclass_relation.py
```

### 2. 新增训练脚本

#### `scripts/train_parent_finder_document.sh`
文档级别 ParentFinder 训练脚本
- 使用 `--level document`
- `max_lines_limit=512`
- 小规模测试: `--max_chunks 1`

#### `scripts/train_relation_document.sh`
文档级别关系分类器训练脚本
- 通过 `LAYOUTLMFT_FEATURES_DIR` 指定文档级别特征目录
- 小规模测试: `MAX_CHUNKS=1`

### 3. 新增文档

#### `docs/document_level_training.md`
完整的文档级别训练指南，包含:
- 页面级别 vs 文档级别对比
- 实现细节说明
- 使用指南
- Bug 修复记录
- 性能和资源说明

#### `docs/CHANGES_DOCUMENT_LEVEL.md` (本文件)
修改摘要

## 核心技术要点

### 数据结构差异

**页面级别**:
```python
{
    "line_features": [1, ~50, 768],     # 一页约50行
    "line_parent_ids": [0, 5, 3, ...],  # 页内局部索引
}
```

**文档级别**:
```python
{
    "line_features": [1, ~363, 768],      # 整个文档可能300+行
    "line_parent_ids": [0, 5, 65, ...],   # 全局索引，可跨页
    "document_name": "文档名",
    "num_pages": 13
}
```

### 关键参数对比

| 参数 | 页面级别 | 文档级别 |
|------|---------|---------|
| max_lines_limit | 256 | 512 |
| batch_size (full mode) | 2 | 1 |
| 显存占用 | ~8 GB | ~12 GB |

## 兼容性

**向后兼容**: 所有修改都保持向后兼容
- 默认 `--level document` (新版本推荐)
- 可通过 `--level page` 使用旧的页面级别模式
- 页面级别特征文件仍可使用

## 验证状态

- [x] `extract_line_features_document_level.py` 验证通过（2个文档测试）
- [x] `train_parent_finder.py` 修改完成
- [x] `train_multiclass_relation.py` 修改完成
- [ ] 小规模训练测试 (待运行)
- [ ] 推理代码修改 (待完成)

## 后续步骤

1. 在云服务器上运行完整的文档级别特征提取
2. 运行小规模训练测试验证正确性
3. 修改推理代码 `inference_build_tree.py`
4. 完整流程测试

## 相关文档

- 实现文档: `docs/document_level_training.md`
- ParentFinder说明: `docs/parent_finding_implementation.md`
- 依赖管理: `docs/DEPENDENCIES.md`
