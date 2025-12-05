# 默认配置说明

## 概述

所有训练阶段默认使用 **文档级别（Document-level）** 配置。

## 三个阶段的默认参数

### 阶段一：特征提取

**脚本**: `examples/extract_line_features_document_level.py`

**默认参数**:
```python
LAYOUTLMFT_FEATURES_DIR = "/mnt/e/models/train_data/layoutlmft/line_features_doc"
HRDOC_DATA_DIR = "/mnt/e/models/data/Section/HRDS"
LAYOUTLMFT_MODEL_PATH = "/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
LAYOUTLMFT_NUM_SAMPLES = "-1"  # 全部文档
LAYOUTLMFT_DOCS_PER_CHUNK = "100"  # 每个chunk 100个文档
```

**直接运行**（使用默认值）:
```bash
python examples/extract_line_features_document_level.py
```

### 阶段二：ParentFinder 训练

**脚本**: `examples/train_parent_finder.py`

**默认参数**:
```python
--level = "document"           # 文档级别
--max_lines_limit = 512        # 文档级别默认512行
--mode = "simple"              # 简化模式（可选 full）
--batch_size = 128 (simple) / 1 (full)
--learning_rate = 1e-3 (simple) / 1e-4 (full)
LAYOUTLMFT_FEATURES_DIR = "/mnt/e/models/train_data/layoutlmft/line_features_doc"
```

**直接运行**（使用默认值）:
```bash
# Simple 模式
python examples/train_parent_finder.py --mode simple

# Full 模式（论文方法）
python examples/train_parent_finder.py --mode full --batch_size 1
```

### 阶段三：关系分类器训练

**脚本**: `examples/train_multiclass_relation.py`

**默认参数**:
```python
LAYOUTLMFT_FEATURES_DIR = "/mnt/e/models/train_data/layoutlmft/line_features_doc"
LAYOUTLMFT_OUTPUT_DIR = "/mnt/e/models/train_data/layoutlmft"
max_steps = 300
batch_size = 32
learning_rate = 5e-4
neg_ratio = 1.5
```

**直接运行**（使用默认值）:
```bash
python examples/train_multiclass_relation.py
```

## 如何切换到页面级别？

如果需要使用页面级别（Page-level），可以通过以下方式：

### 方法一：环境变量

```bash
# 指定页面级别特征目录
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features"

# 运行训练（会自动使用页面级别特征）
python examples/train_parent_finder.py --level page
python examples/train_multiclass_relation.py
```

### 方法二：命令行参数

```bash
# ParentFinder
python examples/train_parent_finder.py \
    --mode full \
    --level page \
    --max_lines_limit 256 \
    --features_dir "/mnt/e/models/train_data/layoutlmft/line_features"
```

## 默认路径统一

所有阶段使用相同的默认路径，确保数据流顺畅：

```
阶段一（提取）输出 → /mnt/e/models/train_data/layoutlmft/line_features_doc/
                    ↓
阶段二（ParentFinder）读取 ← 同一目录
                    ↓
阶段三（关系分类器）读取 ← 同一目录
```

## 快速开始（完整流程）

使用所有默认值运行完整流程：

```bash
# 1. 激活环境
conda activate layoutlmv2

# 2. 特征提取（默认输出到 line_features_doc）
python examples/extract_line_features_document_level.py

# 3. 训练 ParentFinder（默认从 line_features_doc 读取）
python examples/train_parent_finder.py --mode full --batch_size 1 --num_epochs 20

# 4. 训练关系分类器（默认从 line_features_doc 读取）
python examples/train_multiclass_relation.py
```

## 优势

1. **零配置启动**: 直接运行脚本即可，无需设置环境变量
2. **路径一致性**: 所有阶段自动对齐，避免路径错误
3. **文档级别优先**: 符合 HRDoc 论文的设计，支持跨页关系
4. **向后兼容**: 仍可通过参数切换到页面级别

## 常见问题

**Q: 如何验证当前使用的是文档级别还是页面级别？**

A: 查看训练日志开头：
```
============================================================
模式: 完整论文方法（需要24GB显存）
级别: document (max_lines=512)  ← 这里显示当前级别
============================================================
```

**Q: 默认路径不存在怎么办？**

A: 脚本会自动创建目录。如果路径不可访问（如 E 盘未挂载），可以通过环境变量或参数指定其他路径：
```bash
export LAYOUTLMFT_FEATURES_DIR="/tmp/line_features_doc"
python examples/extract_line_features_document_level.py
```

**Q: 如何使用测试模式（少量数据）？**

A: 设置环境变量限制样本数：
```bash
export LAYOUTLMFT_NUM_SAMPLES="20"  # 只处理20个文档
python examples/extract_line_features_document_level.py
```

训练时限制 chunks：
```bash
python examples/train_parent_finder.py --max_chunks 1  # 只用1个chunk
```
