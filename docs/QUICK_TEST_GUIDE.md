# 小规模训练测试指南

## 目的

快速验证文档级别训练代码的正确性，无需等待完整训练（完整训练可能需要数小时）。

## 测试规模

- **文档数量**: 10个文档（而不是全部500+个）
- **训练轮数**: 1 epoch（ParentFinder），50 steps（关系分类器）
- **预计时间**: 约10-15分钟

## 方法一：一键测试脚本（推荐）

```bash
# 运行完整的小规模测试（包括特征提取 + 两个模型训练）
bash scripts/quick_test_document_training.sh
```

该脚本会自动完成：
1. 提取10个文档的特征
2. 训练 ParentFinder (1 epoch)
3. 训练关系分类器 (50 steps)

## 方法二：手动分步测试

### 步骤1: 提取小规模特征（10个文档）

```bash
export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc_test"
export LAYOUTLMFT_NUM_SAMPLES="10"      # 关键：只处理10个文档
export LAYOUTLMFT_DOCS_PER_CHUNK="5"    # 每chunk 5个文档

python examples/extract_line_features_document_level.py
```

**预期输出**:
```
处理文档 [0/10]: 文档名1
  - 页数: 6, 行数: 89
处理文档 [1/10]: 文档名2
  - 页数: 4, 行数: 67
...
✓ train 集完成!
  总文档数: 10
  保存的chunk文件数: 2
```

**验证**:
```bash
ls -lh /mnt/e/models/train_data/layoutlmft/line_features_doc_test/
# 应该看到:
#   train_line_features_chunk_0000.pkl
#   train_line_features_chunk_0001.pkl
#   validation_line_features_chunk_0000.pkl (如果有验证集)
```

### 步骤2: 训练 ParentFinder（1 epoch）

```bash
python examples/train_parent_finder.py \
    --mode full \
    --level document \
    --batch_size 1 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --features_dir "/mnt/e/models/train_data/layoutlmft/line_features_doc_test" \
    --output_dir "/mnt/e/models/train_data/layoutlmft/line_features_doc_test"
```

**预期输出**:
```
============================================================
模式: 完整论文方法（需要24GB显存）
级别: document (max_lines=512)
============================================================
使用设备: cuda
[ChunkIterable] 找到 2 个chunk文件
[ChunkIterable] 总计 10 页（实际是10个文档）

===== Epoch 1/1 =====
训练: 100%|████████| 10/10 [00:15<00:00,  0.65it/s]
训练 - Loss: 2.3456, Acc: 0.1234
验证 - Acc: 0.0987
✓ 保存最佳模型 (Acc: 0.0987)
```

**注意**: 由于数据量很小，准确率会很低，这是正常的！我们只是验证代码能运行。

### 步骤3: 训练关系分类器（50 steps）

```bash
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc_test"
export LAYOUTLMFT_OUTPUT_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc_test"
export MAX_CHUNKS="-1"  # 使用全部chunk（测试时只有2个）

# 使用简化的Python脚本
python -c "
import os
os.environ['LAYOUTLMFT_FEATURES_DIR'] = '/mnt/e/models/train_data/layoutlmft/line_features_doc_test'

# 修改 max_steps
import sys
sys.path.insert(0, '/root/code/layoutlmft')

# 加载模块
exec(open('examples/train_multiclass_relation.py').read().replace('max_steps = 300', 'max_steps = 50'))
"
```

或者直接修改 `train_multiclass_relation.py` 的 `max_steps` 参数后运行：
```bash
# 临时修改
sed -i 's/max_steps = 300/max_steps = 50/' examples/train_multiclass_relation.py

python examples/train_multiclass_relation.py

# 改回来
sed -i 's/max_steps = 50/max_steps = 300/' examples/train_multiclass_relation.py
```

## 预期时间

| 步骤 | 预计时间 | 说明 |
|------|---------|------|
| 特征提取 (10文档) | 2-3分钟 | 包括模型加载和前向传播 |
| ParentFinder (1 epoch) | 3-5分钟 | batch_size=1, 10个文档 |
| 关系分类器 (50 steps) | 2-3分钟 | batch_size=32 |
| **总计** | **10-15分钟** | 取决于GPU性能 |

## 检查训练是否正常

### 1. 检查损失下降

ParentFinder:
```
Epoch 1 - Loss: 2.5 → 应该逐渐下降（但数据太少可能不明显）
```

关系分类器:
```
Step 10 - Loss: 1.2
Step 20 - Loss: 1.0  ← 应该下降
Step 30 - Loss: 0.9
Step 40 - Loss: 0.8
Step 50 - Loss: 0.7
```

### 2. 检查模型保存

```bash
# ParentFinder 模型
ls -lh /mnt/e/models/train_data/layoutlmft/line_features_doc_test/parent_finder_full/best_model.pt

# 关系分类器模型
ls -lh /mnt/e/models/train_data/layoutlmft/line_features_doc_test/multiclass_relation/best_model.pt
```

### 3. 检查数据加载

训练日志应该显示：
```
总共加载了 10 个样本的特征（页面或文档）
训练集大小: XXX
验证集大小: YYY
```

## 常见问题

**Q: 准确率很低（只有10%左右），是不是代码有问题？**

A: 正常！10个文档的数据量太小，模型无法学到有效模式。小规模测试的目的是验证代码能运行，不是验证性能。

**Q: 显存不足怎么办？**

A:
- 减少 batch_size: `--batch_size 1`（已经是最小）
- 减少文档数: `LAYOUTLMFT_NUM_SAMPLES="5"`
- 降低 max_lines_limit: `--max_lines_limit 256`

**Q: 训练速度很慢？**

A:
- 确认使用GPU: 日志应显示 `使用设备: cuda`
- 检查CUDA可用: `python -c "import torch; print(torch.cuda.is_available())"`
- 如果只有CPU，可以进一步减少数据量

**Q: 如何从测试切换到完整训练？**

A:
```bash
# 1. 去掉样本数限制
unset LAYOUTLMFT_NUM_SAMPLES  # 或设为 -1

# 2. 增加训练轮数
# ParentFinder: --num_epochs 20
# 关系分类器: 恢复 max_steps = 300

# 3. 使用正式的输出目录
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
```

## 测试成功的标志

- ✅ 所有脚本运行完成，没有报错
- ✅ 损失值在合理范围（0.5-3.0），没有 NaN 或 Inf
- ✅ 模型文件成功保存
- ✅ GPU 显存占用正常（不超过12GB）
- ✅ 训练日志显示 `level: document (max_lines=512)`

如果以上都满足，说明文档级别训练代码工作正常，可以进行完整训练！
