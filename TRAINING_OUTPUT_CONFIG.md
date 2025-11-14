# 训练输出配置说明

本项目所有训练产物已配置输出到 E 盘，以节省系统盘空间。

## 📁 输出目录结构

```
E:\models\train_data\layoutlmft\          (/mnt/e/models/train_data/layoutlmft/)
├── hrdoc_local/                          # LayoutLMv2 本地测试模型
├── hrdoc_quick/                          # LayoutLMv2 快速测试模型
├── line_features/                        # 行级特征文件
│   └── train_line_features.pkl           # 训练集特征缓存（~27MB）
├── relation_classifier/                  # 二分类关系分类器
└── multiclass_relation/                  # 多分类关系分类器（4类）
```

## 🎯 三个训练模型

### 1. LayoutLMv2 版面识别模型（3个配置）
| 配置 | 输出路径 | 训练步数 | 用途 | 模型大小 |
|------|---------|---------|------|---------|
| **quick** | `/mnt/e/.../hrdoc_quick` | 50步 | 快速验证代码 | ~800MB |
| **local** | `/mnt/e/.../hrdoc_local` | 500步 | 本地小规模测试 | ~800MB |
| **cloud** | `./output/hrdoc_simple_full` | 30000步 | 云服务器完整训练 | ~800MB |

**训练命令**：
```bash
# 快速测试（本地，输出到E盘）
./train_hrdoc.sh quick

# 本地测试（本地，输出到E盘）
./train_hrdoc.sh local

# 完整训练（云服务器，输出到相对路径）
./train_hrdoc.sh cloud
```

### 2. 行级特征提取
从训练好的 LayoutLMv2 模型提取行级特征，缓存到磁盘。

**输出路径**：`/mnt/e/models/train_data/layoutlmft/line_features/`

**提取命令**：
```bash
cd /root/code/layoutlmft
python examples/extract_line_features.py
```

**配置**：
- 读取模型：`./output/hrdoc_test`（可通过 `LAYOUTLMFT_MODEL_PATH` 环境变量覆盖）
- 输出特征：`/mnt/e/.../line_features/`（可通过 `LAYOUTLMFT_FEATURES_DIR` 环境变量覆盖）

### 3. 关系分类器（2种）

#### 3.1 二分类关系分类器
判断两行之间是否有层级关系（是/否）

**输出路径**：`/mnt/e/models/train_data/layoutlmft/relation_classifier/`

**训练命令**：
```bash
cd /root/code/layoutlmft
python examples/train_relation_classifier.py
```

**配置**：
- 训练步数：200步
- 负样本比例：2:1

#### 3.2 多分类关系分类器
判断关系类型：Connect/Contain/Equality/None

**输出路径**：`/mnt/e/models/train_data/layoutlmft/multiclass_relation/`

**训练命令**：
```bash
cd /root/code/layoutlmft
python examples/train_multiclass_relation.py
```

**配置**：
- 训练步数：300步
- 负样本比例：1.5:1

## 🔧 环境变量配置

支持通过环境变量自定义输出路径：

```bash
# 覆盖所有训练输出根目录
export LAYOUTLMFT_OUTPUT_DIR=/custom/path

# 覆盖特征文件目录
export LAYOUTLMFT_FEATURES_DIR=/custom/features

# 覆盖模型路径（用于特征提取）
export LAYOUTLMFT_MODEL_PATH=/custom/model

# 运行训练
./train_hrdoc.sh local  # 输出到 /custom/path/hrdoc_local
```

## 💾 磁盘空间预估

| 组件 | 单个大小 | 数量 | 总计 |
|------|---------|------|------|
| LayoutLMv2 模型 | ~800MB | 3个 | ~2.4GB |
| 行级特征缓存 | ~27MB | 1个 | ~27MB |
| 关系分类器 | ~2MB | 2个 | ~4MB |
| **合计** | | | **~2.5GB** |

## 🚀 完整训练流程

```bash
cd /root/code/layoutlmft

# 1. 训练 LayoutLMv2 版面识别模型
./train_hrdoc.sh local

# 2. 提取行级特征
python examples/extract_line_features.py

# 3. 训练关系分类器（二选一或都训练）
python examples/train_relation_classifier.py       # 二分类
python examples/train_multiclass_relation.py       # 多分类
```

## 📦 云服务器部署注意事项

部署到云服务器时需要修改：

1. **模型路径**：在 `configs/cloud_config.json` 中
   ```json
   "local_model_path": "/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/..."
   ```

2. **Python 环境**：在 `train_hrdoc.sh` 第 22 行
   ```bash
   PYTHON=/path/to/your/python
   ```

3. **数据集路径**：在 `layoutlmft/data/datasets/hrdoc.py` 第 91 行
   ```python
   data_dir = "/path/to/your/data/hrdoc_funsd_format"
   ```

4. **复制文件**：
   - 项目代码：`/root/code/layoutlmft/`
   - 预训练模型：整个 hub 目录（包含 blobs 和 snapshots）
   - 数据集：`data/hrdoc_funsd_format/`

## 🧹 清理旧输出（可选）

如果已将数据迁移到 E 盘，可以清理系统盘旧数据：

```bash
# 检查系统盘占用
du -sh /root/code/layoutlmft/output/*

# 确认 E 盘数据完整后，可以删除系统盘旧数据
# rm -rf /root/code/layoutlmft/output/hrdoc_quick
# rm -rf /root/code/layoutlmft/output/hrdoc_test
```

---

**最后更新**：2025-11-14
