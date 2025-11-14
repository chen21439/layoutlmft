# 部署与运行指南

本项目已优化为可跨环境部署，支持本地和云服务器运行。

## 🚀 快速开始

### 本地 WSL 环境

```bash
cd /root/code/layoutlmft

# 激活 Python 环境
conda activate layoutlmv2

# 快速测试（50步，几分钟）
./train_hrdoc.sh quick

# 本地测试（500步）
./train_hrdoc.sh local
```

### 云服务器环境

```bash
cd /path/to/layoutlmft

# 方式1：使用系统 Python（推荐）
./train_hrdoc.sh cloud

# 方式2：指定 Python 路径
export PYTHON=/path/to/your/python
./train_hrdoc.sh cloud
```

## 🔧 环境变量配置

脚本支持通过环境变量自定义配置：

```bash
# 指定 Python 解释器（可选）
export PYTHON=/usr/bin/python3

# 指定训练输出根目录（可选）
export LAYOUTLMFT_OUTPUT_DIR=/data/models/train_data

# 指定特征文件目录（可选）
export LAYOUTLMFT_FEATURES_DIR=/data/features

# 指定模型路径（可选，用于特征提取）
export LAYOUTLMFT_MODEL_PATH=/data/models/hrdoc_test

# 运行训练
./train_hrdoc.sh local
```

## 📦 云服务器部署清单

### 1. 需要复制的文件

```bash
# 项目代码
layoutlmft/
├── examples/              # Python 脚本
├── layoutlmft/            # 核心代码
├── configs/               # 配置文件
├── data/                  # 数据集
├── train_hrdoc.sh         # 训练脚本
└── requirements.txt       # 依赖

# 预训练模型（完整目录结构）
models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/
├── blobs/                 # 实际文件
└── snapshots/             # 符号链接
    └── ae6f4350.../       # 具体 hash
        ├── config.json
        ├── pytorch_model.bin
        └── vocab.txt
```

### 2. 修改配置文件

**configs/cloud_config.json**：
```json
{
  "local_model_path": "/your/cloud/path/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350..."
}
```

### 3. 修改数据集路径

**layoutlmft/data/datasets/hrdoc.py** 第 91 行：
```python
# 改为云服务器路径或使用相对路径
data_dir = "/your/cloud/data/path"
# 或
data_dir = os.path.join(os.path.dirname(__file__), "../../../data/hrdoc_funsd_format")
```

### 4. 安装依赖

```bash
cd layoutlmft
pip install -r requirements.txt

# 或使用小依赖包（如果网络不好）
pip install -r requirements.small.txt
```

## 📍 路径说明

### 本地 WSL 环境

| 资源 | 路径 |
|------|------|
| 项目代码 | `/root/code/layoutlmft/` |
| 预训练模型 | `/mnt/e/models/HuggingFace/hub/...` |
| 训练输出 | `/mnt/e/models/train_data/layoutlmft/` |
| 数据集 | `./data/hrdoc_funsd_format/` |

### 云服务器环境（需自定义）

| 资源 | 默认路径 | 说明 |
|------|---------|------|
| 项目代码 | `./layoutlmft/` | 当前目录 |
| 预训练模型 | `/models/HuggingFace/hub/...` | 需修改配置 |
| 训练输出 | `./output/hrdoc_simple_full/` | 相对路径 |
| 数据集 | 需配置 | 修改 hrdoc.py |

## 🎯 训练流程

### 完整流程（三个模型）

```bash
# 1. LayoutLMv2 版面识别模型
./train_hrdoc.sh cloud          # 云服务器完整训练（30000步，4-6小时）

# 2. 提取行级特征
python examples/extract_line_features.py

# 3. 训练关系分类器（二选一或都训练）
python examples/train_relation_classifier.py       # 二分类
python examples/train_multiclass_relation.py       # 多分类（4类）
```

### 快速测试流程

```bash
# 1. 快速测试 LayoutLMv2（50步，几分钟）
./train_hrdoc.sh quick

# 2. 提取特征（使用快速测试的模型）
export LAYOUTLMFT_MODEL_PATH=/mnt/e/models/train_data/layoutlmft/hrdoc_quick
python examples/extract_line_features.py

# 3. 训练关系分类器
python examples/train_multiclass_relation.py
```

## 🐛 常见问题

### 1. Permission Denied 错误

**问题**：`bash: ./train_hrdoc.sh: Permission denied`

**解决**：
```bash
chmod +x train_hrdoc.sh
./train_hrdoc.sh quick
```

### 2. Python 环境找不到

**问题**：`python: command not found`

**解决**：
```bash
# 方式1：激活 conda 环境
conda activate layoutlmv2

# 方式2：指定 Python 路径
export PYTHON=/path/to/python
./train_hrdoc.sh quick

# 方式3：使用绝对路径
export PYTHON=/root/miniforge3/envs/layoutlmv2/bin/python
./train_hrdoc.sh quick
```

### 3. 模块找不到

**问题**：`ModuleNotFoundError: No module named 'layoutlmft'`

**解决**：
```bash
# 确保在项目根目录运行
cd /path/to/layoutlmft
./train_hrdoc.sh quick

# 或手动设置 PYTHONPATH
export PYTHONPATH=/path/to/layoutlmft:$PYTHONPATH
```

### 4. 数据集路径错误

**问题**：`FileNotFoundError: [Errno 2] No such file or directory: '/root/code/layoutlmft/data/hrdoc_funsd_format'`

**解决**：
修改 `layoutlmft/data/datasets/hrdoc.py` 第 91 行为实际路径，或使用相对路径：
```python
data_dir = os.path.join(os.path.dirname(__file__), "../../../data/hrdoc_funsd_format")
```

### 5. 特征文件找不到

**问题**：`FileNotFoundError: train_line_features.pkl`

**解决**：
```bash
# 先提取特征
python examples/extract_line_features.py

# 或指定特征目录
export LAYOUTLMFT_FEATURES_DIR=/path/to/features
python examples/train_multiclass_relation.py
```

## 📊 资源需求

### 本地测试（quick）
- GPU 显存：4GB+
- 磁盘空间：2GB
- 训练时间：5-10分钟

### 完整训练（cloud）
- GPU 显存：16GB+（推荐24GB）
- 磁盘空间：5GB
- 训练时间：4-6小时（V100/A100）

## 🔗 相关文档

- 训练输出配置：[TRAINING_OUTPUT_CONFIG.md](./TRAINING_OUTPUT_CONFIG.md)
- 快速开始：[QUICK_START.md](./QUICK_START.md)
- 训练指南：[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)

---

**最后更新**：2025-11-14
