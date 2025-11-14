# Scripts 目录

存放项目正式使用的训练脚本。

## 📁 脚本列表

### 1. train_hrdoc.sh
LayoutLMv2 版面识别模型训练脚本（支持多环境配置）

**用法**：
```bash
cd /path/to/layoutlmft

# 快速测试（50步，几分钟）
./scripts/train_hrdoc.sh quick

# 本地测试（500步）
./scripts/train_hrdoc.sh local

# 云服务器完整训练（30000步）
./scripts/train_hrdoc.sh cloud

# 自动检测环境
./scripts/train_hrdoc.sh auto
```

**环境变量**：
- `PYTHON`: 指定 Python 解释器路径（默认：`python`）
- `LAYOUTLMFT_OUTPUT_DIR`: 自定义训练输出根目录

**配置文件**：
- `configs/quick_config.json`: 快速测试配置
- `configs/local_config.json`: 本地测试配置
- `configs/cloud_config.json`: 云服务器配置

---

### 2. train_hrdoc_official.sh
HRDoc 论文对齐训练脚本（严格按照论文参数）

**用法**：
```bash
cd /path/to/layoutlmft

# HRDoc-Simple 完整训练（30000步）
./scripts/train_hrdoc_official.sh simple

# HRDoc-Hard 完整训练（40000步）
./scripts/train_hrdoc_official.sh hard
```

**参数说明**：
- `simple`: HRDoc-Simple 数据集，30000步，~4.5小时
- `hard`: HRDoc-Hard 数据集，40000步，~6小时

---

## 🔧 环境要求

### Python 环境
```bash
# 方式1：使用 conda
conda activate layoutlmv2

# 方式2：指定 Python 路径
export PYTHON=/path/to/python
```

### 依赖包
```bash
cd layoutlmft
pip install -r requirements.txt
```

### GPU 要求
- **快速测试/本地**: 4GB+ 显存
- **完整训练**: 16GB+ 显存（推荐24GB）

---

## 📝 注意事项

1. **脚本路径**：脚本可以从项目根目录或 scripts 目录运行
   ```bash
   # 从项目根目录
   cd /path/to/layoutlmft
   ./scripts/train_hrdoc.sh quick

   # 从 scripts 目录
   cd /path/to/layoutlmft/scripts
   ./train_hrdoc.sh quick
   ```

2. **跨环境部署**：脚本使用相对路径，无硬编码路径，可直接在云服务器使用

3. **权限问题**：如果遇到 `Permission denied`，执行：
   ```bash
   chmod +x scripts/*.sh
   ```

---

## 📚 相关文档

- [部署指南](../DEPLOYMENT_GUIDE.md)
- [训练输出配置](../TRAINING_OUTPUT_CONFIG.md)
- [快速开始](../QUICK_START.md)
