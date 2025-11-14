# 多环境训练配置系统

自动检测运行环境并选择合适的训练参数，支持本机测试和云服务器完整训练。

## 🎯 快速开始

### 1. 自动检测环境并训练


```bash
./train_hrdoc.sh auto
```

系统会自动检测：
- GPU显存 < 20GB → `local`（本机）
- GPU显存 ≥ 20GB 或云环境标记 → `cloud`（云服务器）

### 2. 手动指定环境

```bash
# 本机快速测试（500步，~30分钟）
./train_hrdoc.sh local

# 云服务器完整训练（30000步，~4-6小时，对齐论文）
./train_hrdoc.sh cloud

# 超快速测试（50步，~5分钟）
./train_hrdoc.sh quick
```

---

## 📊 环境配置对比

| 环境 | max_steps | batch_size | 训练时长 | 用途 |
|------|-----------|------------|---------|------|
| **quick** | 50 | 2 | ~5分钟 | 代码调试 |
| **local** | 500 | 2 | ~30分钟 | 本机测试 |
| **cloud** | 30,000 | 3 | ~4-6小时 | 正式训练（论文配置） |

---

## 🔧 自定义配置

### 方式1: 修改配置文件

编辑 `configs/{env}_config.json`:

```json
{
  "max_steps": 1000,
  "per_device_train_batch_size": 4,
  "learning_rate": 5e-5,
  ...
}
```

### 方式2: 修改 `env_config.py`

编辑 `configs/env_config.py` 中的 `get_config()` 函数。

### 方式3: 创建新环境

```python
from configs.env_config import TrainingConfig

my_config = TrainingConfig(
    output_dir="./output/my_experiment",
    max_steps=2000,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
)
my_config.save_json("./configs/my_env_config.json")
```

然后运行：
```bash
./train_hrdoc.sh my_env
```

---

## 🌐 云环境标记

如果自动检测不准确，可以手动设置环境变量：

```bash
export CLOUD_ENV=1  # 标记为云环境
./train_hrdoc.sh auto
```

---

## 📁 配置文件说明

生成的配置文件位于 `./configs/` 目录：

```
configs/
├── env_config.py           # 环境检测和配置生成
├── local_config.json       # 本机配置
├── cloud_config.json       # 云服务器配置
├── quick_config.json       # 快速测试配置
└── README.md              # 本文档
```

---

## 🚀 完整训练流程

```bash
# 1. 生成配置文件（首次运行）
python configs/env_config.py

# 2. 训练版面识别模型
./train_hrdoc.sh auto

# 3. 提取行级特征
python examples/extract_line_features.py

# 4. 训练关系分类器
python examples/train_relation_classifier.py      # 二分类
# 或
python examples/train_multiclass_relation.py     # 多分类
```

---

## 📌 环境检测逻辑

```python
def detect_environment():
    gpu_memory = get_gpu_memory()  # GB
    is_cloud = check_cloud_indicators()

    if is_cloud or gpu_memory >= 20:
        return "cloud"
    else:
        return "local"
```

检测因素：
- GPU显存大小
- 环境变量（`CLOUD_ENV`, `AWS_EXECUTION_ENV` 等）
- 主机名（包含 `cloud`, `aws`, `gpu-server` 等）

---

## ⚙️ 高级用法

### 从Python代码使用

```python
from configs.env_config import get_config, EnvironmentDetector

# 打印环境信息
EnvironmentDetector.print_environment_info()

# 加载配置
config = get_config("local")
print(f"Max steps: {config.max_steps}")

# 保存配置
config.save_json("./my_config.json")
```

### 转换为命令行参数

```python
config = get_config("cloud")
args = []
for key, value in config.to_dict().items():
    args.append(f"--{key}")
    args.append(str(value))
```

---

## 🐛 故障排查

### 配置文件不存在
```bash
python configs/env_config.py  # 重新生成配置
```

### 环境检测错误
```bash
# 手动指定环境
./train_hrdoc.sh cloud
```

### 显存不足
```bash
# 使用更小的batch size
./train_hrdoc.sh local  # batch_size=2
```
