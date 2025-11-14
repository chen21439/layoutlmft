```bash
# 云服务器指定路径
python /home/linux/code/layoutlmft/data/hrdoc2funsd.py \
  --input /home/linux/code/data/HRDS \
  --output /home/linux/code/layoutlmft/data/hrdoc_funsd_format \
  --split train
```

# 数据目录说明

本目录用于存放 HRDoc 数据集及相关转换脚本。

## 目录结构

```
data/
├── README.md                   # 本说明文档
├── hrdoc2funsd.py             # HRDoc → FUNSD 格式转换脚本
├── hrdoc_funsd_format/        # 转换后的 FUNSD 格式数据（用于训练）
│   ├── train/
│   │   ├── images/           # 训练集图片
│   │   └── annotations/      # 训练集标注（JSON）
│   ├── val/                  # 验证集（可选）
│   ├── test/                 # 测试集（可选）
│   └── labels.txt            # 所有标签列表
├── hrdoc_test/               # 测试数据
└── line_features/            # 行特征数据
```

## 数据格式说明

### HRDoc 原始格式
- 每个文档对应一个 JSON 文件
- 包含文档的层级结构信息（parent_id, relation）
- 每一行包含：text, box, class, page 等字段

### FUNSD 格式（转换后）
- 按页面分割，每页一个 JSON 文件
- 每个文件包含 `form` 字段，存储实体列表
- 每个实体包含：id, text, box, label, words, parent_id, relation 等字段
- 配套的图片文件（.jpg/.png）

## 转换脚本说明

### 两个版本

项目提供两个转换脚本，分别用于不同的数据集结构：

#### 1. `hrdoc2funsd.py` - 完整数据集版本（推荐）
- **用途**: 转换完整的 HRDS 数据集（900+ 训练文件）
- **数据结构**: 支持 train/test 子目录 + 统一的 images 目录
- **默认路径**:
  - 本机: `/mnt/e/models/data/Section/HRDS`
  - 云服务器: 需要用 `--input` 指定
- **适用场景**: 正式训练、完整数据集转换

```bash
# 本机使用默认参数
python data/hrdoc2funsd.py


```

#### 2. `hrdoc2funsd_example.py` - 示例数据版本
- **用途**: 转换示例数据（6个论文，约58个页面）
- **数据结构**: JSON 在根目录，每个论文独立图片文件夹
- **默认路径**: `/mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS`
- **适用场景**: 快速测试、开发调试

```bash
# 使用示例数据
python data/hrdoc2funsd_example.py
```

### 基本用法

```bash
# 转换训练集（本机完整数据）
python data/hrdoc2funsd.py

# 转换测试集
python data/hrdoc2funsd.py --split test


```

### 参数说明

| 参数 | 简写 | 说明 | hrdoc2funsd.py 默认值 | hrdoc2funsd_example.py 默认值 |
|------|------|------|----------------------|------------------------------|
| `--input` | `-i` | HRDoc 原始数据目录 | `/mnt/e/models/data/Section/HRDS` | `/mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS` |
| `--output` | `-o` | 输出目录 | `{project_root}/data/hrdoc_funsd_format` | `{project_root}/data/hrdoc_funsd_format` |
| `--split` | `-s` | 数据集划分 | `train` | `train` |

### 查看帮助

```bash
python data/hrdoc2funsd.py --help
```



## 类别映射

脚本使用以下类别映射（来自 HRDoc 官方）：

| 原始类别 | 映射后类别 |
|---------|-----------|
| title | title |
| author | author |
| mail | mail |
| affili | affili |
| sec1/sec2/sec3 | section |
| fstline | fstline |
| para | paraline |
| tab | table |
| fig | figure |
| tabcap/figcap | caption |
| equ | equation |
| foot | footer |
| header | header |
| fnote | footnote |
| opara | 继承父节点类别 |

## 注意事项

1. **数据与训练解耦**
   - 转换脚本只需运行一次
   - 转换后的数据可以重复用于训练
   - 训练时只读取 FUNSD 格式数据，不需要原始数据

2. **路径配置**
   - `layoutlmft/data/datasets/hrdoc.py` 使用动态路径计算
   - 自动适配不同环境（本机/云服务器）
   - 无需修改代码即可在不同机器上运行

3. **数据完整性**
   - 确保每个 JSON 标注文件都有对应的图片
   - 支持的图片格式：.jpg, .png, .jpeg
   - 缺失图片会在转换时输出警告

## 相关文件

- 数据加载器：`layoutlmft/data/datasets/hrdoc.py`
- 训练脚本：`examples/run_hrdoc.py`
- 训练配置：`configs/quick_config.json`, `configs/full_config.json`
- 训练入口：`scripts/train_hrdoc.sh`

## 常见问题

### Q1: 转换脚本报错 "输入目录不存在"
**A:** 检查 `--input` 参数指定的路径是否正确，确保该目录下有 `.json` 标注文件。

### Q2: 训练时报错 "KeyError: 'train'"
**A:** 说明 `data/hrdoc_funsd_format/train/` 目录不存在或为空，需要先运行转换脚本生成数据。

### Q3: 如何添加验证集或测试集？
**A:** 使用 `--split` 参数多次运行转换脚本：
```bash
python data/hrdoc2funsd.py --input /path/to/train_data --split train
python data/hrdoc2funsd.py --input /path/to/val_data --split val
python data/hrdoc2funsd.py --input /path/to/test_data --split test
```

### Q4: 云服务器和本机路径不同怎么办？
**A:** 使用命令行参数指定实际路径，无需修改脚本代码。脚本已支持自动适配。

## 更新日志

- **2025-11-14**: 添加命令行参数支持，实现路径参数化
- **2025-11-14**: 修复 `hrdoc.py` 硬编码路径问题，改用动态计算
- **初版**: 创建 HRDoc → FUNSD 格式转换脚本
