# SubTask 2: 父节点查找实现说明

## 论文方法 vs 新实现

### 论文的 SubTask 2 设计（HRDoc）

根据论文 [HRDoc](https://ar5iv.labs.arxiv.org/html/2303.13839)，SubTask 2 的关键要素：

1. **模型架构**: GRU decoder + 注意力机制
   - 顺序处理文档中的所有语义单元
   - GRU 维护隐藏状态 h_i，捕获跨页依赖
   - 注意力机制计算候选父节点的权重

2. **Soft-mask 操作**: Child-Parent Distribution Matrix (M_cp)
   - 统计训练数据中不同语义类别的父子关系分布
   - 矩阵大小: [C+1, C]（C=语义类别数，+1 for ROOT）
   - 加性平滑（pseudo-count=5）处理未见过的关系对

3. **父节点概率计算**:
   ```
   P_dom(i,j) = P̃_cls_j · M_cp · P_cls_i^T
   P_par(i,j) = softmax(α(q_i, h_j) · h_j · P_dom(i,j))
   P̂_i = argmax(P_par(i,j))
   ```

4. **训练目标**: 多分类交叉熵
   - 对每个单元 i，预测其父节点索引 ∈ {0, 1, ..., i-1}
   - 0 表示 ROOT，1到i-1表示前面的单元

### 新实现 (`train_parent_finder.py`)

**核心类**:

1. **ChildParentDistributionMatrix**
   - 统计并构建 M_cp 矩阵
   - 支持加性平滑
   - 可保存/加载

2. **ParentFinderGRU**
   - GRU decoder 进行序列建模
   - 注意力机制计算父节点分数
   - Soft-mask 操作约束合理关系
   - 因果mask（只能选择前面的单元作为父节点）

3. **ParentFinderDataset**
   - 加载页面级特征
   - 每个样本是一个文档页面

## 对比：旧实现 vs 新实现

| 方面 | `train_relation_classifier.py` (旧) | `train_parent_finder.py` (新) |
|------|-------------------------------------|-------------------------------|
| **问题类型** | 二分类：(parent, child) -> 0/1 | 多分类：child -> parent_index |
| **模型** | 简单MLP | GRU + 注意力 |
| **序列建模** | ✗ 独立处理每对 | ✓ GRU 顺序处理 |
| **域知识** | ✗ 无 | ✓ Soft-mask (M_cp) |
| **负采样** | ✓ 需要负采样 | ✗ 不需要（多分类） |
| **数据粒度** | 样本对 | 页面 |
| **符合论文** | ✗ | ✓ |

## 使用方法

### 前置条件

需要从 SubTask 1 提取行级特征和语义标签：

```bash
# 1. 训练 SubTask 1（语义分类）
./scripts/train_hrdoc.sh cloud

# 2. 提取行级特征（需要创建提取脚本）
python examples/extract_line_features.py \
    --model_path /path/to/subtask1/checkpoint \
    --output_dir /path/to/line_features
```

### 训练 SubTask 2

```bash
# 设置环境变量
export LAYOUTLMFT_FEATURES_DIR=/path/to/line_features
export LAYOUTLMFT_OUTPUT_DIR=/path/to/output

# 训练
python examples/train_parent_finder.py
```

### 参数说明

在 `train_parent_finder.py` 的 `main()` 函数中：

```python
num_epochs = 10           # 训练轮数
batch_size = 4            # 批大小（页面级）
learning_rate = 1e-4      # 学习率
num_classes = 16          # 语义类别数
use_soft_mask = True      # 是否使用 soft-mask
```

## 数据格式要求

### 输入特征文件

每个 chunk 文件包含多个页面，每个页面包含：

```python
page_data = {
    "line_features": torch.Tensor,  # [1, max_lines, 768]
    "line_mask": torch.Tensor,      # [1, max_lines]
    "line_parent_ids": List[int],   # 每个line的父节点ID
    "line_labels": List[int],       # 每个line的语义类别标签 (NEW!)
    "line_bboxes": np.ndarray       # [num_lines, 4]
}
```

**注意**: 需要添加 `line_labels`，这个需要从 SubTask 1 的预测结果中获取。

### 语义标签映射

根据 `hrdoc.py` 中的 `_LABELS` 定义：

```python
0  -> "O"
1  -> "B-AFFILI"
2  -> "I-AFFILI"
3  -> "B-ALG"
...
```

需要将 BIO 标签转换为 line 级别的语义类别（去掉 B-/I- 前缀）。

## 训练流程

```
1. 构建 M_cp 矩阵
   ├─ 扫描训练数据
   ├─ 统计 (child_label, parent_label) 频次
   ├─ 加性平滑
   └─ 归一化得到概率分布

2. 训练模型
   ├─ 加载页面特征
   ├─ GRU 顺序处理语义单元
   ├─ 计算注意力 + soft-mask
   ├─ 预测父节点索引
   └─ 交叉熵损失

3. 评估
   └─ 父节点预测准确率
```

## 输出

训练完成后，在 `output_dir` 中：

```
parent_finder/
├── child_parent_matrix.npy  # M_cp 矩阵
├── best_model.pt             # 最佳模型checkpoint
└── logs/                     # 训练日志
```

## TODO: 需要创建的脚本

1. **extract_line_features.py** - 从 SubTask 1 模型提取行级特征
   ```python
   # 功能：
   # - 加载训练好的 LayoutLMv2 模型
   # - 对每个文档页面进行推理
   # - 提取 hidden states 并聚合到 line 级别
   # - 预测每个 line 的语义标签
   # - 保存为 chunk 文件
   ```

2. **inference_parent_finding.py** - 推理脚本
   ```python
   # 功能：
   # - 加载训练好的 ParentFinderGRU 模型
   # - 对新文档预测父节点
   # - 输出树结构
   ```

## 与 SubTask 3 的衔接

SubTask 2 输出父子关系后，SubTask 3 接收：

```python
parent_pairs = [
    (parent_id, child_id),  # 从 SubTask 2 预测
    ...
]

# SubTask 3 对每个 pair 预测关系类型
relation_types = subtask3_model.predict(parent_pairs)
# -> ["contain", "connect", "equality", ...]
```

## 性能预期

根据论文，SubTask 2 的性能指标：

- **准确率**: 父节点预测准确率
- **F1-score**: 对不同语义类别的父节点预测

论文中的性能：
- 使用 soft-mask 可以显著提高性能
- GRU 比 Transformer decoder 更高效

## 论文引用

```bibtex
@article{sun2023hrdoc,
  title={HRDoc: Dataset and Baseline Method for Hierarchical Reconstruction of Document Structures},
  author={Sun, Jiefeng and others},
  journal={arXiv preprint arXiv:2303.13839},
  year={2023}
}
```
