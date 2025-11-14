# HRDoc + LayoutLMv2 训练配置信息

## 环境路径
- WSL 中访问 Windows E 盘：`/mnt/e/`
- HRDoc 示例数据：`/mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS`
- HRDoc 源码：`/root/code/layoutlmft/HRDoc/`
- LayoutLMft 代码：`/root/code/layoutlmft/`
- 本地模型缓存：`/mnt/e/models/HuggingFace/`

## HRDoc 数据格式

### JSON 标注格式
每篇论文一个 JSON 文件，格式为列表，每个元素是一行文本标注：

```json
{
  "text": "文本内容",
  "box": [x1, y1, x2, y2],        // 边界框坐标
  "class": "原始类别",             // 见下方类别映射
  "page": 0,                       // 页码（从0开始）
  "is_meta": true/false,           // 是否是元数据（标题、作者等）
  "line_id": 0,                    // 行ID
  "parent_id": -1,                 // 父节点ID（用于层级结构）
  "relation": "meta/content"       // 关系类型
}
```

### 示例数据统计
- **位置**：`/mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS/`
- **论文数量**：6篇
- **总页数**：58页
- **数据组织**：
  - 每篇论文一个目录（如 `ACL_2020.acl-main.1/`）
  - 目录内有多张 JPG 图片（每页一张，命名如 `ACL_2020.acl-main.1_0.jpg`）
  - 一个同名 JSON 标注文件（如 `ACL_2020.acl-main.1.json`）

### 类别体系

#### 原始类别（15种）
```
'author', 'affili', 'tabcap', 'figcap', 'title', 'opara',
'fstline', 'foot', 'fig', 'sec2', 'fnote', 'tab', 'mail',
'para', 'sec1'
```

#### 映射后类别（13种）
使用 `utils/utils.py` 中的 `trans_class()` 函数映射：

```python
class2class = {
    "title": "title",        # 标题
    "author": "author",      # 作者
    "mail": "mail",          # 邮箱
    "affili": "affili",      # 机构
    "sec1": "section",       # 一级标题 → section
    "sec2": "section",       # 二级标题 → section
    "sec3": "section",       # 三级标题 → section
    "fstline": "fstline",    # 段落首行
    "para": "paraline",      # 段落行
    "tab": "table",          # 表格
    "fig": "figure",         # 图片
    "tabcap": "caption",     # 表格标题 → caption
    "figcap": "caption",     # 图片标题 → caption
    "equ": "equation",       # 公式
    "foot": "footer",        # 页脚
    "header": "header",      # 页眉
    "fnote": "footnote",     # 脚注
}
```

#### trans_class 函数逻辑
```python
def trans_class(all_pg_lines, unit):
    """
    如果 class 不是 'opara'（其他段落），直接查表映射
    如果是 'opara'，递归查找父节点，使用父节点的类别
    """
    if unit["class"] != "opara":
        return class2class[unit["class"]]
    else:
        parent_cl = all_pg_lines[unit['parent_id']]
        while parent_cl["class"] == 'opara':
            parent_cl = all_pg_lines[parent_cl['parent_id']]
        return class2class[parent_cl["class"]]
```

## FUNSD 数据格式（目标格式）

layoutlmft 使用的 FUNSD 格式：
```json
{
  "text": "文本内容",
  "box": [x1, y1, x2, y2],
  "label": "类别",
  "words": [
    { "box": [x1, y1, x2, y2], "text": "word1" },
    { "box": [x1, y1, x2, y2], "text": "word2" }
  ],
  "linking": [[0, 1]],  // 实体间关系（可为空）
  "id": 0
}
```

## 训练配置参考

### 论文中的配置
- **Batch size**: 3 (page-level)
- **训练步数**：
  - HRDS (Simple): 30,000 steps
  - HRDH (Hard): 40,000 steps
- **模型**: LayoutLMv2

### 当前成功运行的配置
```bash
export HF_HOME=/mnt/e/models/HuggingFace

python /root/code/layoutlmft/examples/run_funsd.py \
  --model_name_or_path /mnt/e/models/HuggingFace/hub/models--microsoft--layoutlm-base-uncased/snapshots/30e3cdd39c11f09757b0fcf7598533d05052acef \
  --output_dir ./output \
  --do_train \
  --max_steps 10 \
  --overwrite_output_dir
```

### 已解决的问题
1. ✅ **变量名错误**：修改了 `run_funsd.py:150-159` 行，`args` → `model_args`
2. ✅ **模型格式**：将 `model.safetensors` 转换为 `pytorch_model.bin`
3. ✅ **网络问题**：使用本地模型路径避免下载

## 下一步工作

### ✅ 已完成
1. **HRDoc → FUNSD 格式转换脚本**
   - ✅ 脚本路径：`/root/code/layoutlmft/hrdoc2funsd.py`
   - ✅ 成功转换示例数据（6篇论文，58页）
   - ✅ 输出目录：`/root/code/layoutlmft/data/hrdoc_funsd_format/`
   - ✅ 生成了 14 个类别的 labels.txt

### 转换结果统计
```
示例数据转换结果:
- 论文数: 6篇
- 总页数: 58页
- 标注文件: 58个 JSON
- 图片文件: 58张 JPG
- 类别数: 14个

类别列表:
affili, author, caption, equation, figure, footer,
footnote, fstline, mail, opara, paraline, section,
table, title
```

### 待完成任务

#### 1. 获取完整数据集
- [ ] **下载完整 HRDS 数据集**
  - 位置：HRDoc README 中的 Google Drive 链接
  - 需要：完整的训练集、验证集、测试集

- [ ] **下载完整 HRDH 数据集**（可选，用于 Hard 任务）
  - 位置：同上

#### 2. 转换完整数据集
```bash
# 修改 hrdoc2funsd.py 中的路径，分别转换 train/val/test
python hrdoc2funsd.py
```

#### 3. 确认模型选择
- [ ] **LayoutLMv2** (推荐) - 论文中使用的模型
  - HuggingFace: `microsoft/layoutlmv2-base-uncased`
  - 需要检查本地是否有：`/mnt/e/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/`

- [x] **LayoutLM** (已验证可用) - 我们前面测试用的
  - 路径：`/mnt/e/models/HuggingFace/hub/models--microsoft--layoutlm-base-uncased/`

#### 4. 训练命令（准备好后使用）
```bash
export HF_HOME=/mnt/e/models/HuggingFace

# 使用 LayoutLMv2（推荐）
python /root/code/layoutlmft/examples/run_funsd.py \
  --model_name_or_path /mnt/e/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/XXX \
  --output_dir ./output/hrdoc_layoutlmv2 \
  --data_dir /root/code/layoutlmft/data/hrdoc_funsd_format \
  --do_train --do_eval \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 3 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --max_steps 30000 \
  --logging_steps 100 \
  --save_steps 1000 \
  --overwrite_output_dir
```

### 待确认信息
- [ ] 是否有完整 HRDS/HRDH 数据集的访问权限？
- [ ] 本地是否有 LayoutLMv2 模型？（需要检查 /mnt/e/models/HuggingFace/）
- [ ] 训练资源：GPU 显存、预计训练时间
