# coding=utf-8
"""
HRDoc 标签定义 - 统一入口

训练/推理/评估共用此文件，后续添加或修改标签只需改这一个地方。

论文定义的 14 个语义类别（Line 级别标注）：
{Title, Author, Mail, Affiliation, Section, First-Line, Para-Line,
 Equation, Table, Figure, Caption, Page-Footer, Page-Header, Footnote}

参考：
- 论文：https://arxiv.org/abs/2303.13839
- 评估脚本：HRDoc/utils/classify_eval.py
"""

# ============================================================
# 论文 14 类标签（按 class2id_dict 顺序）
# ============================================================
LABEL_LIST = [
    "title",      # 0: 标题
    "author",     # 1: 作者
    "mail",       # 2: 邮箱
    "affili",     # 3: 单位/机构
    "section",    # 4: 章节标题
    "fstline",    # 5: 段落首行
    "paraline",   # 6: 段落后续行
    "table",      # 7: 表格
    "figure",     # 8: 图片
    "caption",    # 9: 图表标题
    "equation",   # 10: 公式
    "footer",     # 11: 页脚
    "header",     # 12: 页眉
    "footnote",   # 13: 脚注
]

# 标签数量
NUM_LABELS = len(LABEL_LIST)  # 14

# ============================================================
# ID <-> Label 映射（模型输出数字 <-> 标签名）
# ============================================================
def id2label(label_id: int) -> str:
    """模型输出的数字 -> 标签名"""
    if 0 <= label_id < NUM_LABELS:
        return LABEL_LIST[label_id]
    return "unknown"

def label2id(label: str) -> int:
    """标签名 -> 数字ID"""
    label = label.lower()
    try:
        return LABEL_LIST.index(label)
    except ValueError:
        return -1

# 字典形式（供模型 config 使用）
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

# ============================================================
# 数据集细粒度标签 -> 论文 14 类
# ============================================================
# 数据集中可能有更细粒度的标签，需要映射到论文的 14 类
_RAW_TO_STANDARD = {
    # 章节：sec1/sec2/sec3/sec4/secx -> section
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "sec4": "section",
    "secx": "section",
    # 段落：para/opara -> paraline
    "para": "paraline",
    "opara": "paraline",
    # 表格/图片
    "tab": "table",
    "fig": "figure",
    # 图表标题：figcap/tabcap -> caption
    "figcap": "caption",
    "tabcap": "caption",
    # 公式：equ/alg -> equation
    "equ": "equation",
    "alg": "equation",
    # 脚注/页脚
    "fnote": "footnote",
    "foot": "footer",
}


def trans_class(raw_label: str, all_lines=None, unit=None) -> str:
    """
    将数据集的原始标签转换为论文 14 类。

    Args:
        raw_label: 原始标签（如 sec1, tab, figcap 等）
        all_lines: 页面所有行数据（用于 opara 特殊处理）
        unit: 当前行数据（用于 opara 特殊处理）

    Returns:
        论文 14 类标签之一
    """
    raw_label = raw_label.lower().strip()

    # opara 特殊处理：根据 parent_id 查找父节点类别
    if raw_label == "opara" and all_lines is not None and unit is not None:
        parent_id = unit.get("parent_id", -1)
        if 0 <= parent_id < len(all_lines):
            parent_unit = all_lines[parent_id]
            parent_label = parent_unit.get("class", parent_unit.get("label", "paraline")).lower()
            # 递归查找非 opara 的父节点
            while parent_label == "opara" and 0 <= parent_id < len(all_lines):
                parent_id = parent_unit.get("parent_id", -1)
                if parent_id < 0 or parent_id >= len(all_lines):
                    break
                parent_unit = all_lines[parent_id]
                parent_label = parent_unit.get("class", parent_unit.get("label", "paraline")).lower()
            return _RAW_TO_STANDARD.get(parent_label, parent_label)
        return "paraline"  # 默认

    # 标准映射
    if raw_label in _RAW_TO_STANDARD:
        return _RAW_TO_STANDARD[raw_label]

    # 已经是标准标签
    if raw_label in LABEL_LIST:
        return raw_label

    # 未知标签，返回原值（小写）
    return raw_label


def is_valid_label(label: str) -> bool:
    """检查是否是有效的论文 14 类标签"""
    return label.lower() in LABEL_LIST


# ============================================================
# 便捷函数
# ============================================================
def get_label_list():
    """获取标签列表（用于模型初始化）"""
    return LABEL_LIST.copy()

def get_num_labels():
    """获取标签数量"""
    return NUM_LABELS

def get_id2label():
    """获取 id->label 字典（用于模型 config）"""
    return ID2LABEL.copy()

def get_label2id():
    """获取 label->id 字典（用于模型 config）"""
    return LABEL2ID.copy()
