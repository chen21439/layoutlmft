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


# ============================================================
# 关系类型定义（Stage4）
# ============================================================
# 模型输出 3 类关系（论文定义）
# ROOT 边的关系单独处理，不占用模型类别
RELATION_LIST = [
    "connect",   # 0: 连接关系（同级续行、段落连接）
    "contain",   # 1: 包含关系（父节点包含子节点，如 section 包含 paragraph）
    "equality",  # 2: 等价关系（如 figure 和 caption）
]

NUM_RELATIONS = len(RELATION_LIST)  # 3

# ID <-> Relation 映射
RELATION2ID = {rel: i for i, rel in enumerate(RELATION_LIST)}
ID2RELATION = {i: rel for i, rel in enumerate(RELATION_LIST)}

def relation2id(relation: str) -> int:
    """关系名 -> 数字ID"""
    return RELATION2ID.get(relation.lower(), -1)

def id2relation(relation_id: int) -> str:
    """数字ID -> 关系名"""
    return ID2RELATION.get(relation_id, "unknown")


# ============================================================
# Label-Pair Gating 规则（P0 约束解码）
# ============================================================
# 定义哪些 (child_label, parent_label) 组合更可能产生哪种关系
# 用于推理时的 label-pair gating

# 语义类别分组（用于简化规则）
HEADING_LABELS = {"title", "section"}  # 标题类
PARAGRAPH_LABELS = {"fstline", "paraline"}  # 段落类
FLOAT_LABELS = {"table", "figure", "equation"}  # 浮动对象
CAPTION_LABELS = {"caption"}  # 标题
META_LABELS = {"author", "mail", "affili", "footer", "header", "footnote"}  # 元信息

# Label-pair 到允许关系的映射
# 格式：{(child_group, parent_group): [allowed_relations]}
# 如果不在表中，默认允许所有关系
_LABEL_PAIR_RULES = {
    # HEADING -> HEADING: 更可能 connect（同级）或 contain（降级）
    ("heading", "heading"): ["connect", "contain"],
    # PARAGRAPH -> HEADING: 更可能 contain（挂靠到 section）
    ("paragraph", "heading"): ["contain", "connect"],
    # PARAGRAPH -> PARAGRAPH: 更可能 connect（同段落续行）
    ("paragraph", "paragraph"): ["connect"],
    # FLOAT -> HEADING: contain
    ("float", "heading"): ["contain"],
    # CAPTION -> FLOAT: equality 或 contain
    ("caption", "float"): ["equality", "contain"],
    # CAPTION -> HEADING: contain
    ("caption", "heading"): ["contain"],
    # META -> any: 通常 connect
    ("meta", "heading"): ["contain", "connect"],
    ("meta", "meta"): ["connect"],
}


def _get_label_group(label: str) -> str:
    """获取标签所属的语义分组"""
    label = label.lower()
    if label in HEADING_LABELS:
        return "heading"
    elif label in PARAGRAPH_LABELS:
        return "paragraph"
    elif label in FLOAT_LABELS:
        return "float"
    elif label in CAPTION_LABELS:
        return "caption"
    elif label in META_LABELS:
        return "meta"
    return "other"


def get_allowed_relations(child_label: str, parent_label: str) -> list:
    """
    获取给定 (child_label, parent_label) 允许的关系类型

    Args:
        child_label: 子节点的语义类别（如 "section", "paraline"）
        parent_label: 父节点的语义类别

    Returns:
        允许的关系名列表（如 ["connect", "contain"]）
    """
    child_group = _get_label_group(child_label)
    parent_group = _get_label_group(parent_label)

    key = (child_group, parent_group)
    if key in _LABEL_PAIR_RULES:
        return _LABEL_PAIR_RULES[key]

    # 默认：允许所有关系
    return RELATION_LIST.copy()


def get_allowed_relation_ids(child_label_id: int, parent_label_id: int) -> list:
    """
    获取给定 (child_label_id, parent_label_id) 允许的关系 ID 列表

    Args:
        child_label_id: 子节点的语义类别 ID
        parent_label_id: 父节点的语义类别 ID

    Returns:
        允许的关系 ID 列表（如 [0, 1]）
    """
    child_label = id2label(child_label_id)
    parent_label = id2label(parent_label_id)
    allowed = get_allowed_relations(child_label, parent_label)
    return [RELATION2ID[rel] for rel in allowed]
