# coding=utf-8
"""
comp_hrdoc 标签定义 - 统一数据加载和推理的标签映射

数据加载流程：
  hrds/hrdh: 原始标签 → trans_class → 标准标签 → LABEL2ID → ID
  tender:    标准标签 → LABEL2ID → ID

推理显示：
  ID → ID2LABEL → 标准标签
"""

from layoutlmft.data.labels import (
    # 标签列表和数量
    LABEL_LIST,
    NUM_LABELS,
    # 标签转换（sec1/sec2/sec3 → section）
    trans_class,
    # ID ↔ 标签映射
    LABEL2ID,
    ID2LABEL,
    id2label,
    label2id,
    # 便捷函数
    get_label_list,
    get_num_labels,
    get_id2label,
    get_label2id,
    # Meta 类别
    META_CLASSES,
    META_CLASS_IDS,
    is_meta_class,
    is_valid_label,
)

# Section 常量
SECTION_LABEL = "section"
SECTION_LABEL_ID = LABEL2ID[SECTION_LABEL]  # 4
