# coding=utf-8

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{wang2021hrdoc,
  title={From layout to Document: A semantic document structure analysis system for born-digital documents},
  author={Wang, Ziliang and others},
  journal={arXiv preprint arXiv:2104.08516},
  year={2021}
}
"""

_DESCRIPTION = """\
HRDoc dataset for hierarchical document structure analysis
"""

# HRDoc 论文定义的 14 个类别（与评估脚本 class2id_dict 一致）
# 参考：HRDoc/utils/classify_eval.py
_LABELS = [
    "O",
    "B-TITLE", "I-TITLE",           # 0: title
    "B-AUTHOR", "I-AUTHOR",         # 1: author
    "B-MAIL", "I-MAIL",             # 2: mail
    "B-AFFILI", "I-AFFILI",         # 3: affili
    "B-SECTION", "I-SECTION",       # 4: section (合并 sec1/sec2/sec3/sec4/secx)
    "B-FSTLINE", "I-FSTLINE",       # 5: fstline
    "B-PARALINE", "I-PARALINE",     # 6: paraline (合并 para/opara)
    "B-TABLE", "I-TABLE",           # 7: table (原 tab)
    "B-FIGURE", "I-FIGURE",         # 8: figure (原 fig)
    "B-CAPTION", "I-CAPTION",       # 9: caption (合并 figcap/tabcap)
    "B-EQUATION", "I-EQUATION",     # 10: equation (合并 equ/alg)
    "B-FOOTER", "I-FOOTER",         # 11: footer (原 foot)
    "B-HEADER", "I-HEADER",         # 12: header
    "B-FOOTNOTE", "I-FOOTNOTE",     # 13: footnote (原 fnote)
]

# 细粒度标签到论文14类的映射（来自 HRDoc/utils/utils.py 的 class2class）
_CLASS2CLASS = {
    "title": "TITLE",
    "author": "AUTHOR",
    "mail": "MAIL",
    "affili": "AFFILI",
    "sec1": "SECTION",
    "sec2": "SECTION",
    "sec3": "SECTION",
    "sec4": "SECTION",
    "secx": "SECTION",
    "fstline": "FSTLINE",
    "para": "PARALINE",
    "opara": "PARALINE",
    "tab": "TABLE",
    "fig": "FIGURE",
    "tabcap": "CAPTION",
    "figcap": "CAPTION",
    "equ": "EQUATION",
    "alg": "EQUATION",
    "foot": "FOOTER",
    "header": "HEADER",
    "fnote": "FOOTNOTE",
    # 旧版标签（已经是论文类别，直接映射为大写）
    "section": "SECTION",
    "paraline": "PARALINE",
    "table": "TABLE",
    "figure": "FIGURE",
    "caption": "CAPTION",
    "equation": "EQUATION",
    "footer": "FOOTER",
    "footnote": "FOOTNOTE",
}


def trans_class(all_pg_lines, unit):
    """
    将细粒度标签转换为论文14类。

    特殊处理 opara：根据 parent_id 递归查找父节点的类别。
    参考：HRDoc/utils/utils.py 的 trans_class
    """
    class_name = unit.get("class", unit.get("label", "O")).lower()

    if class_name != "opara":
        return _CLASS2CLASS.get(class_name, class_name.upper())
    else:
        # opara 需要根据 parent_id 查找父节点的类别
        parent_id = unit.get("parent_id", -1)
        if parent_id < 0 or parent_id >= len(all_pg_lines):
            return "PARALINE"  # 找不到父节点，默认为 PARALINE

        parent_unit = all_pg_lines[parent_id]
        parent_class = parent_unit.get("class", parent_unit.get("label", "O")).lower()

        # 递归查找，直到找到非 opara 的父节点
        while parent_class == "opara":
            parent_id = parent_unit.get("parent_id", -1)
            if parent_id < 0 or parent_id >= len(all_pg_lines):
                return "PARALINE"
            parent_unit = all_pg_lines[parent_id]
            parent_class = parent_unit.get("class", parent_unit.get("label", "O")).lower()

        return _CLASS2CLASS.get(parent_class, parent_class.upper())


class HRDocConfig(datasets.BuilderConfig):
    """BuilderConfig for HRDoc"""

    def __init__(self, **kwargs):
        """BuilderConfig for HRDoc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HRDocConfig, self).__init__(**kwargs)


class HRDoc(datasets.GeneratorBasedBuilder):
    """HRDoc dataset."""

    BUILDER_CONFIGS = [
        HRDocConfig(name="hrdoc", version=datasets.Version("1.0.0"), description="HRDoc dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(names=_LABELS)
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    # 新增：层级关系信息
                    "line_ids": datasets.Sequence(datasets.Value("int64")),  # 每个token对应的line_id
                    "line_parent_ids": datasets.Sequence(datasets.Value("int64")),  # 每个line的parent_id
                    "line_relations": datasets.Sequence(datasets.Value("string")),  # 每个line的relation
                    # 新增：文档名称和页码（用于推理时区分不同文档）
                    "document_name": datasets.Value("string"),  # 文档名称（不含扩展名）
                    "page_number": datasets.Value("int64"),     # 页码
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # 根据环境选择数据路径
        # 优先使用环境变量，否则使用默认路径
        import socket
        hostname = socket.gethostname()

        # 优先使用环境变量
        data_dir = os.getenv("HRDOC_DATA_DIR", None)

        # 如果没有设置环境变量，根据 hostname 自动检测
        if data_dir is None:
            # 本机路径（默认使用 E 盘的 HRDS 数据）
            if hostname == "mi" or os.path.exists("/root/code/layoutlmft"):
                data_dir = "/mnt/e/models/data/Section/HRDS"
            # 云服务器路径
            elif os.path.exists("/home/linux/code/layoutlmft"):
                data_dir = "/home/linux/code/layoutlmft/data/hrdoc_funsd_format"
            # 默认路径
            else:
                data_dir = "/root/code/layoutlmft/data/hrdoc_funsd_format"

        splits = []
        # 训练集
        train_path = os.path.join(data_dir, "train")
        if os.path.exists(train_path):
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": train_path}
                )
            )

        # 验证集（如果存在）
        val_path = os.path.join(data_dir, "val")
        if os.path.exists(val_path):
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": val_path}
                )
            )

        # 测试集（如果存在）
        test_path = os.path.join(data_dir, "test")
        if os.path.exists(test_path):
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": test_path}
                )
            )

        return splits

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)

        # 支持两种目录结构：
        # 1. FUNSD格式: train/annotations/, train/images/
        # 2. HRDS格式: train/*.json, images/*.png (所有图片在根目录的images下)
        if os.path.exists(os.path.join(filepath, "annotations")):
            ann_dir = os.path.join(filepath, "annotations")
            img_dir = os.path.join(filepath, "images")
        else:
            # HRDS格式：JSON 直接在 train/ 下，images 在上层
            ann_dir = filepath
            img_dir = os.path.join(os.path.dirname(filepath), "images")

        guid = 0
        for file in sorted(os.listdir(ann_dir)):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

            # 提取文档名称（不含扩展名）
            document_name = os.path.splitext(file)[0]

            # 支持两种数据格式：
            # 1. FUNSD格式：{"form": [...]} - 每个文件是一页
            # 2. HRDS格式：[...] 带 page 字段 - 一个文件多页
            if isinstance(data, dict) and "form" in data:
                # FUNSD格式：单页
                pages_data = {0: data["form"]}
            elif isinstance(data, list):
                # HRDS格式：多页，按 page 字段分组
                pages_data = {}
                for item in data:
                    page_num = item.get("page", 0)
                    if page_num not in pages_data:
                        pages_data[page_num] = []
                    pages_data[page_num].append(item)
            else:
                logger.warning(f"Unknown data format in {file_path}")
                continue

            # 处理每一页
            for page_num in sorted(pages_data.keys()):
                tokens = []
                bboxes = []
                ner_tags = []
                line_ids = []
                line_parent_ids = []
                line_relations = []

                form_data = pages_data[page_num]

                # 确定图片路径
                # FUNSD格式：train/images/xxx_0.png
                # HRDS格式：images/xxx/xxx_0.jpg
                base_name = file.replace('.json', '')

                # 先尝试 FUNSD 格式
                image_path = os.path.join(img_dir, f"{base_name}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(img_dir, f"{base_name}.jpg")

                # 再尝试 HRDS 格式
                if not os.path.exists(image_path):
                    # HRDS: images/{doc_name}/{doc_name}_{page}.jpg
                    hrds_img_path = os.path.join(img_dir, base_name, f"{base_name}_{page_num}.jpg")
                    if os.path.exists(hrds_img_path):
                        image_path = hrds_img_path
                    else:
                        # 尝试 png
                        hrds_img_path = os.path.join(img_dir, base_name, f"{base_name}_{page_num}.png")
                        if os.path.exists(hrds_img_path):
                            image_path = hrds_img_path

                if not os.path.exists(image_path):
                    logger.warning(f"Image not found for {file} page {page_num}, skipping...")
                    continue

                image, size = load_image(image_path)

                for line_idx, item in enumerate(form_data):
                    # 支持两种字段名：label（FUNSD）或 class（HRDS）
                    raw_label = item.get("label") or item.get("class", "O")

                    # 使用 trans_class 将细粒度标签转换为论文14类
                    # 传入 form_data 用于 opara 的 parent_id 查找
                    label = trans_class(form_data, item)

                    # 支持两种格式：
                    # FUNSD格式：{"words": [{...}]}
                    # HRDS格式：{"text": "...", "box": [...]}
                    if "words" in item:
                        words = item["words"]
                    else:
                        # HRDS格式：单行文本
                        words = [{
                            "text": item["text"],
                            "box": item["box"]
                        }]
                    words = [w for w in words if w["text"].strip() != ""]
                    if len(words) == 0:
                        continue

                    # label 已经是大写了（由 trans_class 返回）

                    # 获取层级关系信息
                    # HRDS格式有 line_id 字段，FUNSD格式用 id 字段
                    item_line_id = item.get("line_id", item.get("id", line_idx))
                    parent_id = item.get("parent_id", -1)
                    relation = item.get("relation", "none")

                    # 记录当前line的元数据
                    line_parent_ids.append(parent_id)
                    line_relations.append(relation)

                    # 处理第一个词（B-）
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label)
                    bboxes.append(normalize_bbox(words[0]["box"], size))
                    line_ids.append(item_line_id)  # 使用实际的line_id

                    # 处理后续词（I-）
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label)
                        bboxes.append(normalize_bbox(w["box"], size))
                        line_ids.append(item_line_id)  # 使用实际的line_id

                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "ner_tags": ner_tags,
                    "image": image,
                    "line_ids": line_ids,
                    "line_parent_ids": line_parent_ids,
                    "line_relations": line_relations,
                    "document_name": document_name,  # 文档名称
                    "page_number": page_num,         # 页码
                }
                guid += 1
