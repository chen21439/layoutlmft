# coding=utf-8
"""
HRDoc Dataset for LayoutLMv2/LayoutXLM

使用论文定义的 14 个语义类别（Line 级别标注，不使用 BIO）。
标签定义统一使用 layoutlmft.data.labels 模块。
"""

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bbox
from layoutlmft.data.labels import (
    LABEL_LIST,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    trans_class,
    get_label_list,
)


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
HRDoc dataset for hierarchical document structure analysis.
Uses 14 semantic classes at line level (no BIO tagging).
"""

# 兼容旧代码：_LABELS 指向 LABEL_LIST（从 labels.py 导入）
_LABELS = LABEL_LIST


class HRDocConfig(datasets.BuilderConfig):
    """BuilderConfig for HRDoc"""

    def __init__(self, **kwargs):
        """BuilderConfig for HRDoc.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HRDocConfig, self).__init__(**kwargs)


class HRDoc(datasets.GeneratorBasedBuilder):
    """HRDoc dataset - 页级别模式（一个样本=一页，保持全局 line_id）"""

    BUILDER_CONFIGS = [
        HRDocConfig(name="hrdoc", version=datasets.Version("1.0.0"), description="HRDoc dataset (page-level)"),
        HRDocConfig(name="hrds", version=datasets.Version("1.0.0"), description="HRDoc-Simple dataset"),
        HRDocConfig(name="hrdh", version=datasets.Version("1.0.0"), description="HRDoc-Hard dataset"),
        HRDocConfig(name="tender", version=datasets.Version("1.0.0"), description="Tender document dataset"),
    ]

    def _info(self):
        # 页级别模式：一个样本 = 一页（兼容 LayoutLM 输入格式）
        # 注意：line_id 和 parent_id 保持全局索引，不重映射
        features = datasets.Features({
            "id": datasets.Value("string"),
            "document_name": datasets.Value("string"),
            "page_number": datasets.Value("int64"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
            "ner_tags": datasets.Sequence(datasets.Value("int64")),
            "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
            "line_ids": datasets.Sequence(datasets.Value("int64")),  # 全局 line_id（不重映射）
            # 该页涉及的行的 parent_id 和 relation（全局索引）
            "line_parent_ids": datasets.Sequence(datasets.Value("int64")),
            "line_relations": datasets.Sequence(datasets.Value("string")),
        })

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
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

        # 检查是否使用 covmatch split（从环境变量读取）
        split_dir = os.getenv("HRDOC_SPLIT_DIR", None)
        train_doc_ids = None
        dev_doc_ids = None

        # 快速模式：限制每个 split 生成的样本数（用于快速调试）
        # HRDOC_MAX_SAMPLES=100 限制最多生成 100 个页面
        # HRDOC_MAX_DOCS=10 限制最多处理 10 个文档（推荐，保证文档完整性）
        max_samples_str = os.getenv("HRDOC_MAX_SAMPLES", None)
        max_samples = int(max_samples_str) if max_samples_str else None
        max_docs_str = os.getenv("HRDOC_MAX_DOCS", None)
        max_docs = int(max_docs_str) if max_docs_str else None
        if max_samples:
            logger.info(f"Quick mode enabled: max_samples={max_samples}")
        if max_docs:
            logger.info(f"Quick mode enabled: max_docs={max_docs}")

        if split_dir and os.path.exists(split_dir):
            # 读取 split 文件
            train_ids_file = os.path.join(split_dir, "train_doc_ids.json")
            dev_ids_file = os.path.join(split_dir, "dev_doc_ids.json")

            if os.path.exists(train_ids_file):
                with open(train_ids_file, 'r') as f:
                    train_doc_ids = set(json.load(f))
                logger.info(f"Using covmatch split: {len(train_doc_ids)} train docs from {train_ids_file}")

            if os.path.exists(dev_ids_file):
                with open(dev_ids_file, 'r') as f:
                    dev_doc_ids = set(json.load(f))
                logger.info(f"Using covmatch split: {len(dev_doc_ids)} dev docs from {dev_ids_file}")

        splits = []
        train_path = os.path.join(data_dir, "train")

        if train_doc_ids is not None and dev_doc_ids is not None:
            # 使用 covmatch split：从 train 目录中按 doc_ids 分割
            if os.path.exists(train_path):
                # Train split
                splits.append(
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"filepath": train_path, "doc_ids": train_doc_ids, "max_samples": max_samples, "max_docs": max_docs}
                    )
                )
                # Validation split (from dev_doc_ids)
                splits.append(
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"filepath": train_path, "doc_ids": dev_doc_ids, "max_samples": max_samples, "max_docs": max_docs}
                    )
                )
        else:
            # 传统模式：使用目录结构
            # 训练集
            if os.path.exists(train_path):
                splits.append(
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"filepath": train_path, "doc_ids": None, "max_samples": max_samples, "max_docs": max_docs}
                    )
                )

            # 验证集（如果存在）
            val_path = os.path.join(data_dir, "val")
            if os.path.exists(val_path):
                splits.append(
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={"filepath": val_path, "doc_ids": None, "max_samples": max_samples, "max_docs": max_docs}
                    )
                )

        # 测试集（始终使用原始 test 目录）
        test_path = os.path.join(data_dir, "test")
        if os.path.exists(test_path):
            splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": test_path, "doc_ids": None, "max_samples": max_samples, "max_docs": max_docs}
                )
            )

        return splits

    def _generate_examples(self, filepath, doc_ids=None, max_samples=None, max_docs=None):
        """
        页级别生成器：一个样本 = 一页（保持全局 line_id）

        输出格式:
        {
            "id": "0",
            "document_name": "doc1",
            "page_number": 0,
            "tokens": [...],
            "bboxes": [...],
            "ner_tags": [...],
            "image": ...,
            "line_ids": [...],       # 全局 line_id（不重映射）
            "line_parent_ids": [...], # 该页行的 parent_id（全局索引）
            "line_relations": [...],  # 该页行的 relation
        }
        """
        logger.info("⏳ Generating examples from = %s (page-level, global line_id)", filepath)
        if doc_ids is not None:
            logger.info(f"  Filtering to {len(doc_ids)} docs")
        if max_samples is not None:
            logger.info(f"  Limiting to {max_samples} samples (quick mode)")
        if max_docs is not None:
            logger.info(f"  Limiting to {max_docs} docs (quick mode)")

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
        doc_count = 0
        for file in sorted(os.listdir(ann_dir)):
            if not file.endswith('.json'):
                continue

            # 快速模式：达到 max_docs 后停止
            if max_docs is not None and doc_count >= max_docs:
                logger.info(f"  Reached max_docs={max_docs}, stopping generation")
                return

            # 提取文档名称（不含扩展名）
            document_name = os.path.splitext(file)[0]

            # 如果指定了 doc_ids，只处理在列表中的文档
            if doc_ids is not None and document_name not in doc_ids:
                continue

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

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

            base_name = file.replace('.json', '')
            doc_has_valid_page = False

            # ==================== 逐页生成样本 ====================
            for page_num in sorted(pages_data.keys()):
                form_data = pages_data[page_num]

                # ==================== 确定图片路径 ====================
                # 先尝试 FUNSD 格式
                image_path = os.path.join(img_dir, f"{base_name}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(img_dir, f"{base_name}.jpg")

                # 再尝试 HRDS 格式: images/{doc_name}/{doc_name}_{page}.jpg
                if not os.path.exists(image_path):
                    hrds_img_path = os.path.join(img_dir, base_name, f"{base_name}_{page_num}.jpg")
                    if os.path.exists(hrds_img_path):
                        image_path = hrds_img_path
                    else:
                        hrds_img_path = os.path.join(img_dir, base_name, f"{base_name}_{page_num}.png")
                        if os.path.exists(hrds_img_path):
                            image_path = hrds_img_path

                # 再尝试 HRDH 格式: images/{doc_name}/{page}.png
                if not os.path.exists(image_path):
                    hrdh_img_path = os.path.join(img_dir, base_name, f"{page_num}.png")
                    if os.path.exists(hrdh_img_path):
                        image_path = hrdh_img_path
                    else:
                        hrdh_img_path = os.path.join(img_dir, base_name, f"{page_num}.jpg")
                        if os.path.exists(hrdh_img_path):
                            image_path = hrdh_img_path

                if not os.path.exists(image_path):
                    logger.warning(f"Image not found for {file} page {page_num}, skipping page...")
                    continue

                image, size = load_image(image_path)

                # ==================== 处理当前页的数据 ====================
                page_tokens = []
                page_bboxes = []
                page_ner_tags = []
                page_line_ids = []
                page_parent_ids = []  # 该页行的 parent_id
                page_relations = []   # 该页行的 relation

                for line_idx, item in enumerate(form_data):
                    # 使用 trans_class 将细粒度标签转换为论文14类
                    label = trans_class(
                        item.get("class", item.get("label", "paraline")),
                        all_lines=form_data,
                        unit=item
                    )
                    # 转换为 int（因为 features 中使用 int64）
                    if isinstance(label, str):
                        label = LABEL2ID.get(label, 0)

                    # 数据格式处理
                    if "words" in item:
                        words = item["words"]
                    else:
                        words = [{
                            "text": item["text"],
                            "box": item["box"]
                        }]
                    words = [w for w in words if w["text"].strip() != ""]
                    if len(words) == 0:
                        continue

                    # 获取全局 line_id（不重映射）
                    item_line_id = item.get("line_id", item.get("id", line_idx))

                    # 获取 parent_id（全局索引，不重映射）
                    raw_parent_id = item.get("parent_id", -1)
                    if raw_parent_id == "" or raw_parent_id is None:
                        parent_id = -1
                    else:
                        try:
                            parent_id = int(raw_parent_id)
                        except (ValueError, TypeError):
                            parent_id = -1

                    # 获取 relation
                    relation = item.get("relation", "none")
                    if relation == "" or relation is None:
                        relation = "none"

                    # 记录该页的 parent_id 和 relation（每行一个）
                    page_parent_ids.append(parent_id)
                    page_relations.append(relation)

                    # 构建当前页的 tokens 列表
                    for w in words:
                        page_tokens.append(w["text"])
                        page_ner_tags.append(label)
                        page_bboxes.append(normalize_bbox(w["box"], size))
                        page_line_ids.append(item_line_id)

                # ==================== 生成页级别样本 ====================
                if len(page_tokens) > 0:
                    yield guid, {
                        "id": str(guid),
                        "document_name": document_name,
                        "page_number": page_num,
                        "tokens": page_tokens,
                        "bboxes": page_bboxes,
                        "ner_tags": page_ner_tags,
                        "image": image,
                        "line_ids": page_line_ids,  # 全局 line_id
                        "line_parent_ids": page_parent_ids,  # 该页行的 parent_id（全局索引）
                        "line_relations": page_relations,
                    }
                    guid += 1
                    doc_has_valid_page = True

                    # 快速模式：达到 max_samples 后停止生成
                    if max_samples is not None and guid >= max_samples:
                        logger.info(f"  Reached max_samples={max_samples}, stopping generation")
                        return

            # 统计文档数
            if doc_has_valid_page:
                doc_count += 1
