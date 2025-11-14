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
HRDoc test dataset for hierarchical document structure analysis
"""

# HRDoc 的 31 个类别 (BIO tagging)
_LABELS = [
    "O",
    "B-TITLE", "I-TITLE",
    "B-AUTHOR", "I-AUTHOR",
    "B-AFFILI", "I-AFFILI",
    "B-MAIL", "I-MAIL",
    "B-SECTION", "I-SECTION",
    "B-FSTLINE", "I-FSTLINE",
    "B-PARALINE", "I-PARALINE",
    "B-TABLE", "I-TABLE",
    "B-FIGURE", "I-FIGURE",
    "B-CAPTION", "I-CAPTION",
    "B-EQUATION", "I-EQUATION",
    "B-FOOTER", "I-FOOTER",
    "B-HEADER", "I-HEADER",
    "B-FOOTNOTE", "I-FOOTNOTE",
    "B-OPARA", "I-OPARA",
]


class HRDocTestConfig(datasets.BuilderConfig):
    """BuilderConfig for HRDoc Test"""

    def __init__(self, **kwargs):
        """BuilderConfig for HRDoc Test.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HRDocTestConfig, self).__init__(**kwargs)


class HRDocTest(datasets.GeneratorBasedBuilder):
    """HRDoc test dataset."""

    BUILDER_CONFIGS = [
        HRDocTestConfig(name="hrdoc_test", version=datasets.Version("1.0.0"), description="HRDoc test dataset"),
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
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = "/root/code/layoutlmft/data/hrdoc_test"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir}
            )
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []
            line_ids = []  # 新增：记录每个token属于哪个line
            line_parent_ids = []  # 新增：每个line的parent_id
            line_relations = []  # 新增：每个line的relation

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

            # 支持 jpg 和 png
            image_path = os.path.join(img_dir, file)
            for ext in ['.jpg', '.png', '.jpeg']:
                temp_path = image_path.replace('.json', ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break

            image, size = load_image(image_path)

            for line_idx, item in enumerate(data["form"]):
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                # 统一标签格式为大写
                label = label.upper()

                # 获取层级关系信息
                item_id = item.get("id", line_idx)
                parent_id = item.get("parent_id", -1)
                relation = item.get("relation", "none")

                # 记录当前line的元数据
                line_parent_ids.append(parent_id)
                line_relations.append(relation)

                # 处理第一个词（B-）
                tokens.append(words[0]["text"])
                ner_tags.append("B-" + label)
                bboxes.append(normalize_bbox(words[0]["box"], size))
                line_ids.append(line_idx)  # 记录token所属的line

                # 处理后续词（I-）
                for w in words[1:]:
                    tokens.append(w["text"])
                    ner_tags.append("I-" + label)
                    bboxes.append(normalize_bbox(w["box"], size))
                    line_ids.append(line_idx)  # 记录token所属的line

            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": image,
                "line_ids": line_ids,
                "line_parent_ids": line_parent_ids,
                "line_relations": line_relations,
            }
