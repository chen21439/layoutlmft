"""Comp_HRDH 数据加载器

加载论文作者提供的 Comp_HRDH 数据集，用于训练 Order 模块。

数据格式：
- unified_layout_analysis_train.json: 包含 reading_order_id（阅读顺序）
- hdsa_train.json: 包含文档级别的层级结构标注

关键字段：
- reading_order_id: 阅读顺序索引
- reading_order_label: 阅读顺序类型 (0=连续, 1=换列?, 2=?)
- parent_id: 父节点 ID
- relation: 关系类型
- textline_contents: 文本内容
- textline_polys: 文本行坐标
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image

# 从 tree_utils 导入层级父节点和兄弟分组计算函数
from ..utils.tree_utils import resolve_parent_and_sibling_from_tree

logger = logging.getLogger(__name__)


# ==================== 环境配置 ====================

@dataclass
class CompHRDHPaths:
    """Comp_HRDoc 数据路径配置"""

    # dev 环境路径（本地开发）
    DEV_ROOT = "/mnt/e/models/data/Section/Comp_HRDoc"
    DEV_TRAIN_DIR = os.path.join(DEV_ROOT, "HRDH_MSRA_POD_TRAIN")
    DEV_TEST_DIR = os.path.join(DEV_ROOT, "HRDH_MSRA_POD_TEST")

    # test 环境路径（服务器）
    TEST_ROOT = "/data/LLM_group/layoutlmft/data/Comp_HRDoc"
    TEST_TRAIN_DIR = os.path.join(TEST_ROOT, "HRDH_MSRA_POD_TRAIN")
    TEST_TEST_DIR = os.path.join(TEST_ROOT, "HRDH_MSRA_POD_TEST")

    @classmethod
    def get_paths(cls, env: str = "dev") -> Dict[str, str]:
        """获取指定环境的路径

        Args:
            env: 环境名称 ("dev" 或 "test")

        Returns:
            包含 train_dir, test_dir, images_dir 的字典
        """
        if env == "dev":
            train_dir = cls.DEV_TRAIN_DIR
            test_dir = cls.DEV_TEST_DIR
        elif env == "test":
            train_dir = cls.TEST_TRAIN_DIR
            test_dir = cls.TEST_TEST_DIR
        else:
            raise ValueError(f"Unknown env: {env}, expected 'dev' or 'test'")

        return {
            "train_dir": train_dir,
            "test_dir": test_dir,
            "train_images": os.path.join(train_dir, "Images"),
            "test_images": os.path.join(test_dir, "Images"),
            "train_unified": os.path.join(train_dir, "unified_layout_analysis_train.json"),
            "test_unified": os.path.join(test_dir, "unified_layout_analysis_test.json") if os.path.exists(os.path.join(test_dir, "unified_layout_analysis_test.json")) else None,
            "train_hdsa": os.path.join(train_dir, "hdsa_train.json"),
            "test_hdsa": os.path.join(test_dir, "hdsa_test.json"),
        }


# ==================== 路径工具函数 ====================

def convert_filename_to_image_path(images_dir: str, file_name: str) -> str:
    """将 HDSA 格式的 file_name 转换为实际图片路径

    HDSA file_name 格式: 1511.06408_0.png
    实际存储路径: {images_dir}/1511.06408/0.png

    Args:
        images_dir: 图片根目录
        file_name: HDSA 格式的文件名 (如 "1511.06408_0.png")

    Returns:
        实际图片路径 (如 "{images_dir}/1511.06408/0.png")
    """
    if '_' in file_name:
        # 1511.06408_0.png -> 1511.06408, 0.png
        paper_id, page_file = file_name.rsplit('_', 1)
        return os.path.join(images_dir, paper_id, page_file)
    else:
        # 兼容旧格式
        return os.path.join(images_dir, file_name)


# ==================== 类别定义 ====================

# Comp_HRDoc 的类别
COMP_HRDOC_CATEGORIES = {
    1: "fig",      # 图片
    2: "tab",      # 表格
    3: "para",     # 段落
    4: "other",    # 其他
}

# 阅读顺序标签含义（根据数据分析推测）
READING_ORDER_LABELS = {
    0: "continue",   # 连续阅读
    1: "new_column", # 换列
    2: "new_region", # 新区域
}


# ==================== 数据加载器 ====================

@dataclass
class CompHRDHConfig:
    """数据加载配置"""
    env: str = "dev"
    max_length: int = 512
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    image_size: Tuple[int, int] = (224, 224)
    use_images: bool = True
    # 验证集比例（从训练集划分）
    val_split_ratio: float = 0.1
    # 文档级别模式：将同一文档的所有页面聚合，支持跨页 parent 关系
    document_level: bool = False


class CompHRDHDataset(Dataset):
    """Comp_HRDoc 数据集

    支持两种模式:
    - 页面级别 (document_level=False): 每个样本是一个页面，跨页 parent 设为 -1
    - 文档级别 (document_level=True): 每个样本是一个文档，支持跨页 parent 关系
    """

    def __init__(
        self,
        config: CompHRDHConfig,
        split: str = "train",
        tokenizer=None,
    ):
        """
        Args:
            config: 数据配置
            split: 数据划分 ("train", "validation", "test")
            tokenizer: Tokenizer（可选，用于文本编码）
        """
        self.config = config
        self.split = split
        self.tokenizer = tokenizer

        # 获取路径
        self.paths = CompHRDHPaths.get_paths(config.env)

        # 加载数据
        if config.document_level:
            self.samples = self._load_document_level_data()
        else:
            self.samples = self._load_data()

        mode = "document-level" if config.document_level else "page-level"
        logger.info(f"Loaded {len(self.samples)} {mode} samples for {split}")

    def _load_data(self) -> List[Dict]:
        """加载并处理数据"""

        # 确定使用哪个文件
        if self.split in ["train", "validation"]:
            json_path = self.paths["train_unified"]
            images_dir = self.paths["train_images"]
        else:
            # test split
            json_path = self.paths.get("test_unified")
            if json_path is None or not os.path.exists(json_path):
                # 如果没有专门的 test 文件，使用 train 的后部分
                json_path = self.paths["train_unified"]
            images_dir = self.paths.get("test_images", self.paths["train_images"])

        logger.info(f"Loading data from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解析数据
        images = {img['id']: img for img in data['images']}
        categories = {cat['id']: cat['name'] for cat in data['categories']}

        # 按 image_id 分组 annotations
        image_annotations = defaultdict(list)
        for ann in data['annotations']:
            image_annotations[ann['image_id']].append(ann)

        # 构建样本（每个页面一个样本）
        samples = []
        for image_id, anns in image_annotations.items():
            image_info = images.get(image_id)
            if image_info is None:
                continue

            # 按 reading_order_id 排序
            anns_sorted = sorted(anns, key=lambda x: x.get('reading_order_id', 0))

            sample = {
                'image_id': image_id,
                'image_file': image_info['file_name'],
                'image_path': convert_filename_to_image_path(images_dir, image_info['file_name']),
                'width': image_info['width'],
                'height': image_info['height'],
                'annotations': anns_sorted,
                'num_regions': len(anns_sorted),
            }

            # 提取阅读顺序信息
            reading_orders = [ann.get('reading_order_id', i) for i, ann in enumerate(anns_sorted)]
            reading_labels = [ann.get('reading_order_label', 0) for ann in anns_sorted]
            relations = [ann.get('relation', 0) for ann in anns_sorted]

            # 将 parent_id (全局 annotation id) 转换为本地索引
            # 注意: parent_id 是全局 annotation id，可能跨页面
            # 只有当 parent 在当前页面内时才能映射，否则视为根节点
            id_to_idx = {ann['id']: idx for idx, ann in enumerate(anns_sorted)}
            parent_ids = []
            for ann in anns_sorted:
                pid = ann.get('parent_id', -1)
                if pid == -1:
                    parent_ids.append(-1)  # 根节点
                elif pid in id_to_idx:
                    parent_ids.append(id_to_idx[pid])  # 转换为本地索引
                else:
                    parent_ids.append(-1)  # 父节点不在当前页面，视为根节点

            sample['reading_orders'] = reading_orders
            sample['reading_labels'] = reading_labels
            sample['parent_ids'] = parent_ids
            sample['relations'] = relations

            # 转换 reading_orders → successor_labels (论文4.2.3格式)
            # reading_orders[i] 表示 region i 的阅读顺序位置
            # successor_labels[i] 表示 region i 的后继 region 索引
            num_regions = len(reading_orders)
            if num_regions > 0:
                # 按阅读顺序排序得到 region 索引序列
                sorted_indices = sorted(range(num_regions), key=lambda x: reading_orders[x])
                # 构建后继关系
                successor_labels = [-1] * num_regions
                for pos, idx in enumerate(sorted_indices):
                    if pos < num_regions - 1:
                        successor_labels[idx] = sorted_indices[pos + 1]
                    else:
                        successor_labels[idx] = idx  # 最后一个指向自己
            else:
                successor_labels = []
            sample['successor_labels'] = successor_labels

            # 提取文本和 bbox
            texts = []
            bboxes = []
            for ann in anns_sorted:
                # 合并所有 textline_contents
                text = ' '.join(ann.get('textline_contents', []))
                texts.append(text)

                # 使用 AxisAlignedBBox
                bbox = ann.get('AxisAlignedBBox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    # [x, y, w, h] -> [x1, y1, x2, y2]
                    x, y, w, h = bbox
                    bboxes.append([x, y, x + w, y + h])
                else:
                    bboxes.append([0, 0, 0, 0])

            sample['texts'] = texts
            sample['bboxes'] = bboxes
            sample['categories'] = [ann.get('category_id', 3) for ann in anns_sorted]

            samples.append(sample)

        # 划分 train/validation
        if self.split in ["train", "validation"]:
            # 按文档分组（根据 file_name 前缀）
            doc_samples = defaultdict(list)
            for sample in samples:
                # 提取文档名（去掉页码后缀）
                file_name = sample['image_file']
                doc_name = '_'.join(file_name.split('_')[:-1])
                doc_samples[doc_name].append(sample)

            # 按文档划分
            doc_names = list(doc_samples.keys())
            num_docs = len(doc_names)
            val_size = int(num_docs * self.config.val_split_ratio)

            if self.split == "train":
                selected_docs = doc_names[val_size:]
            else:  # validation
                selected_docs = doc_names[:val_size]

            samples = []
            for doc_name in selected_docs:
                samples.extend(doc_samples[doc_name])

        # 限制样本数
        max_samples = None
        if self.split == "train":
            max_samples = self.config.max_train_samples
        elif self.split == "validation":
            max_samples = self.config.max_val_samples

        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]

        return samples

    def _load_document_level_data(self) -> List[Dict]:
        """加载文档级别数据（支持跨页 parent 关系）

        每个样本是一个完整文档，包含所有页面的区域。
        parent_id 在文档范围内映射，支持跨页引用。
        """

        # 确定使用哪个文件
        if self.split in ["train", "validation"]:
            json_path = self.paths["train_unified"]
            images_dir = self.paths["train_images"]
        else:
            json_path = self.paths.get("test_unified")
            if json_path is None or not os.path.exists(json_path):
                json_path = self.paths["train_unified"]
            images_dir = self.paths.get("test_images", self.paths["train_images"])

        logger.info(f"Loading document-level data from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}

        # 按文档名分组 annotations（从 image file_name 提取文档名）
        doc_annotations = defaultdict(list)
        doc_images = defaultdict(list)

        for image_id, img_info in images.items():
            file_name = img_info['file_name']
            # 1511.06408_0.png -> 1511.06408
            doc_name = '_'.join(file_name.split('_')[:-1])
            doc_images[doc_name].append((image_id, img_info))

        for ann in data['annotations']:
            image_id = ann['image_id']
            img_info = images.get(image_id)
            if img_info is None:
                continue
            file_name = img_info['file_name']
            doc_name = '_'.join(file_name.split('_')[:-1])
            doc_annotations[doc_name].append(ann)

        # 构建文档级别样本
        samples = []
        for doc_name in doc_annotations.keys():
            anns = doc_annotations[doc_name]
            doc_imgs = doc_images[doc_name]

            # 按 annotation id 排序（全局顺序）
            anns_sorted = sorted(anns, key=lambda x: x['id'])

            # 构建文档内 annotation id -> 本地索引的映射
            id_to_idx = {ann['id']: idx for idx, ann in enumerate(anns_sorted)}

            # 提取所有数据
            texts = []
            bboxes = []
            categories = []
            reading_orders = []
            reading_labels = []
            parent_ids = []
            relations = []
            page_indices = []  # 记录每个区域所属的页面

            for idx, ann in enumerate(anns_sorted):
                # 文本
                text = ' '.join(ann.get('textline_contents', []))
                texts.append(text)

                # Bbox
                bbox = ann.get('AxisAlignedBBox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    bboxes.append([x, y, x + w, y + h])
                else:
                    bboxes.append([0, 0, 0, 0])

                # 类别
                categories.append(ann.get('category_id', 3))

                # 阅读顺序
                reading_orders.append(ann.get('reading_order_id', idx))
                reading_labels.append(ann.get('reading_order_label', 0))
                relations.append(ann.get('relation', 0))

                # Parent ID（文档内全局映射）
                pid = ann.get('parent_id', -1)
                if pid == -1:
                    parent_ids.append(-1)
                elif pid in id_to_idx:
                    parent_ids.append(id_to_idx[pid])
                else:
                    parent_ids.append(-1)  # parent 不在文档内（异常情况）

                # 页面索引
                page_indices.append(ann['image_id'])

            # 构建 successor_labels
            num_regions = len(anns_sorted)
            if num_regions > 0:
                sorted_indices = sorted(range(num_regions), key=lambda x: reading_orders[x])
                successor_labels = [-1] * num_regions
                for pos, idx in enumerate(sorted_indices):
                    if pos < num_regions - 1:
                        successor_labels[idx] = sorted_indices[pos + 1]
                    else:
                        successor_labels[idx] = idx
            else:
                successor_labels = []

            sample = {
                'doc_name': doc_name,
                'num_regions': num_regions,
                'num_pages': len(doc_imgs),
                'texts': texts,
                'bboxes': bboxes,
                'categories': categories,
                'reading_orders': reading_orders,
                'reading_labels': reading_labels,
                'parent_ids': parent_ids,
                'relations': relations,
                'successor_labels': successor_labels,
                'page_indices': page_indices,
                'images_dir': images_dir,
                'image_files': [img['file_name'] for _, img in sorted(doc_imgs, key=lambda x: x[1]['file_name'])],
            }

            samples.append(sample)

        # 划分 train/validation
        if self.split in ["train", "validation"]:
            doc_names = list(doc_annotations.keys())
            num_docs = len(doc_names)
            val_size = int(num_docs * self.config.val_split_ratio)

            if self.split == "train":
                selected_docs = set(doc_names[val_size:])
            else:
                selected_docs = set(doc_names[:val_size])

            samples = [s for s in samples if s['doc_name'] in selected_docs]

        # 限制样本数
        max_samples = None
        if self.split == "train":
            max_samples = self.config.max_train_samples
        elif self.split == "validation":
            max_samples = self.config.max_val_samples

        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]

        # 统计 parent 分布
        total_regions = sum(s['num_regions'] for s in samples)
        has_parent = sum(sum(1 for p in s['parent_ids'] if p >= 0) for s in samples)
        logger.info(f"Document-level: {has_parent}/{total_regions} ({100*has_parent/total_regions:.1f}%) regions have valid parent")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.config.document_level:
            # 文档级别模式
            result = {
                'doc_name': sample['doc_name'],
                'num_regions': sample['num_regions'],
                'num_pages': sample['num_pages'],
                'texts': sample['texts'],
                'bboxes': sample['bboxes'],
                'categories': sample['categories'],
                'reading_orders': sample['reading_orders'],
                'reading_labels': sample['reading_labels'],
                'parent_ids': sample['parent_ids'],
                'relations': sample['relations'],
                'successor_labels': sample['successor_labels'],
                'page_indices': sample['page_indices'],
                'image_files': sample['image_files'],
            }
        else:
            # 页面级别模式
            result = {
                'image_id': sample['image_id'],
                'image_file': sample['image_file'],
                'num_regions': sample['num_regions'],
                'texts': sample['texts'],
                'bboxes': sample['bboxes'],
                'categories': sample['categories'],
                'reading_orders': sample['reading_orders'],
                'reading_labels': sample['reading_labels'],
                'parent_ids': sample['parent_ids'],
                'relations': sample['relations'],
                'successor_labels': sample['successor_labels'],
            }

            # 加载图像（可选，仅页面级别）
            if self.config.use_images and os.path.exists(sample['image_path']):
                try:
                    image = Image.open(sample['image_path']).convert('RGB')
                    image = image.resize(self.config.image_size)
                    result['image'] = image
                except Exception as e:
                    logger.warning(f"Failed to load image {sample['image_path']}: {e}")
                    result['image'] = None
            else:
                result['image'] = None

        # Tokenize（可选）
        if self.tokenizer is not None:
            result = self._tokenize(result)

        return result

    def _tokenize(self, sample: Dict) -> Dict:
        """对文本进行 tokenization"""
        texts = sample['texts']
        bboxes = sample['bboxes']

        # 简单的 tokenization（每个区域作为一个单元）
        all_tokens = []
        all_bboxes = []
        all_region_ids = []

        for region_idx, (text, bbox) in enumerate(zip(texts, bboxes)):
            # 对每个区域的文本进行 tokenization
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(encoded)

            all_tokens.extend(tokens)
            all_bboxes.extend([bbox] * len(tokens))
            all_region_ids.extend([region_idx] * len(tokens))

        # 截断到最大长度
        max_len = self.config.max_length - 2  # 预留 [CLS] 和 [SEP]
        if len(all_tokens) > max_len:
            all_tokens = all_tokens[:max_len]
            all_bboxes = all_bboxes[:max_len]
            all_region_ids = all_region_ids[:max_len]

        # 添加特殊 token
        tokens = ['[CLS]'] + all_tokens + ['[SEP]']
        bboxes = [[0, 0, 0, 0]] + all_bboxes + [[0, 0, 0, 0]]
        region_ids = [-1] + all_region_ids + [-1]

        # 转换为 input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.config.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            bboxes += [[0, 0, 0, 0]] * padding_length
            region_ids += [-1] * padding_length

        sample['input_ids'] = input_ids
        sample['attention_mask'] = attention_mask
        sample['token_bboxes'] = bboxes
        sample['region_ids'] = region_ids

        return sample


class CompHRDHCollator:
    """Comp_HRDoc Data Collator (页面级别)

    将多个样本 collate 成一个 batch，用于训练。

    输出格式 (Dict[str, Tensor]):
        input_ids:        [B, seq_len]      - Token IDs
        attention_mask:   [B, seq_len]      - Attention mask
        token_bboxes:     [B, seq_len, 4]   - Token 级别 bbox
        region_ids:       [B, seq_len]      - Token 对应的区域索引
        parent_ids:       [B, N]            - 层级父节点索引 (-1 表示 ROOT)
        sibling_labels:   [B, N, N]         - 兄弟关系矩阵
        categories:       [B, N]            - 分类标签
        region_mask:      [B, N]            - 有效区域掩码
        bboxes:           [B, N, 4]         - 区域级别 bbox

    parent_ids 示例:
        parent_ids = [-1, -1, 1, 0, 3]   (N=5)

        解读：
          区域0: parent=-1 → ROOT
          区域1: parent=-1 → ROOT
          区域2: parent=1  → 区域1
          区域3: parent=0  → 区域0
          区域4: parent=3  → 区域3

        树结构：
                 ROOT
                /    \\
            区域0    区域1
              |        |
            区域3    区域2
              |
            区域4

    sibling_labels 示例:
        根据上面的树，区域0 和 区域1 互为兄弟（都在 ROOT 下）

        sibling_labels[j, k] = 1 表示区域 j 和 k 是兄弟
              0  1  2  3  4
           0 [0, 1, 0, 0, 0]   ← 区域0 和 区域1 是兄弟
           1 [1, 0, 0, 0, 0]   ← 对称
           2 [0, 0, 0, 0, 0]
           3 [0, 0, 0, 0, 0]
           4 [0, 0, 0, 0, 0]
    """

    def __init__(self, tokenizer=None, max_regions: int = 128):
        self.tokenizer = tokenizer
        self.max_regions = max_regions

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # 找出最大区域数
        max_regions = min(
            max(f['num_regions'] for f in features),
            self.max_regions
        )

        # 初始化 batch tensors
        batch = {
            'batch_size': batch_size,
            'num_regions': [f['num_regions'] for f in features],
            'image_files': [f['image_file'] for f in features],
        }

        # 区域级别数据
        reading_orders = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        reading_labels = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        parent_ids = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        relations = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        categories = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        region_mask = torch.zeros(batch_size, max_regions, dtype=torch.bool)
        # sibling_labels: 兄弟关系矩阵，从 parent_ids 推导
        sibling_labels = torch.zeros(batch_size, max_regions, max_regions, dtype=torch.long)

        # Bboxes
        bboxes = torch.zeros(batch_size, max_regions, 4, dtype=torch.float)

        for i, f in enumerate(features):
            num_regions = min(f['num_regions'], max_regions)

            reading_orders[i, :num_regions] = torch.tensor(f['reading_orders'][:num_regions])
            # 处理 successor_labels，需要修正超出范围的索引
            succ_labels = f['successor_labels'][:num_regions]
            for j, succ in enumerate(succ_labels):
                if succ >= num_regions:
                    succ_labels[j] = j  # 指向自己
            successor_labels[i, :num_regions] = torch.tensor(succ_labels)
            reading_labels[i, :num_regions] = torch.tensor(f['reading_labels'][:num_regions])
            parent_ids[i, :num_regions] = torch.tensor(f['parent_ids'][:num_regions])
            relations[i, :num_regions] = torch.tensor(f['relations'][:num_regions])
            categories[i, :num_regions] = torch.tensor(f['categories'][:num_regions])
            region_mask[i, :num_regions] = True

            # 使用 resolve_parent_and_sibling_from_tree 计算正确的层级父节点和兄弟关系
            # 这能正确处理 equality 关系（同一父节点下的兄弟节点）
            sample_parent_ids = f['parent_ids'][:num_regions]
            sample_relations = f['relations'][:num_regions]
            hierarchical_parents, sibling_groups = resolve_parent_and_sibling_from_tree(
                sample_parent_ids, sample_relations
            )
            # 根据 sibling_groups 填充 sibling_labels
            for group in sibling_groups:
                for j_idx in range(len(group)):
                    for k_idx in range(j_idx + 1, len(group)):
                        j, k = group[j_idx], group[k_idx]
                        if j < num_regions and k < num_regions:
                            sibling_labels[i, j, k] = 1
                            sibling_labels[i, k, j] = 1  # 对称

            for j, bbox in enumerate(f['bboxes'][:num_regions]):
                bboxes[i, j] = torch.tensor(bbox, dtype=torch.float)

        batch['reading_orders'] = reading_orders
        batch['successor_labels'] = successor_labels
        batch['reading_labels'] = reading_labels
        batch['parent_ids'] = parent_ids
        batch['relations'] = relations
        batch['categories'] = categories
        batch['region_mask'] = region_mask
        batch['bboxes'] = bboxes
        batch['sibling_labels'] = sibling_labels

        # Token 级别数据（如果有）
        if 'input_ids' in features[0]:
            max_seq_len = max(len(f['input_ids']) for f in features)

            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            token_bboxes = torch.zeros(batch_size, max_seq_len, 4, dtype=torch.long)
            region_ids = torch.full((batch_size, max_seq_len), -1, dtype=torch.long)

            for i, f in enumerate(features):
                seq_len = len(f['input_ids'])
                input_ids[i, :seq_len] = torch.tensor(f['input_ids'])
                attention_mask[i, :seq_len] = torch.tensor(f['attention_mask'])
                for j, bbox in enumerate(f['token_bboxes']):
                    token_bboxes[i, j] = torch.tensor(bbox)
                region_ids[i, :seq_len] = torch.tensor(f['region_ids'])

            batch['input_ids'] = input_ids
            batch['attention_mask'] = attention_mask
            batch['token_bboxes'] = token_bboxes
            batch['region_ids'] = region_ids

        # 图像（如果有）
        images = [f.get('image') for f in features]
        if images[0] is not None:
            import torchvision.transforms as T
            transform = T.ToTensor()
            batch['images'] = torch.stack([transform(img) for img in images if img is not None])

        return batch


class CompHRDHDocumentCollator:
    """Comp_HRDoc 文档级别 Data Collator

    用于文档级别训练，每个样本是一个完整文档（包含所有页面的区域）。
    支持跨页 parent 关系。

    数据处理流程:
        平铺 JSON                     解析层级关系                    Tensor
        (parent_id + relation)  →  hierarchical_parents  →  parent_ids [B, N]
                                   sibling_groups          sibling_labels [B, N, N]

    输出格式 (Dict[str, Tensor]):
        parent_ids:       [B, N]            - 层级父节点索引 (-1 表示 ROOT)
        sibling_labels:   [B, N, N]         - 兄弟关系矩阵
        categories:       [B, N]            - 分类标签
        region_mask:      [B, N]            - 有效区域掩码
        bboxes:           [B, N, 4]         - 区域级别 bbox
        page_indices:     [B, N]            - 区域所属页面索引

    parent_ids 示例:
        parent_ids[i] = j  表示区域 i 的父节点是区域 j
        parent_ids[i] = -1 表示区域 i 的父节点是 ROOT

    sibling_labels 示例:
        sibling_labels[j, k] = 1 表示区域 j 和区域 k 是兄弟
        矩阵是对称的: sibling_labels[j, k] = sibling_labels[k, j]
    """

    def __init__(self, tokenizer=None, max_regions: int = 512):
        """
        Args:
            tokenizer: Tokenizer（可选）
            max_regions: 每个文档最大区域数（比页面级别大）
        """
        self.tokenizer = tokenizer
        self.max_regions = max_regions

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # 找出最大区域数
        max_regions = min(
            max(f['num_regions'] for f in features),
            self.max_regions
        )

        # 初始化 batch tensors
        batch = {
            'batch_size': batch_size,
            'num_regions': [min(f['num_regions'], max_regions) for f in features],
            'doc_names': [f['doc_name'] for f in features],
            'num_pages': [f['num_pages'] for f in features],
        }

        # 区域级别数据
        reading_orders = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        reading_labels = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        parent_ids = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        relations = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        categories = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        region_mask = torch.zeros(batch_size, max_regions, dtype=torch.bool)
        sibling_labels = torch.zeros(batch_size, max_regions, max_regions, dtype=torch.long)
        page_indices = torch.full((batch_size, max_regions), -1, dtype=torch.long)

        # Bboxes
        bboxes = torch.zeros(batch_size, max_regions, 4, dtype=torch.float)

        for i, f in enumerate(features):
            num_regions = min(f['num_regions'], max_regions)

            reading_orders[i, :num_regions] = torch.tensor(f['reading_orders'][:num_regions])

            # 处理 successor_labels，需要修正超出范围的索引
            succ_labels = list(f['successor_labels'][:num_regions])
            for j, succ in enumerate(succ_labels):
                if succ >= num_regions:
                    succ_labels[j] = j  # 指向自己
            successor_labels[i, :num_regions] = torch.tensor(succ_labels)

            reading_labels[i, :num_regions] = torch.tensor(f['reading_labels'][:num_regions])

            # parent_ids 已经是文档内全局索引，但需要修正超出范围的
            sample_parent_ids = list(f['parent_ids'][:num_regions])
            for j, pid in enumerate(sample_parent_ids):
                if pid >= num_regions:
                    sample_parent_ids[j] = -1  # parent 被截断，视为根节点
            parent_ids[i, :num_regions] = torch.tensor(sample_parent_ids)

            relations[i, :num_regions] = torch.tensor(f['relations'][:num_regions])
            categories[i, :num_regions] = torch.tensor(f['categories'][:num_regions])
            region_mask[i, :num_regions] = True

            # page_indices
            if 'page_indices' in f:
                page_indices[i, :num_regions] = torch.tensor(f['page_indices'][:num_regions])

            # 使用 resolve_parent_and_sibling_from_tree 计算正确的层级父节点和兄弟关系
            # 这能正确处理 equality 关系（同一父节点下的兄弟节点）
            sample_relations = f['relations'][:num_regions]
            hierarchical_parents, sibling_groups = resolve_parent_and_sibling_from_tree(
                sample_parent_ids, sample_relations
            )
            # 根据 sibling_groups 填充 sibling_labels
            for group in sibling_groups:
                for j_idx in range(len(group)):
                    for k_idx in range(j_idx + 1, len(group)):
                        j, k = group[j_idx], group[k_idx]
                        if j < num_regions and k < num_regions:
                            sibling_labels[i, j, k] = 1
                            sibling_labels[i, k, j] = 1  # 对称

            for j, bbox in enumerate(f['bboxes'][:num_regions]):
                bboxes[i, j] = torch.tensor(bbox, dtype=torch.float)

        batch['reading_orders'] = reading_orders
        batch['successor_labels'] = successor_labels
        batch['reading_labels'] = reading_labels
        batch['parent_ids'] = parent_ids
        batch['relations'] = relations
        batch['categories'] = categories
        batch['region_mask'] = region_mask
        batch['bboxes'] = bboxes
        batch['sibling_labels'] = sibling_labels
        batch['page_indices'] = page_indices

        return batch


def create_comp_hrdh_datasets(
    config: CompHRDHConfig,
    tokenizer=None,
) -> Dict[str, CompHRDHDataset]:
    """创建 Comp_HRDoc 数据集

    Args:
        config: 数据配置
        tokenizer: Tokenizer（可选）

    Returns:
        包含 train 和 validation 数据集的字典
    """
    datasets = {}

    datasets['train'] = CompHRDHDataset(config, split='train', tokenizer=tokenizer)
    datasets['validation'] = CompHRDHDataset(config, split='validation', tokenizer=tokenizer)

    logger.info(f"Created datasets: train={len(datasets['train'])}, validation={len(datasets['validation'])}")

    return datasets


# ==================== 测试 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 dev 环境
    config = CompHRDHConfig(env="dev", max_train_samples=100)

    print("=== 测试数据加载 ===")
    dataset = CompHRDHDataset(config, split="train")
    print(f"Train samples: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== 第一个样本 ===")
        print(f"image_file: {sample['image_file']}")
        print(f"num_regions: {sample['num_regions']}")
        print(f"reading_orders: {sample['reading_orders'][:5]}...")
        print(f"texts[0]: {sample['texts'][0][:50]}..." if sample['texts'] else "No texts")

    # 测试 collator
    print("\n=== 测试 Collator ===")
    collator = CompHRDHCollator()
    batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
    print(f"Batch keys: {list(batch.keys())}")
    print(f"reading_orders shape: {batch['reading_orders'].shape}")
    print(f"sibling_labels shape: {batch['sibling_labels'].shape}")
    print(f"sibling_labels 非零元素数: {(batch['sibling_labels'] > 0).sum().item()}")
