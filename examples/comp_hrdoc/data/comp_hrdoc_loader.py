"""Comp_HRDoc 数据加载器

加载论文作者提供的 Comp_HRDoc 数据集，用于训练 Order 模块。

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

logger = logging.getLogger(__name__)


# ==================== 环境配置 ====================

@dataclass
class CompHRDocPaths:
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
class CompHRDocConfig:
    """数据加载配置"""
    env: str = "dev"
    max_length: int = 512
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    image_size: Tuple[int, int] = (224, 224)
    use_images: bool = True
    # 验证集比例（从训练集划分）
    val_split_ratio: float = 0.1


class CompHRDocDataset(Dataset):
    """Comp_HRDoc 数据集

    用于 Order 模块训练，每个样本是一个页面。
    """

    def __init__(
        self,
        config: CompHRDocConfig,
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
        self.paths = CompHRDocPaths.get_paths(config.env)

        # 加载数据
        self.samples = self._load_data()

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

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

            # 将 parent_id (reading_order_id) 转换为本地索引
            # 注意: parent_id 对应的是 reading_order_id，不是 annotation id
            ro_to_idx = {ann.get('reading_order_id', i): i for i, ann in enumerate(anns_sorted)}
            parent_ids = []
            for ann in anns_sorted:
                pid = ann.get('parent_id', -1)
                if pid == -1:
                    parent_ids.append(-1)  # 根节点
                elif pid in ro_to_idx:
                    parent_ids.append(ro_to_idx[pid])  # 转换为本地索引
                else:
                    parent_ids.append(-1)  # 父节点不在当前样本中，视为根节点

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

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
            'successor_labels': sample['successor_labels'],  # 论文4.2.3格式：后继索引
        }

        # 加载图像（可选）
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


class CompHRDocCollator:
    """Comp_HRDoc Data Collator"""

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

            # 从 parent_ids 推导 sibling_labels（同一父节点的区域互为兄弟）
            sample_parent_ids = f['parent_ids'][:num_regions]
            for j in range(num_regions):
                for k in range(j + 1, num_regions):
                    # 有效父节点且相同 => 兄弟关系
                    if (sample_parent_ids[j] >= 0 and
                        sample_parent_ids[j] == sample_parent_ids[k]):
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


def create_comp_hrdoc_datasets(
    config: CompHRDocConfig,
    tokenizer=None,
) -> Dict[str, CompHRDocDataset]:
    """创建 Comp_HRDoc 数据集

    Args:
        config: 数据配置
        tokenizer: Tokenizer（可选）

    Returns:
        包含 train 和 validation 数据集的字典
    """
    datasets = {}

    datasets['train'] = CompHRDocDataset(config, split='train', tokenizer=tokenizer)
    datasets['validation'] = CompHRDocDataset(config, split='validation', tokenizer=tokenizer)

    logger.info(f"Created datasets: train={len(datasets['train'])}, validation={len(datasets['validation'])}")

    return datasets


# ==================== 测试 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 dev 环境
    config = CompHRDocConfig(env="dev", max_train_samples=100)

    print("=== 测试数据加载 ===")
    dataset = CompHRDocDataset(config, split="train")
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
    collator = CompHRDocCollator()
    batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
    print(f"Batch keys: {list(batch.keys())}")
    print(f"reading_orders shape: {batch['reading_orders'].shape}")
    print(f"sibling_labels shape: {batch['sibling_labels'].shape}")
    print(f"sibling_labels 非零元素数: {(batch['sibling_labels'] > 0).sum().item()}")
