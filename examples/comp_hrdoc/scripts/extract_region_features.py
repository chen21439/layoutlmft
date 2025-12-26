#!/usr/bin/env python
# coding=utf-8
"""
从 LayoutXLM 模型提取 line-level 特征并缓存到磁盘
用于 Order 模块 (4.3) 的独立训练

【HDSA 格式版本】使用 hdsa_train.json，直接是 line-level 标注
每个 line 单独编码，保持与论文一致的处理方式
"""

import logging
import os
import sys
import torch
import pickle
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any

# ==================== GPU 设置（必须在 import torch 之前）====================
def _setup_gpu_early():
    """在 import torch 之前设置 GPU"""
    env = "dev"
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]
            break

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "order.yaml"
    )

    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        gpu_config = config.get('gpu', {})
        cuda_visible_devices = gpu_config.get(env)
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            print(f"[GPU Setup] env={env}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}")


# 仅在直接运行时设置GPU（导入时不设置）
if __name__ == "__main__":
    _setup_gpu_early()
# ==================== GPU 设置结束 ====================

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from layoutlmft.models.layoutxlm import (
    LayoutXLMModel,
    LayoutXLMConfig,
    LayoutXLMTokenizerFast,
)
from layoutlmft.data.utils import load_image

# 添加 comp_hrdoc 路径
COMP_HRDOC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP_HRDOC_PATH)
from data.comp_hrdoc_loader import convert_filename_to_image_path

logger = logging.getLogger(__name__)


# ==================== 路径配置 ====================

def get_data_path(env: str) -> str:
    """获取数据路径 (hdsa 格式)"""
    paths = {
        "dev": "/mnt/e/models/data/Section/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/hdsa_train.json",
        "test": "/data/LLM_group/layoutlmft/data/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/hdsa_train.json",
    }
    return paths.get(env, paths["dev"])


def get_layoutxlm_path(env: str) -> str:
    """获取 LayoutXLM 模型路径"""
    paths = {
        "dev": "/mnt/e/models/HuggingFace/hub/models--microsoft--layoutxlm-base/snapshots/8e04ebc4d3ba0013cf943b697c0aedf19b06472a",
        "test": "/data/LLM_group/HuggingFace/Hub/models--microsoft--layoutxlm-base/snapshots/8e04ebc4d3ba0013cf943b697c0aedf19b06472a",
    }
    return paths.get(env, paths["dev"])


def get_output_path(env: str) -> str:
    """获取输出路径"""
    paths = {
        "dev": "/mnt/e/models/data/Section/Comp_HRDoc/line_features",
        "test": "/data/LLM_group/layoutlmft/data/Comp_HRDoc/line_features",
    }
    return paths.get(env, paths["dev"])


def get_image_dir(env: str) -> str:
    """获取图片目录路径"""
    paths = {
        "dev": "/mnt/e/models/data/Section/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/Images",
        "test": "/data/LLM_group/layoutlmft/data/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/Images",
    }
    return paths.get(env, paths["dev"])


# ==================== 数据加载 ====================

def load_hdsa_data(json_path: str) -> List[Dict]:
    """加载 HDSA 格式数据

    hdsa_train.json 结构:
    {
        "images": [{"id": 1, "file_name": ["page0.png", ...], "width": [...], "height": [...]}],
        "annotations": [{"id": 1, "in_doc_annotations": [line1, line2, ...]}],
        "categories": [...]
    }

    每个 line 包含:
        - textline_contents: ["text"]
        - textline_polys: [[x1,y1,x2,y2,...]]
        - page_id: 页面索引
        - parent_id: 父节点 id (-1 表示根)
        - relation: 关系类型
        - reading_order_id: 阅读顺序
        - category_id: 类别
        - id: 全局唯一 id
        - in_doc_id: 文档内索引

    Returns:
        List of document samples
    """
    logger.info(f"Loading HDSA data from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}

    samples = []
    total_lines = 0
    valid_parent_count = 0

    for ann in data['annotations']:
        doc_id = ann['id']
        image_info = images.get(doc_id)

        if image_info is None:
            continue

        lines = ann.get('in_doc_annotations', [])
        if len(lines) == 0:
            continue

        # 获取页面信息
        file_names = image_info.get('file_name', [])
        widths = image_info.get('width', [])
        heights = image_info.get('height', [])

        if isinstance(widths, int):
            widths = [widths] * len(file_names)
        if isinstance(heights, int):
            heights = [heights] * len(file_names)

        pages = []
        for i, fname in enumerate(file_names):
            pages.append({
                'file_name': fname,
                'width': widths[i] if i < len(widths) else widths[0],
                'height': heights[i] if i < len(heights) else heights[0],
            })

        # 构建 id -> index 映射 (用于 parent_id 转换)
        id_to_idx = {line['id']: idx for idx, line in enumerate(lines)}

        # 处理每个 line
        processed_lines = []
        for idx, line in enumerate(lines):
            # 提取文本 (textline_contents 是列表)
            text_list = line.get('textline_contents', [])
            text = text_list[0] if text_list else ""

            # 提取 bbox (textline_polys 是 [[x1,y1,x2,y2,x3,y3,x4,y4]] 格式)
            polys = line.get('textline_polys', [[0, 0, 0, 0, 0, 0, 0, 0]])
            poly = polys[0] if polys else [0, 0, 0, 0, 0, 0, 0, 0]

            # 转换为 [x1, y1, x2, y2] 格式
            if len(poly) >= 8:
                x1, y1 = poly[0], poly[1]
                x2, y2 = poly[4], poly[5]  # 对角点
                bbox = [x1, y1, x2, y2]
            elif len(poly) == 4:
                bbox = poly
            else:
                bbox = [0, 0, 0, 0]

            # parent_id 转换为索引
            raw_parent_id = line.get('parent_id', -1)
            if raw_parent_id == -1:
                parent_idx = -1
            elif raw_parent_id in id_to_idx:
                parent_idx = id_to_idx[raw_parent_id]
                valid_parent_count += 1
            else:
                parent_idx = -1

            processed_lines.append({
                'text': text,
                'bbox': bbox,
                'page_id': line.get('page_id', 0),
                'parent_idx': parent_idx,
                'relation': line.get('relation', 0),
                'reading_order_id': line.get('reading_order_id', idx),
                'category_id': line.get('category_id', 3),
                'in_doc_id': line.get('in_doc_id', idx),
            })

        total_lines += len(processed_lines)

        # 从第一个文件名提取 doc_id
        doc_name = file_names[0].rsplit('_', 1)[0] if file_names else str(doc_id)

        samples.append({
            'doc_id': doc_name,
            'doc_idx': doc_id,
            'pages': pages,
            'lines': processed_lines,
            'num_lines': len(processed_lines),
            'num_pages': len(pages),
        })

    logger.info(f"Loaded {len(samples)} documents, {total_lines} lines")
    logger.info(f"Valid parent relationships: {valid_parent_count}")
    logger.info(f"Average lines per document: {total_lines / len(samples):.1f}")

    return samples


# ==================== 特征提取 ====================

class LineFeatureExtractor:
    """Line-level 特征提取器

    使用 LayoutXLM 为每个 line 提取特征向量
    """

    def __init__(
        self,
        model: LayoutXLMModel,
        tokenizer: LayoutXLMTokenizerFast,
        device: torch.device,
        max_length: int = 512,
        image_dir: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.image_dir = image_dir
        self.model.eval()
        # 当前页面的图片缓存
        self._current_image = None

    def set_page_image(self, image_path: str):
        """加载并缓存页面图片

        Args:
            image_path: 图片文件路径
        """
        if os.path.exists(image_path):
            image, _ = load_image(image_path)
            self._current_image = image.unsqueeze(0).to(self.device)
        else:
            logger.warning(f"Image not found: {image_path}, using zeros")
            self._current_image = torch.zeros(1, 3, 224, 224, device=self.device)

    @torch.no_grad()
    def extract_line_feature(
        self,
        text: str,
        bbox: List[float],
        page_width: int,
        page_height: int,
    ) -> torch.Tensor:
        """提取单个 line 的特征

        Args:
            text: line 文本
            bbox: [x1, y1, x2, y2]
            page_width: 页面宽度
            page_height: 页面高度

        Returns:
            [hidden_size] 特征向量
        """
        text = text.strip() if text else ""
        if not text:
            text = "[empty]"

        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            tokens = ["[UNK]"]

        # 截断
        max_tokens = self.max_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # 归一化 bbox
        x1, y1, x2, y2 = bbox
        norm_bbox = [
            int(x1 * 1000 / page_width) if page_width > 0 else 0,
            int(y1 * 1000 / page_height) if page_height > 0 else 0,
            int(x2 * 1000 / page_width) if page_width > 0 else 0,
            int(y2 * 1000 / page_height) if page_height > 0 else 0,
        ]
        norm_bbox = [max(0, min(1000, v)) for v in norm_bbox]

        # 构造输入
        input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_bboxes = [[0, 0, 0, 0]] + [norm_bbox] * len(tokens) + [[0, 0, 0, 0]]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            input_bboxes += [[0, 0, 0, 0]] * padding_length

        # 转 tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        bbox_tensor = torch.tensor([input_bboxes], dtype=torch.long, device=self.device)

        # Forward (image 由外部传入)
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox_tensor,
            attention_mask=attention_mask,
            image=self._current_image,  # 使用当前缓存的图片
            output_hidden_states=True,
        )

        # 提取特征 (mean pooling over tokens)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_size]
        token_features = hidden_states[0, 1:1+len(tokens), :]  # 去掉 [CLS] 和 [SEP]
        line_feature = token_features.mean(dim=0)  # [hidden_size]

        return line_feature

    @torch.no_grad()
    def extract_document_features(
        self,
        doc_sample: Dict,
    ) -> Dict[str, Any]:
        """提取一个文档所有 lines 的特征

        Args:
            doc_sample: 文档样本

        Returns:
            Dict containing features and metadata
        """
        pages = doc_sample['pages']
        lines = doc_sample['lines']
        doc_id = doc_sample.get('doc_id', 'unknown')
        num_lines = len(lines)

        if num_lines == 0:
            return None

        hidden_size = self.model.config.hidden_size

        # 提取每个 line 的特征
        all_features = torch.zeros(num_lines, hidden_size)
        all_mask = torch.ones(num_lines, dtype=torch.bool)

        # 按页处理，避免重复加载图片
        current_page_id = -1
        for idx, line in enumerate(lines):
            page_id = line['page_id']
            page_info = pages[page_id] if page_id < len(pages) else pages[0]

            # 如果换页了，加载新页面的图片
            if page_id != current_page_id:
                current_page_id = page_id
                # 使用共享函数转换路径
                image_path = convert_filename_to_image_path(
                    self.image_dir, page_info['file_name']
                )
                self.set_page_image(image_path)

            feature = self.extract_line_feature(
                text=line['text'],
                bbox=line['bbox'],
                page_width=page_info['width'],
                page_height=page_info['height'],
            )
            all_features[idx] = feature.cpu()

        # 收集标签信息
        bboxes = []
        categories = []
        reading_orders = []
        parent_ids = []
        relations = []
        page_ids = []

        for line in lines:
            page_id = line['page_id']
            page_info = pages[page_id] if page_id < len(pages) else pages[0]

            # 归一化 bbox
            x1, y1, x2, y2 = line['bbox']
            norm_bbox = [
                int(x1 * 1000 / page_info['width']) if page_info['width'] > 0 else 0,
                int(y1 * 1000 / page_info['height']) if page_info['height'] > 0 else 0,
                int(x2 * 1000 / page_info['width']) if page_info['width'] > 0 else 0,
                int(y2 * 1000 / page_info['height']) if page_info['height'] > 0 else 0,
            ]
            norm_bbox = [max(0, min(1000, v)) for v in norm_bbox]

            bboxes.append(norm_bbox)
            categories.append(line['category_id'])
            reading_orders.append(line['reading_order_id'])
            parent_ids.append(line['parent_idx'])
            relations.append(line['relation'])
            page_ids.append(line['page_id'])

        return {
            'line_features': all_features,
            'line_mask': all_mask,
            'bboxes': torch.tensor(bboxes, dtype=torch.float),
            'categories': torch.tensor(categories, dtype=torch.long),
            'reading_orders': torch.tensor(reading_orders, dtype=torch.long),
            'parent_ids': torch.tensor(parent_ids, dtype=torch.long),
            'relations': torch.tensor(relations, dtype=torch.long),
            'page_ids': torch.tensor(page_ids, dtype=torch.long),
            'num_lines': num_lines,
        }


def extract_all_features(
    samples: List[Dict],
    extractor: LineFeatureExtractor,
    output_dir: str,
    split_name: str,
    samples_per_chunk: int = 100,
) -> List[str]:
    """提取所有文档的特征并保存

    Args:
        samples: 文档样本列表
        extractor: 特征提取器
        output_dir: 输出目录
        split_name: 数据集名称 (train/validation)
        samples_per_chunk: 每个 chunk 的文档数

    Returns:
        保存的 chunk 文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)

    total_lines = sum(s['num_lines'] for s in samples)
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting {split_name} set features (HDSA line-level)")
    logger.info(f"Total documents: {len(samples)}")
    logger.info(f"Total lines: {total_lines}")
    logger.info(f"Documents per chunk: {samples_per_chunk}")
    logger.info(f"{'='*60}\n")

    chunk_files = []
    current_chunk = []
    chunk_idx = 0

    for sample in tqdm(samples, desc=f"Extracting {split_name}"):
        if sample['num_lines'] == 0:
            continue

        features = extractor.extract_document_features(sample)

        if features is None:
            continue

        # 添加元数据
        features['doc_id'] = sample['doc_id']
        features['num_pages'] = sample['num_pages']
        features['pages'] = sample['pages']

        current_chunk.append(features)

        if len(current_chunk) >= samples_per_chunk:
            chunk_file = os.path.join(
                output_dir,
                f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl"
            )
            logger.info(f"Saving chunk {chunk_idx}: {len(current_chunk)} documents")
            with open(chunk_file, 'wb') as f:
                pickle.dump(current_chunk, f)
            chunk_files.append(chunk_file)
            current_chunk = []
            chunk_idx += 1

    # 保存剩余
    if len(current_chunk) > 0:
        chunk_file = os.path.join(
            output_dir,
            f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl"
        )
        logger.info(f"Saving final chunk {chunk_idx}: {len(current_chunk)} documents")
        with open(chunk_file, 'wb') as f:
            pickle.dump(current_chunk, f)
        chunk_files.append(chunk_file)

    logger.info(f"\n✓ {split_name} complete: {len(samples)} documents, {len(chunk_files)} chunks")

    return chunk_files


# ==================== 主函数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract line features (HDSA format)")

    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--layoutxlm-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--samples-per-chunk", type=int, default=100)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quick", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Paths
    data_path = args.data_path or get_data_path(args.env)
    layoutxlm_path = args.layoutxlm_path or get_layoutxlm_path(args.env)
    output_dir = args.output_dir or get_output_path(args.env)
    image_dir = get_image_dir(args.env)

    logger.info(f"Environment: {args.env}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"LayoutXLM path: {layoutxlm_path}")
    logger.info(f"Image dir: {image_dir}")
    logger.info(f"Output dir: {output_dir}")

    if args.quick:
        args.max_samples = args.max_samples or 10
        logger.info(f"Quick mode: max_samples={args.max_samples}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    samples = load_hdsa_data(data_path)

    if args.max_samples:
        samples = samples[:args.max_samples]
        logger.info(f"Limited to {len(samples)} samples")

    # Split
    num_val = int(len(samples) * args.val_split_ratio)
    train_samples = samples[num_val:]
    val_samples = samples[:num_val]
    logger.info(f"Train: {len(train_samples)}, Validation: {len(val_samples)}")

    # Load model
    logger.info("Loading LayoutXLM...")
    config = LayoutXLMConfig.from_pretrained(layoutxlm_path)
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(layoutxlm_path)
    model = LayoutXLMModel.from_pretrained(layoutxlm_path, config=config)
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"Model loaded: hidden_size={config.hidden_size}")

    # Extract
    extractor = LineFeatureExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        image_dir=image_dir,
    )

    train_chunks = extract_all_features(
        samples=train_samples,
        extractor=extractor,
        output_dir=output_dir,
        split_name="train",
        samples_per_chunk=args.samples_per_chunk,
    )

    val_chunks = extract_all_features(
        samples=val_samples,
        extractor=extractor,
        output_dir=output_dir,
        split_name="validation",
        samples_per_chunk=args.samples_per_chunk,
    )

    # Save metadata
    metadata = {
        "hidden_size": config.hidden_size,
        "max_length": args.max_length,
        "val_split_ratio": args.val_split_ratio,
        "train_chunks": train_chunks,
        "val_chunks": val_chunks,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "data_format": "hdsa",
        "feature_level": "line",
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Feature extraction complete!")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
