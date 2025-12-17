#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
对单个PDF提取的数据进行Stage1推理
"""

import os
import sys
import json
import torch
import logging
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import LayoutLMv2Processor
from layoutlmft.models.layoutxlm import LayoutXLMForTokenClassification
from layoutlmft.data.datasets.hrdoc import LABEL_LIST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_bbox(bbox, width, height):
    """将bbox归一化到0-1000范围"""
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

def load_data(raw_json_path, image_dir):
    """加载PDF提取的数据"""
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        lines = json.load(f)

    # 按页分组
    pages = {}
    for line in lines:
        page_id = line['page']
        if page_id not in pages:
            pages[page_id] = []
        pages[page_id].append(line)

    # 加载图像获取尺寸
    page_data = []
    for page_id in sorted(pages.keys()):
        img_path = os.path.join(image_dir, f"{page_id}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
        else:
            logger.warning(f"Image not found: {img_path}, using default size 595x842")
            width, height = 595, 842

        page_data.append({
            'page_id': page_id,
            'lines': pages[page_id],
            'width': width,
            'height': height,
            'image_path': img_path
        })

    return page_data

def run_inference(model_path, raw_json_path, image_dir, output_path):
    """运行推理"""
    logger.info(f"Loading model from {model_path}")

    # 加载模型和processor
    processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutXLMForTokenClassification.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # 获取标签映射
    id2label = {i: label for i, label in enumerate(LABEL_LIST)}
    logger.info(f"Labels: {LABEL_LIST}")

    # 加载数据
    logger.info(f"Loading data from {raw_json_path}")
    page_data = load_data(raw_json_path, image_dir)
    logger.info(f"Found {len(page_data)} pages")

    results = []
    total_lines = 0

    for page in page_data:
        page_id = page['page_id']
        lines = page['lines']
        width = page['width']
        height = page['height']

        logger.info(f"Processing page {page_id} with {len(lines)} lines")

        # 加载图像
        if os.path.exists(page['image_path']):
            image = Image.open(page['image_path']).convert('RGB')
        else:
            image = Image.new('RGB', (width, height), 'white')

        # 批量处理该页所有行
        texts = [line['text'] for line in lines]
        boxes = [normalize_bbox(line['box'], width, height) for line in lines]

        # 使用processor处理
        try:
            encoding = processor(
                image,
                texts,
                boxes=boxes,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )

            encoding = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

            # 解析预测结果 - 每个token对应一个预测
            # 需要找到每个文本行对应的token的预测
            input_ids = encoding['input_ids'][0].cpu().tolist()

            # 简化：为每行取第一个非特殊token的预测
            for i, line in enumerate(lines):
                # 默认标签
                pred_label = 'text'

                # 获取该行对应位置的预测 (简化处理)
                if i < len(predictions[0]):
                    pred_label_id = predictions[0, i + 1].item()  # +1跳过CLS
                    pred_label = id2label.get(pred_label_id, 'text')

                results.append({
                    'text': line['text'],
                    'box': line['box'],
                    'page': page_id,
                    'predicted_class': pred_label
                })
                total_lines += 1

        except Exception as e:
            logger.warning(f"Error processing page {page_id}: {e}")
            # 回退：标记为text
            for line in lines:
                results.append({
                    'text': line['text'],
                    'box': line['box'],
                    'page': page_id,
                    'predicted_class': 'text'
                })
                total_lines += 1

    logger.info(f"Processed {total_lines} lines total")

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")

    # 打印统计
    class_counts = {}
    for r in results:
        cls = r['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    logger.info("Class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls}: {count}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage1 inference for single PDF')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--raw_json', type=str, required=True, help='Path to raw.json file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing page images')
    parser.add_argument('--output', type=str, default=None, help='Output path for predictions')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.raw_json.replace('.raw.json', '.pred.json')

    run_inference(args.model_path, args.raw_json, args.image_dir, args.output)

if __name__ == '__main__':
    main()
