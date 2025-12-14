#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PDF文本行提取工具 - 通过调用原始 HRDoc 模块实现

功能：
1. 从PDF提取文本行信息
2. 支持单个PDF或批量处理
3. 支持HRDS格式输出
"""

import os
import re
import sys
import json
import tqdm
import logging
import pdfplumber
import os.path as osp

# 添加 HRDoc 路径
HRDOC_PATH = "/root/code/HRDoc/end2end_system"
sys.path.insert(0, HRDOC_PATH)

# 从原始 HRDoc 模块导入核心函数
from stage1_data_prepare import (
    get_pdf_paths,
    convert_pdf2img,
    get_page2img_size,
    find_two_column_spliter,
    merge_cl_lines,
    find_bold_section,
)
from utils import extract_pdf_line

logger = logging.getLogger(__name__)


def save_info(content_lines, json_path):
    """
    保存文本行信息到JSON（修复编码问题）

    Args:
        content_lines: 提取的文本行列表
        json_path: 输出JSON路径
    """
    anno_json = []
    for page_id in range(len(content_lines)):
        page_cl = content_lines[page_id]
        for cl_id in range(len(page_cl)):
            cl = page_cl[cl_id]
            anno_json.append({
                "text": cl[0],
                "box": [int(x) for x in cl[1]],
                "page": page_id
            })

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(anno_json, f, indent=4, ensure_ascii=False)


def extract_single_pdf(pdf_path, output_dir=None, doc_name=None):
    """
    提取单个PDF的文本行信息

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录（HRDS格式），如果为None则输出到PDF同目录
        doc_name: 文档名称，如果为None则使用PDF文件名

    Returns:
        dict: {"images": [...], "annotation": "...", "doc_name": "..."}
    """
    if doc_name is None:
        doc_name = osp.splitext(osp.basename(pdf_path))[0]
        # 清理文件名中的特殊字符
        doc_name = re.sub(r'[^\w\-_\.]', '_', doc_name)

    if output_dir:
        # HRDS格式输出
        vis_folder = osp.join(output_dir, "images", doc_name)
        json_path = osp.join(output_dir, "test", f"{doc_name}.json")
        os.makedirs(osp.join(output_dir, "test"), exist_ok=True)
        os.makedirs(osp.join(output_dir, "train"), exist_ok=True)
    else:
        # 原始格式：输出到PDF同目录
        vis_folder = pdf_path[:-4] + "_vis" if pdf_path.endswith(".pdf") else pdf_path + "_vis"
        json_path = pdf_path[:-4] + ".raw.json" if pdf_path.endswith(".pdf") else pdf_path + ".raw.json"

    # 调用原始 HRDoc 函数
    raw_image_paths = convert_pdf2img(pdf_path, vis_folder)
    content_lines = extract_pdf_line(pdf_path, visual=False)

    pdf = pdfplumber.open(pdf_path)
    page2img_size = get_page2img_size(pdf)
    space_spliters = find_two_column_spliter(content_lines, page2img_size)
    content_lines = merge_cl_lines(content_lines, space_spliters)
    content_lines = find_bold_section(content_lines, page2img_size)

    # 使用修复编码的保存函数
    save_info(content_lines, json_path)

    return {
        "images": raw_image_paths,
        "annotation": json_path,
        "doc_name": doc_name
    }


def extract_pdf_folder(pdf_folder, output_dir=None, recursive=False, split='test'):
    """
    批量提取PDF文件夹中的所有PDF

    Args:
        pdf_folder: PDF文件夹路径
        output_dir: 输出目录（HRDS格式）
        recursive: 是否递归搜索子目录
        split: 输出到 'train' 或 'test' 目录

    Returns:
        dict: {pdf_path: result_dict, ...}
    """
    pdf_list = get_pdf_paths(pdf_folder, recursive=recursive)

    if isinstance(pdf_list, str):  # 错误信息
        logger.error(pdf_list)
        return {}

    if not pdf_list:
        logger.warning("No PDF files found!")
        return {}

    logger.info(f"Found {len(pdf_list)} PDF files")

    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(osp.join(output_dir, "train"), exist_ok=True)
        os.makedirs(osp.join(output_dir, "test"), exist_ok=True)
        os.makedirs(osp.join(output_dir, "images"), exist_ok=True)

    all_info = {}
    success_count = 0

    pbar = tqdm.tqdm(pdf_list)
    for pdf_path in pbar:
        pbar.set_description(f"Processing {osp.basename(pdf_path)[:30]}")
        try:
            result = extract_single_pdf(pdf_path, output_dir)

            # 如果指定了output_dir且split为train，移动json到train目录
            if output_dir and split == 'train':
                old_path = result["annotation"]
                new_path = old_path.replace("/test/", "/train/")
                if old_path != new_path and osp.exists(old_path):
                    os.rename(old_path, new_path)
                    result["annotation"] = new_path

            all_info[pdf_path] = result
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {repr(e)}")
            continue

    logger.info(f"Processed {success_count}/{len(pdf_list)} PDFs successfully")

    # 保存索引文件
    if output_dir:
        index_path = osp.join(output_dir, "index.json")
        with open(index_path, "w", encoding='utf-8') as f:
            json.dump(all_info, f, indent=4, ensure_ascii=False)
        logger.info(f"Index saved to: {index_path}")

    return all_info


# 命令行入口
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Extract PDF text lines for HRDoc')
    parser.add_argument('--pdf_folder', type=str, default=None,
                        help='Directory containing PDF files to process')
    parser.add_argument('--pdf_file', type=str, default=None,
                        help='Single PDF file to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (HRDS format). If not specified, outputs to PDF directory')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for PDFs in subdirectories')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Which split to save JSONs to (default: test)')
    args = parser.parse_args()

    if args.pdf_file:
        # 处理单个PDF
        result = extract_single_pdf(args.pdf_file, args.output_dir)
        print(f"Annotation: {result['annotation']}")
        print(f"Images: {result['images'][0]} ... ({len(result['images'])} pages)")
    elif args.pdf_folder:
        # 批量处理
        all_info = extract_pdf_folder(
            args.pdf_folder,
            output_dir=args.output_dir,
            recursive=args.recursive,
            split=args.split
        )
        print(f"\nProcessed {len(all_info)} PDFs")
    else:
        parser.error("Must specify either --pdf_folder or --pdf_file")
