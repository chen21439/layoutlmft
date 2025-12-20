#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PDF文本行提取工具 - 通过调用原始 HRDoc 模块实现

功能：
1. 从PDF提取文本行信息
2. 支持单个PDF或批量处理
3. 支持HRDS格式输出
4. 支持状态跟踪，跳过已处理的PDF
"""

import os
import re
import sys
import json
import glob
import math
import tqdm
import logging
import pdfplumber
import fitz
import os.path as osp
from datetime import datetime

# 添加 HRDoc 路径
HRDOC_PATH = "/root/code/HRDoc/end2end_system"
sys.path.insert(0, HRDOC_PATH)

# 从原始 HRDoc 模块导入核心函数
from utils import extract_pdf_line

logger = logging.getLogger(__name__)


# ============ 从 HRDoc stage1_data_prepare.py 导入的核心函数 ============

def get_pdf_paths(folder, recursive=False):
    if not osp.exists(folder):
        return "Error: No such file or directory: {}".format(folder)
    if recursive:
        pdf_list = glob.glob(osp.join(folder, "**/*.pdf"), recursive=True)
    else:
        pdf_list = glob.glob(osp.join(folder, "*.pdf"), recursive=False)
    return pdf_list


def convert_pdf2img(pdf_path, img_savedir, zoom_x=1.0, zoom_y=1.0, rotate=0):
    raw_image_paths = []
    if not osp.exists(img_savedir):
        os.makedirs(img_savedir)
    pdfDoc = fitz.open(pdf_path)
    for page_index in range(pdfDoc.page_count):
        page = pdfDoc[page_index]
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = osp.join(img_savedir, '%s.png' % (page_index))
        pix.save(img_path)
        raw_image_paths.append(img_path)
    return raw_image_paths


def get_page2img_size(pdf):
    page2img_size = []
    for page in pdf.pages:
        page2img_size.append([int(page.width), int(page.height)])
    return page2img_size


def find_two_column_spliter(content_lines, page2img_size):
    spliter = []
    for page_id in range(len(content_lines)):
        boxes = [x[1] for x in content_lines[page_id]]
        page_width = page2img_size[page_id][0]
        l_box_ids, r_box_ids = [], []
        for i in range(len(boxes)):
            x0, x1 = boxes[i][0], boxes[i][2]
            if (x1-x0) < page_width/3:
                continue
            if x1 < page_width/2:
                l_box_ids.append(i)
            if x0 > page_width/2:
                r_box_ids.append(i)
        l_l = min([boxes[i][0] for i in l_box_ids]) if len(l_box_ids) != 0 else -1
        l_r = max([boxes[i][2] for i in l_box_ids]) if len(l_box_ids) != 0 else -1
        r_l = min([boxes[i][0] for i in r_box_ids]) if len(r_box_ids) != 0 else -1
        r_r = max([boxes[i][2] for i in r_box_ids]) if len(r_box_ids) != 0 else -1
        spliter.append([l_l, l_r, r_l, r_r])
    return spliter


def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


def rect_distance(bx1, bx2):
    x1, y1, x1b, y1b = bx1[0], bx1[1], bx1[2], bx1[3]
    x2, y2, x2b, y2b = bx2[0], bx2[1], bx2[2], bx2[3]
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:
        return 0


def max_bbox(boxes):
    x = [b[0] for b in boxes if b[0] > 0] + [b[2] for b in boxes if b[2] > 0]
    y = [b[1] for b in boxes if b[1] > 0] + [b[3] for b in boxes if b[3] > 0]
    if len(x) == 0 or len(y) == 0:
        return [-1, -1, -1, -1]
    return [min(x), min(y), max(x), max(y)]


def merge_cl_lines(content_lines, space_spliters, page2img_size):
    def overlap_len(min1, len1, min2, len2):
        min_ = min1
        max_ = min1 + len1
        if min1 > min2:
            min_ = min2
        if (min1 + len1) < (min2 + len2):
            max_ = min2 + len2
        return max(0, len1 + len2 - (max_ - min_))

    def needs_merge(cl1, cl2, page_id, th=0.2, ave_char_width=5):
        overlap_l = overlap_len(cl1[1][1], cl1[1][3]-cl1[1][1], cl2[1][1], cl2[1][3]-cl2[1][1])
        if rect_distance(cl1[1], cl2[1]) < ave_char_width \
            and min(cl1[1][2]-cl1[1][0], cl2[1][2]-cl2[1][0]) < 2*ave_char_width \
                and overlap_l/max(min(cl1[1][3]-cl1[1][1], cl2[1][3]-cl2[1][1]), 1) > th:
            return True
        width = page2img_size[page_id][0]
        if abs(cl1[1][0]-space_spliters[page_id][0])+abs(cl1[1][2]-space_spliters[page_id][1]) < 4*ave_char_width or abs(cl1[1][0]-space_spliters[page_id][2])+abs(cl1[1][2]-space_spliters[page_id][3]) < 4*ave_char_width:
            return False
        if abs(cl2[1][0]-space_spliters[page_id][0])+abs(cl2[1][2]-space_spliters[page_id][1]) < 4*ave_char_width or abs(cl2[1][0]-space_spliters[page_id][2])+abs(cl2[1][2]-space_spliters[page_id][3]) < 4*ave_char_width:
            return False
        if (cl1[1][2]-cl1[1][0]) > width/6 and (cl2[1][2]-cl2[1][0]) > width/6:
            return False
        if rect_distance(cl1[1][:4], cl2[1][:4]) > 5*ave_char_width:
            return False
        overlap_l = overlap_len(cl1[1][1], cl1[1][3]-cl1[1][1], cl2[1][1], cl2[1][3]-cl2[1][1])
        if overlap_l/max(min(cl1[1][3]-cl1[1][1], cl2[1][3]-cl2[1][1]), 1) > th:
            return True
        return False

    def need_merge_ids_in_groups(cur_id, id_groups):
        ids = set()
        for x in id_groups[cur_id]:
            for i in range(len(id_groups)):
                if x in id_groups[i]:
                    ids.add(i)
        return ids

    def make_unique(id_groups):
        cur_id = 0
        while cur_id < len(id_groups):
            need_merge_ids = need_merge_ids_in_groups(cur_id, id_groups)
            if len(need_merge_ids) == 1:
                cur_id += 1
            else:
                need_merge_ids = list(need_merge_ids)
                need_merge_ids.sort()
                assert need_merge_ids[0] == cur_id
                for i in range(len(need_merge_ids)-1, 0, -1):
                    id_groups[cur_id].extend(id_groups[need_merge_ids[i]])
                    id_groups.pop(need_merge_ids[i])
                id_groups[cur_id] = list(set(id_groups[cur_id]))
                id_groups[cur_id].sort()
        for g in id_groups:
            g.sort()

    def which_group(x, groups):
        for gid, g in enumerate(groups):
            if x in g:
                return gid
        return None

    def cl_join(cl_s, page_id):
        if len(cl_s) == 0:
            return cl_s
        if len(cl_s) == 1:
            return cl_s[0]
        cl_x_id = [[cl_s[cl_id][1][0], cl_id] for cl_id in range(len(cl_s))]
        cl_x_id.sort(key=lambda x: x[0])
        sorted_cl_ids = [x[1] for x in cl_x_id]
        strings = [cl_s[cl_i][0] for cl_i in sorted_cl_ids]
        max_box_ = max_bbox([cl[1] for cl in cl_s])
        chars = []
        for cl_i in sorted_cl_ids:
            chars.extend(cl_s[cl_i][2])
        return [' '.join(strings), max_box_, chars]

    content_lines_merged = []
    for _ in range(len(content_lines)):
        content_lines_merged.append([])
    for page_id in range(len(content_lines)):
        need_merge_line_id_groups = []
        RANGE = 5
        cur_group = []
        for cl_id in range(len(content_lines[page_id])):
            cur_group = []
            for cur_cl_id in range(max(0, cl_id-RANGE), min(len(content_lines[page_id]), cl_id+RANGE)):
                if needs_merge(content_lines[page_id][cl_id], content_lines[page_id][cur_cl_id], page_id):
                    cur_group.append(cur_cl_id)
            if len(cur_group) > 1:
                need_merge_line_id_groups.append(cur_group)

        if len(cur_group) > 1:
            need_merge_line_id_groups.append(cur_group)
        make_unique(need_merge_line_id_groups)
        all_need_to_merge_ids = []
        for group in need_merge_line_id_groups:
            all_need_to_merge_ids.extend(group)
        new_cl_this_page = []
        merged_to_new = [False] * len(need_merge_line_id_groups)
        for cl_id in range(len(content_lines[page_id])):
            if cl_id not in all_need_to_merge_ids:
                new_cl_this_page.append(content_lines[page_id][cl_id])
            else:
                group_id = which_group(cl_id, need_merge_line_id_groups)
                if not merged_to_new[group_id]:
                    cl_lists = [content_lines[page_id][i] for i in need_merge_line_id_groups[group_id]]
                    new_cl_this_page.append(cl_join(cl_lists, page_id))
                    merged_to_new[group_id] = True
        content_lines_merged[page_id] = new_cl_this_page
    return content_lines_merged


def in_which_region(page2img_size, bbox, page_id):
    page_width = page2img_size[page_id][0]
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        return 'Mid'
    if bbox[2] <= page_width/2:
        return 'Left'
    if bbox[0] >= page_width/2:
        return 'Right'
    return 'Mid'


def is_in_same_paragraph(page2img_size, space_spliters, pre_line, next_line, pre_line_page_id, next_line_page_id, ave_char_width=5):
    pre_line_box = pre_line[1]
    next_line_box = next_line[1]
    pre_line_h = pre_line_box[3] - pre_line_box[1]
    next_line_h = next_line_box[3] - next_line_box[1]
    cond1 = abs(next_line_h - pre_line_h) / max(max(pre_line_h, next_line_h), 1) < 0.5
    if not cond1:
        return False
    if len(pre_line[2]) == 0 or len(next_line[2]) == 0:
        return False
    pre_in_ls = in_which_region(page2img_size, pre_line_box, pre_line_page_id) == 'Left'
    pre_in_rs = in_which_region(page2img_size, pre_line_box, pre_line_page_id) == 'Right'
    next_in_ls = in_which_region(page2img_size, next_line_box, next_line_page_id) == 'Left'
    next_in_rs = in_which_region(page2img_size, next_line_box, next_line_page_id) == 'Right'
    cond2 = (pre_in_ls and next_in_rs and pre_line_page_id == next_line_page_id) \
            or (pre_in_rs and next_in_ls and pre_line_page_id == next_line_page_id - 1)
    cond3 = ((pre_in_ls and next_in_ls) or (pre_in_rs and next_in_rs)) and pre_line_page_id == next_line_page_id
    cond4 = next_line_box[1] - pre_line_box[3] < 0.25 * (pre_line_h + next_line_h)
    cond5 = (pre_line_box[0] - next_line_box[0]) > -ave_char_width \
        and (pre_line_box[0] - next_line_box[0]) < 4 * ave_char_width \
        and (pre_line_box[2] - next_line_box[2]) > -ave_char_width
    cond6 = False
    if next_in_rs:
        cond6 = abs(next_line_box[0] - space_spliters[next_line_page_id][2]) < ave_char_width and abs(pre_line_box[2] - space_spliters[pre_line_page_id][1]) < ave_char_width
    elif next_in_ls:
        cond6 = abs(next_line_box[0] - space_spliters[next_line_page_id][0]) < ave_char_width and abs(pre_line_box[2] - space_spliters[pre_line_page_id][3]) < ave_char_width
    cond7 = (pre_line_box[2] - pre_line_box[0]) > 0.6 * page2img_size[pre_line_page_id][0] \
        and abs((pre_line_box[2] + pre_line_box[0]) / 2 - page2img_size[pre_line_page_id][0] / 2) < 0.1 * page2img_size[pre_line_page_id][0]
    if cond1:
        if cond2:
            if cond6:
                return True
        if cond3:
            if cond4 and cond5:
                return True
        if cond7:
            if cond4 and cond5:
                return True
    return False


def font_bold(s, section_suffix=['-Medi', 'Bold', 'BoldMT']):
    for i in section_suffix:
        if s.endswith(i):
            return True
    return False


def find_bold_section(content_lines, page2img_size, space_spliters):
    def merge_cl_using_chars(chars):
        str_ = ""
        max_box_ = max_bbox([ch[3] for ch in chars])
        last_right_x = chars[0][3][0]
        for ch in chars:
            if ch[3][0] <= last_right_x:
                str_ = str_ + ch[0]
            else:
                str_ = str_ + ' ' + ch[0]
            last_right_x = ch[3][2]
        return [str_, max_box_, chars]

    for page_id in range(len(content_lines)):
        cl_id = 0
        while cl_id < len(content_lines[page_id]):
            cl = content_lines[page_id][cl_id]
            if len(cl[2]) == 0:
                cl_id += 1
                continue
            if cl_id > 0 and is_in_same_paragraph(page2img_size, space_spliters, content_lines[page_id][cl_id-1], cl, page_id, page_id):
                cl_id += 1
                continue
            if not font_bold(cl[2][0][1]):
                cl_id += 1
                continue
            fonts_are_bold = [font_bold(x) for x in [x[1] for x in cl[2]]]
            if False not in fonts_are_bold:
                cl_id += 1
                if cl_id >= len(content_lines[page_id]):
                    break
                cur_cl = content_lines[page_id][cl_id]
                if len(cur_cl[2]) == 0:
                    continue
                cur_cl_fonts_are_bold = [font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                while False not in cur_cl_fonts_are_bold:
                    cl_id += 1
                    if cl_id >= len(content_lines[page_id]):
                        break
                    cur_cl = content_lines[page_id][cl_id]
                    if len(cur_cl[2]) == 0:
                        break
                    cur_cl_fonts_are_bold = [font_bold(x) for x in [x[1] for x in cur_cl[2]]]
                if cl_id >= len(content_lines[page_id]):
                    break
                if len(cur_cl[2]) == 0:
                    cl_id += 1
                    continue
                bold_font_max_id = cur_cl_fonts_are_bold.index(False) if False in cur_cl_fonts_are_bold else len(cur_cl_fonts_are_bold)
                if bold_font_max_id == 0:
                    cl_id += 1
                    continue
                new_cl1 = merge_cl_using_chars(cur_cl[2][:bold_font_max_id])
                new_cl2 = merge_cl_using_chars(cur_cl[2][bold_font_max_id:])
                new_merged_cls = content_lines[page_id][:cl_id]
                new_merged_cls.append(new_cl1)
                if bold_font_max_id < len(cur_cl[2]):
                    new_merged_cls.append(new_cl2)
                new_merged_cls.extend(content_lines[page_id][cl_id+1:])
                content_lines[page_id] = new_merged_cls
                cl_id += 2
            else:
                bold_font_max_id = fonts_are_bold.index(False)
                new_cl1 = merge_cl_using_chars(cl[2][:bold_font_max_id])
                new_cl2 = merge_cl_using_chars(cl[2][bold_font_max_id:])
                new_merged_cls = content_lines[page_id][:cl_id]
                new_merged_cls.append(new_cl1)
                new_merged_cls.append(new_cl2)
                new_merged_cls.extend(content_lines[page_id][cl_id+1:])
                content_lines[page_id] = new_merged_cls
                cl_id += 2
    return content_lines


# ============ 状态管理 ============

class ExtractionStatus:
    """PDF提取状态管理"""

    def __init__(self, status_file):
        self.status_file = status_file
        self.status = self._load()

    def _load(self):
        if osp.exists(self.status_file):
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"extracted": {}, "failed": {}}

    def save(self):
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)

    def is_extracted(self, pdf_path):
        return pdf_path in self.status["extracted"]

    def mark_extracted(self, pdf_path, result):
        record = {
            "doc_name": result["doc_name"],
            "num_pages": len(result["images"]),
            "timestamp": datetime.now().isoformat()
        }
        if result.get("annotation"):
            record["annotation"] = result["annotation"]
        self.status["extracted"][pdf_path] = record
        self.save()

    def mark_failed(self, pdf_path, error):
        self.status["failed"][pdf_path] = {
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        self.save()

    def get_stats(self):
        return {
            "extracted": len(self.status["extracted"]),
            "failed": len(self.status["failed"])
        }


def save_info(content_lines, json_path):
    """
    保存文本行信息到JSON（HRDS格式，修复编码问题）

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


def extract_single_pdf(pdf_path, output_dir=None, doc_name=None, images_only=False):
    """
    提取单个PDF的文本行信息

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录（HRDS格式），如果为None则输出到PDF同目录
        doc_name: 文档名称，如果为None则使用PDF文件名
        images_only: 是否只提取图片，不生成JSON

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
        os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
        if not images_only:
            os.makedirs(osp.join(output_dir, "test"), exist_ok=True)
            os.makedirs(osp.join(output_dir, "train"), exist_ok=True)
    else:
        # 原始格式：输出到PDF同目录
        vis_folder = pdf_path[:-4] + "_vis" if pdf_path.endswith(".pdf") else pdf_path + "_vis"
        json_path = pdf_path[:-4] + ".raw.json" if pdf_path.endswith(".pdf") else pdf_path + ".raw.json"

    # 提取PDF图片
    raw_image_paths = convert_pdf2img(pdf_path, vis_folder)

    # 如果只需要图片，直接返回
    if images_only:
        return {
            "images": raw_image_paths,
            "annotation": None,
            "doc_name": doc_name
        }

    # 提取文本行
    content_lines = extract_pdf_line(pdf_path, visual=False)

    # 处理文本行
    pdf = pdfplumber.open(pdf_path)
    page2img_size = get_page2img_size(pdf)
    space_spliters = find_two_column_spliter(content_lines, page2img_size)
    content_lines = merge_cl_lines(content_lines, space_spliters, page2img_size)
    content_lines = find_bold_section(content_lines, page2img_size, space_spliters)

    # 保存结果
    save_info(content_lines, json_path)

    return {
        "images": raw_image_paths,
        "annotation": json_path,
        "doc_name": doc_name
    }


def extract_pdf_folder(pdf_folder, output_dir, work_dir, recursive=False, split='test', images_only=False):
    """
    批量提取PDF文件夹中的所有PDF，支持状态跟踪

    Args:
        pdf_folder: PDF文件夹路径
        output_dir: 输出目录（HRDS格式）
        work_dir: 工作目录，用于存储状态文件
        recursive: 是否递归搜索子目录
        split: 输出到 'train' 或 'test' 目录
        images_only: 是否只提取图片，不生成JSON

    Returns:
        dict: {pdf_path: result_dict, ...}
    """
    # 初始化状态管理
    status_file = osp.join(work_dir, "pdf_extract_status.json")
    status = ExtractionStatus(status_file)

    pdf_list = get_pdf_paths(pdf_folder, recursive=recursive)

    if isinstance(pdf_list, str):  # 错误信息
        logger.error(pdf_list)
        return {}

    if not pdf_list:
        logger.warning("No PDF files found!")
        return {}

    logger.info(f"Found {len(pdf_list)} PDF files")
    if images_only:
        logger.info("Mode: images only (no JSON)")

    # 过滤已处理的PDF
    pending_pdfs = [p for p in pdf_list if not status.is_extracted(p)]
    skipped = len(pdf_list) - len(pending_pdfs)
    if skipped > 0:
        logger.info(f"Skipping {skipped} already extracted PDFs")
    logger.info(f"Processing {len(pending_pdfs)} PDFs")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    if not images_only:
        os.makedirs(osp.join(output_dir, "train"), exist_ok=True)
        os.makedirs(osp.join(output_dir, "test"), exist_ok=True)

    all_info = {}
    success_count = 0

    pbar = tqdm.tqdm(pending_pdfs)
    for pdf_path in pbar:
        pbar.set_description(f"Processing {osp.basename(pdf_path)[:30]}")
        try:
            result = extract_single_pdf(pdf_path, output_dir, images_only=images_only)

            # 如果split为train且不是images_only模式，移动json到train目录
            if split == 'train' and not images_only and result["annotation"]:
                old_path = result["annotation"]
                new_path = old_path.replace("/test/", "/train/")
                if old_path != new_path and osp.exists(old_path):
                    os.rename(old_path, new_path)
                    result["annotation"] = new_path

            all_info[pdf_path] = result
            status.mark_extracted(pdf_path, result)
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {repr(e)}")
            status.mark_failed(pdf_path, e)
            continue

    stats = status.get_stats()
    logger.info(f"This run: {success_count}/{len(pending_pdfs)} PDFs processed")
    logger.info(f"Total: {stats['extracted']} extracted, {stats['failed']} failed")

    return all_info


# 命令行入口
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 默认路径配置
    DEFAULT_WORK_DIR = "/mnt/e/models/data/Section/tender_document"
    DEFAULT_PDF_FOLDER = "/mnt/e/models/data/Section/tender_document/pdf"
    DEFAULT_OUTPUT_DIR = "/mnt/e/models/data/Section/tender_document"

    parser = argparse.ArgumentParser(description='Extract PDF text lines for HRDoc (HRDS format)')
    parser.add_argument('--pdf_folder', type=str, default=DEFAULT_PDF_FOLDER,
                        help=f'Directory containing PDF files (default: {DEFAULT_PDF_FOLDER})')
    parser.add_argument('--pdf_file', type=str, default=None,
                        help='Single PDF file to process')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory in HRDS format (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--work_dir', type=str, default=DEFAULT_WORK_DIR,
                        help=f'Working directory for status file (default: {DEFAULT_WORK_DIR})')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for PDFs in subdirectories')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Which split to save JSONs to (default: test)')
    parser.add_argument('--images_only', action='store_true', default=True,
                        help='Only extract images from PDF, skip JSON generation (default: True)')
    parser.add_argument('--with_json', action='store_true',
                        help='Also generate JSON annotations (overrides --images_only)')
    args = parser.parse_args()

    # --with_json 覆盖 --images_only
    images_only = not args.with_json

    if args.pdf_file:
        # 处理单个PDF
        result = extract_single_pdf(args.pdf_file, args.output_dir, images_only=images_only)
        print(f"Doc name: {result['doc_name']}")
        if result['annotation']:
            print(f"Annotation: {result['annotation']}")
        print(f"Images: {result['images'][0]} ... ({len(result['images'])} pages)")
    else:
        # 批量处理
        all_info = extract_pdf_folder(
            args.pdf_folder,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            recursive=args.recursive,
            split=args.split,
            images_only=images_only
        )
        print(f"\nProcessed {len(all_info)} PDFs in this run")
        print(f"Output directory: {args.output_dir}")
        print(f"Images saved to: {osp.join(args.output_dir, 'images')}")
        print(f"Status file: {osp.join(args.work_dir, 'pdf_extract_status.json')}")
