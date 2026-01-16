"""
复制 taskId 对应的图片文件夹，并追加第四章图片

流程:
  1. 复制 static/{taskId}/images/{filename}/ → layoutlmft/images/{filename}/
  2. 追加 src/48-83.png（重命名）→ static/{taskId}/images/{filename}/
  3. 追加 src/48-83.png（重命名）→ layoutlmft/images/{filename}/
  4. 提取 static/{taskId}/{filename}_construct.json 的 prediction 字段 → layoutlmft/train/{filename}.json
  5. 将 filename 添加到 train_doc_ids.json 数组中（如果不存在）

用法:
  python copy_task_images.py taskId1,taskId2,taskId3
  python copy_task_images.py taskId1
"""

import json
import re
import shutil
import sys
from pathlib import Path


# 源基础目录 (ontology static)
SOURCE_BASE = Path("/data/LLM_group/ontology/static/upload")

# 目标目录 (layoutlmft)
TARGET_DIR = Path("/data/LLM_group/layoutlmft/data/tender_document/images")

# train 目录 (存放 _construct.json)
TRAIN_DIR = Path("/data/LLM_group/layoutlmft/data/tender_document/train")

# 文档A第四章图片源目录
CHAPTER4_SRC_DIR = Path("/data/LLM_group/layoutlmft/data/tender_document/src/批注_[GMCG2025000068-A]深圳市光明区机关事务管理中心光明区档案综合服务中心物业管理服务")

# 第四章图片范围
CHAPTER4_START = 48
CHAPTER4_END = 83

# train_doc_ids.json 文件列表
TRAIN_DOC_IDS_FILES = [
    Path("/data/LLM_group/layoutlmft/data/tender_document/covmatch/doc_random_dev2_seed99/train_doc_ids.json"),
    Path("/data/LLM_group/layoutlmft/data/tender_document/covmatch/doc_random_dev2_section/train_doc_ids.json"),
]


def convert_location_to_box(item: dict) -> dict:
    """
    将 location 格式转换为 page + box 格式

    输入格式:
        location: [{page, l, t, r, b, coord_origin}]

    输出格式:
        page: "0" (字符串)
        box: [l, t, r, b] (整数数组)
    """
    result = {k: v for k, v in item.items() if k != "location"}

    if "location" in item and item["location"]:
        loc = item["location"][0]
        result["page"] = str(loc.get("page", 0))
        result["box"] = [
            int(loc.get("l", 0)),
            int(loc.get("t", 0)),
            int(loc.get("r", 0)),
            int(loc.get("b", 0)),
        ]

    return result


def get_max_image_number(image_dir: Path) -> int:
    """获取目录中最大的图片编号"""
    max_num = -1
    pattern = re.compile(r"^(\d+)\.png$")

    for f in image_dir.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    return max_num


def append_chapter4_images(target_dir: Path, max_num: int) -> int:
    """
    追加第四章图片到目标目录

    Args:
        target_dir: 目标目录
        max_num: 目标目录中当前最大的图片编号

    Returns:
        复制的文件数量
    """
    copied = 0

    for src_num in range(CHAPTER4_START, CHAPTER4_END + 1):
        src_file = CHAPTER4_SRC_DIR / f"{src_num}.png"
        if not src_file.exists():
            continue

        # 新编号 = 原编号 - CHAPTER4_START + max_num + 1
        new_num = src_num - CHAPTER4_START + max_num + 1
        dst_file = target_dir / f"{new_num}.png"

        shutil.copy2(src_file, dst_file)
        copied += 1

    return copied


def process_task(task_id: str) -> dict:
    """
    处理单个 taskId：复制原图片 + 追加第四章图片

    返回:
        dict: 包含处理结果信息
    """
    result = {
        "task_id": task_id,
        "success": False,
        "filename": None,
        "step1_count": 0,  # 复制原图片数量
        "step2_count": 0,  # 追加到 static 的数量
        "step3_count": 0,  # 追加到 layoutlmft 的数量
        "step4_copied": False,  # 复制 _construct.json
        "step5_added": [],  # 添加到 train_doc_ids.json 的文件列表
    }

    # ========== 定位源目录 ==========
    source_images_dir = SOURCE_BASE / task_id / "images"

    if not source_images_dir.exists():
        result["error"] = f"源目录不存在: {source_images_dir}"
        return result

    # 找到 images 下唯一的子文件夹
    subdirs = [d for d in source_images_dir.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        result["error"] = f"images 目录下没有子文件夹"
        return result

    if len(subdirs) > 1:
        result["error"] = f"images 目录下有多个子文件夹: {[d.name for d in subdirs]}"
        return result

    filename_dir = subdirs[0]
    filename = filename_dir.name
    result["filename"] = filename

    # 目标路径
    static_target = filename_dir  # static/{taskId}/images/{filename}/
    layoutlmft_target = TARGET_DIR / filename

    try:
        # ========== Step 1: 复制到 layoutlmft ==========
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

        if layoutlmft_target.exists():
            shutil.rmtree(layoutlmft_target)

        shutil.copytree(filename_dir, layoutlmft_target)
        result["step1_count"] = sum(1 for f in layoutlmft_target.rglob("*") if f.is_file())

        # ========== Step 2: 追加第四章图片到 static ==========
        max_num_static = get_max_image_number(static_target)
        if max_num_static >= 0:
            result["step2_count"] = append_chapter4_images(static_target, max_num_static)

        # ========== Step 3: 追加第四章图片到 layoutlmft ==========
        max_num_layoutlmft = get_max_image_number(layoutlmft_target)
        if max_num_layoutlmft >= 0:
            result["step3_count"] = append_chapter4_images(layoutlmft_target, max_num_layoutlmft)

        # ========== Step 4: 提取 {filename}_construct.json 的 prediction 字段到 train 目录 ==========
        construct_src = SOURCE_BASE / task_id / f"{filename}_construct.json"
        if construct_src.exists():
            TRAIN_DIR.mkdir(parents=True, exist_ok=True)
            with open(construct_src, "r", encoding="utf-8") as f:
                construct_data = json.load(f)
            predictions = construct_data.get("predictions", [])
            # 转换 location 格式为 page + box 格式
            converted = [convert_location_to_box(item) for item in predictions]
            construct_dst = TRAIN_DIR / f"{filename}.json"
            with open(construct_dst, "w", encoding="utf-8") as f:
                json.dump(converted, f, ensure_ascii=False, indent=2)
            result["step4_copied"] = True

        # ========== Step 5: 添加 filename 到 train_doc_ids.json ==========
        for json_file in TRAIN_DOC_IDS_FILES:
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    doc_ids = json.load(f)
            else:
                doc_ids = []

            if filename not in doc_ids:
                doc_ids.append(filename)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(doc_ids, f, ensure_ascii=False, indent=2)
                result["step5_added"].append(json_file.name)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python copy_task_images.py taskId1,taskId2,taskId3")
        sys.exit(1)

    # 解析 taskId 列表
    task_ids_arg = sys.argv[1]
    task_ids = [tid.strip() for tid in task_ids_arg.split(",") if tid.strip()]

    if not task_ids:
        print("错误: 未提供有效的 taskId")
        sys.exit(1)

    print("=" * 80)
    print("复制图片文件夹 + 追加第四章图片 + 复制_construct.json")
    print(f"源基础目录: {SOURCE_BASE}")
    print(f"图片目标目录: {TARGET_DIR}")
    print(f"JSON目标目录: {TRAIN_DIR}")
    print(f"第四章图片: {CHAPTER4_SRC_DIR}")
    print(f"第四章范围: {CHAPTER4_START}.png ~ {CHAPTER4_END}.png")
    print(f"taskId 数量: {len(task_ids)}")
    print("=" * 80)
    print()

    results = []
    success_count = 0

    for task_id in task_ids:
        print(f"[处理] {task_id}")
        result = process_task(task_id)
        results.append(result)

        if result["success"]:
            success_count += 1
            print(f"  ✓ {result['filename']}")
            print(f"    Step1 复制到layoutlmft: {result['step1_count']} 个文件")
            print(f"    Step2 追加到static: {result['step2_count']} 个文件")
            print(f"    Step3 追加到layoutlmft: {result['step3_count']} 个文件")
            print(f"    Step4 复制_construct.json: {'✓' if result['step4_copied'] else '✗ 文件不存在'}")
            if result["step5_added"]:
                print(f"    Step5 添加到train_doc_ids: {', '.join(result['step5_added'])}")
            else:
                print(f"    Step5 添加到train_doc_ids: 已存在，跳过")
        else:
            print(f"  ✗ 失败: {result.get('error', '未知错误')}")
        print()

    # 汇总
    print("=" * 80)
    print(f"处理完成: 成功 {success_count}/{len(task_ids)}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
