#!/usr/bin/env python3
"""将上传的图片和文档ID添加到训练目录。

用法：
    python add_upload_to_train.py --id 79
    python add_upload_to_train.py --id 79 --no-rsync  # 使用 shutil 而非 rsync
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 路径配置
SRC_BASE = Path("/data/LLM_group/ontology/static/upload")
JSON_FILE = Path(
    "/data/LLM_group/layoutlmft/data/tender_document/covmatch/doc_random_dev2_section/train_doc_ids.json"
)
DEST_BASE = Path("/data/LLM_group/layoutlmft/data/tender_document/images")
TRAIN_JSON_BASE = Path("/data/LLM_group/layoutlmft/data/tender_document/train")


def sync_with_rsync(src: Path, dest: Path) -> None:
    """使用 rsync 同步目录（带进度显示）。"""
    cmd = ["rsync", "-avP", "--progress", str(src), str(dest) + "/"]
    subprocess.run(cmd, check=True)


def sync_with_shutil(src: Path, dest: Path) -> None:
    """使用 shutil 复制目录（纯 Python）。"""
    dest_folder = dest / src.name
    print(f"Copying {src} -> {dest_folder}")
    shutil.copytree(src, dest_folder)
    print("Copy completed.")


def transform_prediction(pred: dict) -> dict:
    """转换单个 prediction 的格式。

    将 location 格式:
        "location": [{"page": 0, "l": 267.0, "t": 72.0, "r": 351.0, "b": 86.0, ...}]
    转换为:
        "page": "0", "box": [267, 72, 351, 86]
    """
    result = {}
    for key, value in pred.items():
        if key == "location" and isinstance(value, list) and len(value) > 0:
            loc = value[0]
            result["page"] = str(loc.get("page", 0))
            result["box"] = [
                int(loc.get("l", 0)),
                int(loc.get("t", 0)),
                int(loc.get("r", 0)),
                int(loc.get("b", 0)),
            ]
        else:
            result[key] = value
    return result


def copy_construct_json(upload_dir: Path, doc_id: str) -> None:
    """复制 construct.json 到训练目录，只保留 predictions 字段。

    Args:
        upload_dir: 上传目录，如 /data/.../upload/79
        doc_id: 文档 ID
    """
    # 查找 *_construct.json 文件
    construct_files = list(upload_dir.glob("*_construct.json"))
    if not construct_files:
        print(f"WARNING: No *_construct.json found in {upload_dir}", file=sys.stderr)
        return

    if len(construct_files) > 1:
        print(f"WARNING: Multiple construct.json found, using first: {construct_files}", file=sys.stderr)

    src_json = construct_files[0]
    print(f"Found construct.json: {src_json}")

    # 读取并提取 predictions
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    print(f"Extracted {len(predictions)} predictions")

    # 转换格式
    predictions = [transform_prediction(p) for p in predictions]

    # 确保目标目录存在
    TRAIN_JSON_BASE.mkdir(parents=True, exist_ok=True)

    # 保存到目标目录，文件名为 {doc_id}.json
    dest_json = TRAIN_JSON_BASE / f"{doc_id}.json"
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to: {dest_json}")


def main():
    parser = argparse.ArgumentParser(description="添加上传的图片到训练目录")
    parser.add_argument("--id", required=True, type=int, help="Upload ID (e.g. 79)")
    parser.add_argument(
        "--no-rsync",
        action="store_true",
        help="使用 shutil.copytree 而非 rsync",
    )
    args = parser.parse_args()

    upload_id = args.id

    # 构建源路径
    src_images = SRC_BASE / str(upload_id) / "images"

    # 检查源目录
    if not src_images.is_dir():
        print(f"ERROR: not found dir: {src_images}", file=sys.stderr)
        sys.exit(1)

    # 检查 JSON 文件
    if not JSON_FILE.is_file():
        print(f"ERROR: JSON file not found: {JSON_FILE}", file=sys.stderr)
        sys.exit(2)

    # 确保目标目录存在
    DEST_BASE.mkdir(parents=True, exist_ok=True)

    # 查找唯一的子目录
    subdirs = sorted([d for d in src_images.iterdir() if d.is_dir()])
    if len(subdirs) != 1:
        print(
            f"ERROR: expected exactly 1 subfolder under: {src_images}",
            file=sys.stderr,
        )
        print(f"Found {len(subdirs)}:", file=sys.stderr)
        for d in subdirs:
            print(f"  - {d.name}", file=sys.stderr)
        sys.exit(3)

    src_folder = subdirs[0]
    doc_id = src_folder.name
    print(f"Detected folder name (will append to JSON & sync images): {doc_id}")
    print(f"Source folder: {src_folder}")

    # 同步图片目录
    dest_folder = DEST_BASE / doc_id
    if dest_folder.exists():
        print(f"Destination exists, removing for overwrite: {dest_folder}")
        shutil.rmtree(dest_folder)

    print(f"Syncing images to: {DEST_BASE}/")
    if args.no_rsync:
        sync_with_shutil(src_folder, DEST_BASE)
    else:
        sync_with_rsync(src_folder, DEST_BASE)

    # 复制 construct.json 到训练目录
    upload_dir = SRC_BASE / str(upload_id)
    copy_construct_json(upload_dir, doc_id)

    # 备份 JSON 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = JSON_FILE.with_suffix(f".json.bak.{timestamp}")
    shutil.copy2(JSON_FILE, backup)
    print(f"Backup created: {backup}")

    # 更新 JSON 文件
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: JSON root is not a list", file=sys.stderr)
        sys.exit(4)

    if doc_id in data:
        print(f"Already exists, skip: {doc_id}")
    else:
        data.append(doc_id)
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Appended: {doc_id}")

    print("Done.")


if __name__ == "__main__":
    main()
