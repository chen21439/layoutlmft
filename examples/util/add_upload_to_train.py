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
    "/data/LLM_group/layoutlmft/data/tender_document/covmatch/doc_random_dev2_seed99/train_doc_ids.json"
)
DEST_BASE = Path("/data/LLM_group/layoutlmft/data/tender_document/images")


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
