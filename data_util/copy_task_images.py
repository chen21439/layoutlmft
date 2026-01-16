"""
复制 taskId 对应的图片文件夹到目标目录

源路径: /data/LLM_group/ontology/static/upload/{taskId}/images/{filename}/
目标路径: /data/LLM_group/layoutlmft/data/tender_document/images/{filename}/

用法:
  python copy_task_images.py taskId1,taskId2,taskId3
  python copy_task_images.py taskId1
"""

import shutil
import sys
from pathlib import Path


# 源基础目录
SOURCE_BASE = Path("/data/LLM_group/ontology/static/upload")

# 目标目录
TARGET_DIR = Path("/data/LLM_group/layoutlmft/data/tender_document/images")


def copy_task_images(task_id: str) -> dict:
    """
    复制单个 taskId 的图片文件夹

    返回:
        dict: 包含 success, filename, file_count, error 等信息
    """
    result = {
        "task_id": task_id,
        "success": False,
        "filename": None,
        "file_count": 0,
    }

    # 源 images 目录
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

    # 获取 filename 文件夹
    filename_dir = subdirs[0]
    filename = filename_dir.name
    result["filename"] = filename

    # 目标路径
    target_path = TARGET_DIR / filename

    try:
        # 确保目标父目录存在
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

        # 如果目标已存在，先删除
        if target_path.exists():
            shutil.rmtree(target_path)

        # 复制整个文件夹
        shutil.copytree(filename_dir, target_path)

        # 统计文件数量
        file_count = sum(1 for f in target_path.rglob("*") if f.is_file())
        result["file_count"] = file_count
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
    print("复制 taskId 图片文件夹")
    print(f"源基础目录: {SOURCE_BASE}")
    print(f"目标目录: {TARGET_DIR}")
    print(f"taskId 数量: {len(task_ids)}")
    print("=" * 80)
    print()

    results = []
    success_count = 0
    total_files = 0

    for task_id in task_ids:
        print(f"[处理] {task_id}")
        result = copy_task_images(task_id)
        results.append(result)

        if result["success"]:
            success_count += 1
            total_files += result["file_count"]
            print(f"  ✓ 成功: {result['filename']} ({result['file_count']} 个文件)")
        else:
            print(f"  ✗ 失败: {result.get('error', '未知错误')}")
        print()

    # 汇总
    print("=" * 80)
    print(f"处理完成: 成功 {success_count}/{len(task_ids)}, 共 {total_files} 个文件")
    print(f"目标目录: {TARGET_DIR}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
