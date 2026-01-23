"""
批量上传docx文件并自动下载结果并修复

主要功能：
  指定一个目录，自动找到所有docx文件并上传，默认间隔15秒
  上传完成1分钟后，自动下载JSON结果到 dir/json 目录
  下载完成后，自动使用253_construct.json作为参考修复所有JSON

用法：
  python batch_upload_docx.py <folder_path> <reference_json>

示例：
  python batch_upload_docx.py "E:\\models\\data\\Section\\tender_document\\docx\\Section\\批注" "E:\\models\\data\\Section\\tender_document\\docx\\Section\\fulltext\\253_construct.json"
"""

import os
import sys
import time
import json
import subprocess
import requests
from pathlib import Path


def convert_windows_path_to_wsl(path_str: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not path_str:
        return path_str
    if len(path_str) >= 2 and path_str[1] == ':':
        drive_letter = path_str[0].lower()
        rest_path = path_str[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive_letter}/{rest_path}"
    return path_str


def upload_docx_file(file_path: Path, api_url: str):
    """上传单个docx文件"""
    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_path.name, f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            }
            response = requests.post(api_url, files=files, timeout=300)

            if response.status_code == 200:
                result = response.json() if response.text else {"status": "success"}
                # 提取taskId
                task_id = None
                if isinstance(result, dict):
                    if 'data' in result and isinstance(result['data'], dict):
                        task_id = result['data'].get('taskId')
                    elif 'taskId' in result:
                        task_id = result['taskId']
                return True, task_id, result
            else:
                return False, None, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, None, str(e)


def download_task_results(task_ids: list, output_dir: str, api_base_url: str = "http://localhost:9801"):
    """下载任务结果到指定目录

    Args:
        task_ids: 任务ID列表
        output_dir: 输出目录
        api_base_url: API基础URL

    Returns:
        list: 成功下载的文件路径列表
    """
    output_path = Path(output_dir)

    # 创建输出目录
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"创建输出目录: {output_path}")

    print(f"\n下载任务结果")
    print(f"API: {api_base_url}")
    print(f"输出目录: {output_path}")
    print("=" * 80)

    success_count = 0
    failed_count = 0
    failed_tasks = []
    downloaded_files = []

    for task_id in task_ids:
        result_url = f"{api_base_url}/python/api/pdf/task/{task_id}/result?result_type=construct"
        output_file = output_path / f"{task_id}_construct.json"

        try:
            response = requests.get(result_url, timeout=10)

            if response.status_code == 200:
                # 保存JSON文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(response.json(), f, ensure_ascii=False, indent=2)

                print(f"  ✓ taskId {task_id}: 已下载 -> {output_file.name}")
                success_count += 1
                downloaded_files.append(output_file)
            else:
                print(f"  ✗ taskId {task_id}: 失败 (HTTP {response.status_code})")
                failed_count += 1
                failed_tasks.append(task_id)
        except Exception as e:
            print(f"  ✗ taskId {task_id}: 失败 ({str(e)})")
            failed_count += 1
            failed_tasks.append(task_id)

    print("\n" + "=" * 80)
    print(f"下载完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {len(task_ids)}")

    if failed_tasks:
        print(f"\n失败的任务ID: {', '.join(map(str, failed_tasks))}")

    return downloaded_files


def fix_json_files(downloaded_files: list, reference_file: str):
    """使用参考文件修复下载的JSON文件

    Args:
        downloaded_files: 下载的JSON文件列表
        reference_file: 参考文件路径（253_construct.json）
    """
    if not downloaded_files:
        return

    print(f"\n修复JSON文件")
    print(f"参考文件: {reference_file}")
    print("=" * 80)

    fix_script = Path(__file__).parent / "fix_section_class_v2.py"
    if not fix_script.exists():
        print(f"[错误] 修复脚本不存在: {fix_script}")
        return

    success_count = 0
    failed_count = 0

    for i, json_file in enumerate(downloaded_files, 1):
        print(f"\n[{i}/{len(downloaded_files)}] 修复: {json_file.name}")

        # 输出文件：覆盖原文件
        output_file = json_file

        try:
            # 调用fix_section_class_v2.py
            result = subprocess.run(
                [sys.executable, str(fix_script), reference_file, str(json_file), str(output_file)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"  ✓ 修复成功")
                success_count += 1
            else:
                print(f"  ✗ 修复失败")
                if result.stderr:
                    print(f"    错误: {result.stderr[:200]}")
                failed_count += 1
        except Exception as e:
            print(f"  ✗ 修复失败: {str(e)}")
            failed_count += 1

    print("\n" + "=" * 80)
    print(f"修复完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {len(downloaded_files)}")


def batch_upload_and_download(folder_path: str,
                               reference_file: str,
                               api_url: str = "http://localhost:9801/python/api/pdf/upload_pdf",
                               api_base_url: str = "http://localhost:9801",
                               delay: int = 15,
                               wait_before_download: int = 60):
    """批量上传文件夹中的所有docx文件，并自动下载和修复结果

    Args:
        folder_path: 文件夹路径
        reference_file: 参考文件路径（253_construct.json）
        api_url: 上传API地址
        api_base_url: API基础URL
        delay: 每个文件之间的延迟秒数（默认15秒）
        wait_before_download: 上传完成后等待多少秒再下载（默认60秒）
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"[错误] 文件夹不存在: {folder}")
        return

    # 查找所有docx文件
    docx_files = list(folder.glob("*.docx"))

    # 过滤掉临时文件（以~$开头的）
    docx_files = [f for f in docx_files if not f.name.startswith('~$')]

    if not docx_files:
        print(f"[提示] 文件夹中没有找到docx文件: {folder}")
        return

    print(f"找到 {len(docx_files)} 个docx文件")
    print(f"上传到: {api_url}")
    print(f"上传间隔: {delay} 秒")
    print("=" * 80)

    success_count = 0
    failed_count = 0
    task_ids = []

    # 上传所有文件
    for i, file_path in enumerate(docx_files, 1):
        print(f"\n[{i}/{len(docx_files)}] 上传: {file_path.name}")

        success, task_id, result = upload_docx_file(file_path, api_url)

        if success:
            print(f"  ✓ 成功")
            if task_id:
                print(f"    taskId: {task_id}")
                task_ids.append(task_id)
            success_count += 1
        else:
            print(f"  ✗ 失败: {result}")
            failed_count += 1

        # 延迟（除了最后一个文件）
        if delay > 0 and i < len(docx_files):
            print(f"  等待 {delay} 秒...")
            time.sleep(delay)

    print("\n" + "=" * 80)
    print(f"上传完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {len(docx_files)}")

    # 如果有成功上传的任务，等待后下载结果
    if task_ids:
        print(f"\n等待 {wait_before_download} 秒后下载结果...")
        time.sleep(wait_before_download)

        # 自动下载到 folder/json 目录
        json_dir = folder / "json"
        downloaded_files = download_task_results(task_ids, str(json_dir), api_base_url)

        # 自动修复下载的JSON文件
        if downloaded_files:
            fix_json_files(downloaded_files, reference_file)
    else:
        print("\n[提示] 没有成功上传的任务，跳过下载")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n[错误] 请提供文件夹路径和参考JSON文件")
        print("\n示例:")
        print('  python batch_upload_docx.py \\')
        print('    "E:\\\\models\\\\data\\\\Section\\\\tender_document\\\\docx\\\\Section\\\\批注" \\')
        print('    "E:\\\\models\\\\data\\\\Section\\\\tender_document\\\\docx\\\\Section\\\\fulltext\\\\253_construct.json"')
        sys.exit(1)

    input_path = sys.argv[1]
    reference_path = sys.argv[2]

    # 转换Windows路径到WSL路径
    input_path = convert_windows_path_to_wsl(input_path)
    reference_path = convert_windows_path_to_wsl(reference_path)

    path = Path(input_path)
    ref_file = Path(reference_path)

    if not path.exists():
        print(f"[错误] 文件夹不存在: {path}")
        sys.exit(1)

    if not path.is_dir():
        print(f"[错误] 请提供文件夹路径，不是文件路径")
        sys.exit(1)

    if not ref_file.exists():
        print(f"[错误] 参考文件不存在: {ref_file}")
        sys.exit(1)

    # 执行批量上传、下载和修复
    batch_upload_and_download(
        folder_path=str(path),
        reference_file=str(ref_file),
        api_url="http://localhost:9801/python/api/pdf/upload_pdf",
        api_base_url="http://localhost:9801",
        delay=15,
        wait_before_download=60
    )


if __name__ == "__main__":
    main()
