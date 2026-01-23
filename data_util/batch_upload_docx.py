"""
批量上传docx文件并自动下载结果

主要功能：
  指定一个目录，自动找到所有docx文件并上传，默认间隔15秒
  上传完成1分钟后，自动下载JSON结果到 dir/json 目录

用法：
  python batch_upload_docx.py <folder_path>

示例：
  python batch_upload_docx.py "E:\\models\\data\\Section\\tender_document\\docx\\Section\\批注"
"""

import os
import sys
import time
import json
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


def batch_upload_and_download(folder_path: str,
                               api_url: str = "http://localhost:9801/python/api/pdf/upload_pdf",
                               api_base_url: str = "http://localhost:9801",
                               delay: int = 15,
                               wait_before_download: int = 60):
    """批量上传文件夹中的所有docx文件，并自动下载结果

    Args:
        folder_path: 文件夹路径
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
        download_task_results(task_ids, str(json_dir), api_base_url)
    else:
        print("\n[提示] 没有成功上传的任务，跳过下载")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n[错误] 请提供文件夹路径")
        print("\n示例:")
        print('  python batch_upload_docx.py "E:\\\\models\\\\data\\\\Section\\\\tender_document\\\\docx\\\\Section\\\\批注"')
        sys.exit(1)

    input_path = sys.argv[1]

    # 转换Windows路径到WSL路径
    input_path = convert_windows_path_to_wsl(input_path)
    path = Path(input_path)

    if not path.exists():
        print(f"[错误] 路径不存在: {path}")
        sys.exit(1)

    if not path.is_dir():
        print(f"[错误] 请提供文件夹路径，不是文件路径")
        sys.exit(1)

    # 执行批量上传和下载
    batch_upload_and_download(
        folder_path=str(path),
        api_url="http://localhost:9801/python/api/pdf/upload_pdf",
        api_base_url="http://localhost:9801",
        delay=15,
        wait_before_download=60
    )


if __name__ == "__main__":
    main()
