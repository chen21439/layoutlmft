"""
批量上传docx文件并自动下载结果并修复

支持4个独立步骤：
  1. 上传docx
  2. 下载JSON
  3. 修复section节点（使用253作为参考）
  4. 重新挂载非section节点

用法：
  # 1. 仅上传
  python batch_upload_docx.py --upload <folder_path>

  # 2. 仅下载（需要提供taskId范围）
  python batch_upload_docx.py --download <start_id-end_id> <output_dir>

  # 3. 仅修复section
  python batch_upload_docx.py --fix <json_dir> <reference_json>

  # 4. 仅重新挂载
  python batch_upload_docx.py --remount <json_dir>

  # 完整流程（上传+下载+修复+重新挂载）
  python batch_upload_docx.py --all <folder_path> <reference_json>

示例：
  # 完整流程
  python batch_upload_docx.py --all "E:\\批注" "E:\\fulltext\\253_construct.json"

  # 分步执行
  python batch_upload_docx.py --upload "E:\\批注"
  python batch_upload_docx.py --download 271-298 "E:\\批注\\json"
  python batch_upload_docx.py --fix "E:\\批注\\json" "E:\\fulltext\\253_construct.json"
  python batch_upload_docx.py --remount "E:\\批注\\json"
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
                # 解析响应并提取dataList数组
                result = response.json()

                # 提取dataList数组
                data_to_save = []
                if isinstance(result, dict):
                    if 'data' in result and isinstance(result['data'], dict):
                        data_to_save = result['data'].get('dataList', [])
                    elif 'dataList' in result:
                        data_to_save = result.get('dataList', [])
                elif isinstance(result, list):
                    data_to_save = result

                # 保存dataList数组
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

                print(f"  ✓ taskId {task_id}: 已下载 -> {output_file.name} ({len(data_to_save)} 行)")
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


def fix_json_files(json_files, reference_file: str):
    """使用参考文件修复JSON文件（仅修复section节点）

    Args:
        json_files: JSON文件列表（Path对象）或目录路径
        reference_file: 参考文件路径（253_construct.json）
    """
    # 如果传入的是目录路径，则获取所有JSON文件
    if isinstance(json_files, (str, Path)):
        json_dir = Path(json_files)
        if json_dir.is_dir():
            json_files = list(json_dir.glob("*.json"))
            # 过滤掉带_backup的文件
            json_files = [f for f in json_files if '_backup' not in f.name]
        else:
            print(f"[错误] {json_dir} 不是一个目录")
            return

    if not json_files:
        print("[提示] 没有找到需要修复的JSON文件")
        return

    print(f"\n修复JSON文件（section节点）")
    print(f"参考文件: {reference_file}")
    print(f"文件数量: {len(json_files)}")
    print("=" * 80)

    fix_script = Path(__file__).parent / "fix_section_class_v2.py"
    if not fix_script.exists():
        print(f"[错误] 修复脚本不存在: {fix_script}")
        return

    success_count = 0
    failed_count = 0

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] 修复: {json_file.name}")

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
    print(f"  总计: {len(json_files)}")


def remount_json_files(json_dir: str):
    """重新挂载非section节点

    Args:
        json_dir: JSON目录路径
    """
    json_path = Path(json_dir)
    if not json_path.is_dir():
        print(f"[错误] {json_path} 不是一个目录")
        return

    print(f"\n重新挂载非section节点")
    print(f"JSON目录: {json_path}")
    print("=" * 80)

    concat_script = Path(__file__).parent / "concat_chapter4.py"
    if not concat_script.exists():
        print(f"[错误] concat_chapter4.py 不存在: {concat_script}")
        return

    try:
        # 调用concat_chapter4.py --dir <json_dir> --remount
        result = subprocess.run(
            [sys.executable, str(concat_script), "--dir", str(json_path), "--remount"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"✓ 重新挂载成功")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"✗ 重新挂载失败")
            if result.stderr:
                print(f"错误: {result.stderr}")
            if result.stdout:
                print(result.stdout)
    except Exception as e:
        print(f"✗ 重新挂载失败: {str(e)}")


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
            # 自动重新挂载
            remount_json_files(str(json_dir))
    else:
        print("\n[提示] 没有成功上传的任务，跳过下载")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    # 1. 仅上传
    if mode == '--upload':
        if len(sys.argv) < 3:
            print("[错误] --upload 需要指定文件夹路径")
            sys.exit(1)

        folder_path = convert_windows_path_to_wsl(sys.argv[2])
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            print(f"[错误] 文件夹不存在: {folder}")
            sys.exit(1)

        # 查找所有docx文件
        docx_files = list(folder.glob("*.docx"))
        docx_files = [f for f in docx_files if not f.name.startswith('~$')]

        if not docx_files:
            print(f"[提示] 文件夹中没有找到docx文件: {folder}")
            return

        print(f"找到 {len(docx_files)} 个docx文件")
        print("=" * 80)

        api_url = "http://localhost:9801/python/api/pdf/upload_pdf"
        task_ids = []

        for i, file_path in enumerate(docx_files, 1):
            print(f"\n[{i}/{len(docx_files)}] 上传: {file_path.name}")
            success, task_id, result = upload_docx_file(file_path, api_url)

            if success:
                print(f"  ✓ 成功 (taskId: {task_id})")
                if task_id:
                    task_ids.append(task_id)
            else:
                print(f"  ✗ 失败: {result}")

            if i < len(docx_files):
                print(f"  等待 15 秒...")
                time.sleep(15)

        if task_ids:
            print(f"\n上传完成，taskId 范围: {min(task_ids)}-{max(task_ids)}")
            print(f"\n下一步执行:")
            print(f'  python batch_upload_docx.py --download {min(task_ids)}-{max(task_ids)} "{folder}/json"')

    # 2. 仅下载
    elif mode == '--download':
        if len(sys.argv) < 4:
            print("[错误] --download 需要 taskId范围 和 输出目录")
            print("示例: --download 271-298 \"E:\\\\批注\\\\json\"")
            sys.exit(1)

        task_range = sys.argv[2]
        output_dir = convert_windows_path_to_wsl(sys.argv[3])

        try:
            start_id, end_id = map(int, task_range.split('-'))
        except ValueError:
            print(f"[错误] taskId范围格式错误: {task_range}")
            sys.exit(1)

        task_ids = list(range(start_id, end_id + 1))
        downloaded_files = download_task_results(task_ids, output_dir)

        if downloaded_files:
            print(f"\n下一步执行:")
            print(f'  python batch_upload_docx.py --fix "{output_dir}" "E:\\\\fulltext\\\\253_construct.json"')

    # 3. 仅修复
    elif mode == '--fix':
        if len(sys.argv) < 4:
            print("[错误] --fix 需要 JSON目录 和 参考文件")
            print("示例: --fix \"E:\\\\批注\\\\json\" \"E:\\\\fulltext\\\\253_construct.json\"")
            sys.exit(1)

        json_dir = convert_windows_path_to_wsl(sys.argv[2])
        reference_path = convert_windows_path_to_wsl(sys.argv[3])

        json_path = Path(json_dir)
        ref_file = Path(reference_path)

        if not json_path.exists():
            print(f"[错误] JSON目录不存在: {json_path}")
            sys.exit(1)

        if not ref_file.exists():
            print(f"[错误] 参考文件不存在: {ref_file}")
            sys.exit(1)

        fix_json_files(json_path, str(ref_file))

        print(f"\n下一步执行:")
        print(f'  python batch_upload_docx.py --remount "{json_dir}"')

    # 4. 仅重新挂载
    elif mode == '--remount':
        if len(sys.argv) < 3:
            print("[错误] --remount 需要 JSON目录")
            print("示例: --remount \"E:\\\\批注\\\\json\"")
            sys.exit(1)

        json_dir = convert_windows_path_to_wsl(sys.argv[2])
        json_path = Path(json_dir)

        if not json_path.exists():
            print(f"[错误] JSON目录不存在: {json_path}")
            sys.exit(1)

        remount_json_files(str(json_path))

    # 5. 完整流程
    elif mode == '--all':
        if len(sys.argv) < 4:
            print("[错误] --all 需要 文件夹路径 和 参考文件")
            print("示例: --all \"E:\\\\批注\" \"E:\\\\fulltext\\\\253_construct.json\"")
            sys.exit(1)

        folder_path = convert_windows_path_to_wsl(sys.argv[2])
        reference_path = convert_windows_path_to_wsl(sys.argv[3])

        path = Path(folder_path)
        ref_file = Path(reference_path)

        if not path.exists() or not path.is_dir():
            print(f"[错误] 文件夹不存在: {path}")
            sys.exit(1)

        if not ref_file.exists():
            print(f"[错误] 参考文件不存在: {ref_file}")
            sys.exit(1)

        # 执行完整流程
        batch_upload_and_download(
            folder_path=str(path),
            reference_file=str(ref_file),
            api_url="http://localhost:9801/python/api/pdf/upload_pdf",
            api_base_url="http://localhost:9801",
            delay=15,
            wait_before_download=60
        )

    else:
        print(f"[错误] 未知模式: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
