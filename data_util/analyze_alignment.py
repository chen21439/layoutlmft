"""
详细分析两个 construct.json 文件的对齐情况

用法：
  python analyze_alignment.py <file1.json> <file2.json>
"""

import json
import sys
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


def load_json(file_path: Path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_lines(data):
    """提取所有 line 对象"""
    if isinstance(data, dict):
        if 'data' in data and 'dataList' in data['data']:
            return data['data']['dataList']
        else:
            return []
    elif isinstance(data, list):
        return data
    else:
        return []


def analyze_match_regions(lines1, lines2, sample_size=100):
    """分析匹配区域"""

    lines1_by_id = {line['line_id']: line for line in lines1}
    lines2_by_id = {line['line_id']: line for line in lines2}

    max_id = min(max(lines1_by_id.keys()), max(lines2_by_id.keys()))

    print("=" * 80)
    print("分段匹配分析（每100行统计一次）")
    print("=" * 80)

    regions = []

    for start in range(0, max_id + 1, sample_size):
        end = min(start + sample_size, max_id + 1)

        match_count = 0
        check_count = 0

        for i in range(start, end):
            if i in lines1_by_id and i in lines2_by_id:
                text1 = lines1_by_id[i].get('text', '')
                text2 = lines2_by_id[i].get('text', '')

                if text1 == text2:
                    match_count += 1
                check_count += 1

        if check_count > 0:
            match_ratio = match_count / check_count
            regions.append({
                'start': start,
                'end': end - 1,
                'match': match_count,
                'total': check_count,
                'ratio': match_ratio
            })
            print(f"  line_id {start:4d}-{end-1:4d}: 匹配 {match_count:3d}/{check_count:3d} ({match_ratio:6.1%})")

    print()

    # 找出匹配率低的区域
    print("=" * 80)
    print("匹配率较低的区域（<50%）")
    print("=" * 80)

    low_match_regions = [r for r in regions if r['ratio'] < 0.5]

    if low_match_regions:
        for r in low_match_regions:
            print(f"  line_id {r['start']:4d}-{r['end']:4d}: 匹配 {r['match']:3d}/{r['total']:3d} ({r['ratio']:6.1%})")

            # 显示这个区域的前5个不匹配示例
            print(f"    示例:")
            shown = 0
            for i in range(r['start'], r['end'] + 1):
                if shown >= 5:
                    break

                if i in lines1_by_id and i in lines2_by_id:
                    text1 = lines1_by_id[i].get('text', '')
                    text2 = lines2_by_id[i].get('text', '')

                    if text1 != text2:
                        print(f"      [{i}] ✗")
                        print(f"        文件1: {text1[:60]}")
                        print(f"        文件2: {text2[:60]}")
                        shown += 1
            print()
    else:
        print("  所有区域匹配率都 >= 50%")

    print()

    # 统计总体
    total_match = sum(r['match'] for r in regions)
    total_check = sum(r['total'] for r in regions)
    overall_ratio = total_match / total_check if total_check > 0 else 0

    print("=" * 80)
    print("总体统计")
    print("=" * 80)
    print(f"  总匹配: {total_match}/{total_check} ({overall_ratio:.1%})")
    print(f"  不匹配: {total_check - total_match}")

    return regions


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n[错误] 请提供两个JSON文件路径")
        sys.exit(1)

    path1 = convert_windows_path_to_wsl(sys.argv[1])
    path2 = convert_windows_path_to_wsl(sys.argv[2])

    file1 = Path(path1)
    file2 = Path(path2)

    if not file1.exists():
        print(f"[错误] 文件不存在: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"[错误] 文件不存在: {file2}")
        sys.exit(1)

    print("=" * 80)
    print("详细对齐分析")
    print("=" * 80)
    print(f"文件1: {file1.name}")
    print(f"文件2: {file2.name}")
    print()

    data1 = load_json(file1)
    data2 = load_json(file2)

    lines1 = extract_lines(data1)
    lines2 = extract_lines(data2)

    print(f"文件1: {len(lines1)} 个 line")
    print(f"文件2: {len(lines2)} 个 line")
    print()

    analyze_match_regions(lines1, lines2, sample_size=100)


if __name__ == "__main__":
    main()
