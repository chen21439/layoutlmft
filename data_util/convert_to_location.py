"""
将 JSON 文件中的 page + box/bbox 格式转换为 location 格式

输入格式:
  "page": 48,
  "box": [192, 92, 403, 106]

输出格式:
  "location": [{"page": 48, "l": 192.0, "t": 92.0, "r": 403.0, "b": 106.0, "coord_origin": "TOPLEFT"}]

用法:
  python convert_to_location.py input.json output.json
"""

import json
import sys
from pathlib import Path


def convert_item(item: dict) -> dict:
    """转换单个 item 的格式"""
    result = {}

    for key, value in item.items():
        if key in ("page", "box", "bbox"):
            continue  # 这些字段会被合并到 location
        result[key] = value

    # 提取 page 和 box
    page = item.get("page", 0)
    if isinstance(page, str):
        page = int(page) if page else 0

    box = item.get("box") or item.get("bbox") or [0, 0, 0, 0]

    # 生成 location
    result["location"] = [{
        "page": page,
        "l": float(box[0]) if len(box) > 0 else 0.0,
        "t": float(box[1]) if len(box) > 1 else 0.0,
        "r": float(box[2]) if len(box) > 2 else 0.0,
        "b": float(box[3]) if len(box) > 3 else 0.0,
        "coord_origin": "TOPLEFT"
    }]

    return result


def convert_file(input_path: Path, output_path: Path):
    """转换整个文件"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理可能的嵌套格式
    if isinstance(data, dict):
        if "data" in data and "dataList" in data["data"]:
            data_list = data["data"]["dataList"]
        elif "dataList" in data:
            data_list = data["dataList"]
        else:
            data_list = [data]
    else:
        data_list = data

    # 转换每个 item
    converted = [convert_item(item) for item in data_list]

    # 保存为扁平数组
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"已转换: {input_path.name}")
    print(f"  条目数: {len(converted)}")
    print(f"  输出: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_to_location.py input.json [output.json]")
        print("  如果不指定 output.json，将保存为 input_1.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # 默认输出为 xxx_1.json
        output_path = input_path.parent / f"{input_path.stem}_1.json"

    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        sys.exit(1)

    convert_file(input_path, output_path)


if __name__ == "__main__":
    main()
