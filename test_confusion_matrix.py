#!/usr/bin/env python
# coding=utf-8
"""
测试脚本：演示新的 Parent Confusion Matrix 输出格式
"""

from collections import defaultdict
from typing import List, Dict

# 模拟 ID2LABEL 映射
ID2LABEL = {
    0: "fstline",
    1: "section",
    2: "paraline",
    3: "table",
    4: "header",
}


def print_parent_confusion_matrix(stats: List[Dict], id2label: Dict) -> None:
    """
    以表格格式打印 Parent 混淆矩阵

    格式示例：
    +-------------+-------------+----------+-------------------------+
    | Child Class | GT Parent   | Acc      | Mispredictions          |
    +-------------+-------------+----------+-------------------------+
    | fstline     | fstline     | 90% (587/652) | section:54, paraline:11 |
    ...
    """
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for item in stats:
        child_cls = item["child_class"]
        gt_p_cls = item["gt_parent_class"]
        pred_p_cls = item["pred_parent_class"]
        confusion[child_cls][gt_p_cls][pred_p_cls] += 1

    # 收集所有需要显示的行（只显示有错误的）
    rows = []
    for child_cls in sorted(confusion.keys()):
        child_name = id2label.get(child_cls, f"cls_{child_cls}")

        for gt_p_cls in sorted(confusion[child_cls].keys(), key=lambda x: (x is None, x)):
            gt_p_name = id2label.get(gt_p_cls, f"cls_{gt_p_cls}") if gt_p_cls is not None else "ROOT"
            pred_counts = confusion[child_cls][gt_p_cls]
            total = sum(pred_counts.values())
            correct = pred_counts.get(gt_p_cls, 0)

            # 只显示有错误的情况
            if correct < total:
                error_count = total - correct

                # 收集错误详情，按数量从大到小排序
                errors_detail = []
                for pred_p_cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
                    if pred_p_cls != gt_p_cls:
                        pred_p_name = id2label.get(pred_p_cls, f"cls_{pred_p_cls}") if pred_p_cls is not None else "ROOT"
                        errors_detail.append(f"{pred_p_name}:{cnt}")

                acc_pct = 100 * correct / total if total > 0 else 0
                rows.append({
                    'child_name': child_name,
                    'gt_name': gt_p_name,
                    'acc_pct': acc_pct,
                    'correct': correct,
                    'total': total,
                    'error_count': error_count,
                    'errors_detail': ', '.join(errors_detail),
                })

    # 按错误数量从大到小排序
    rows.sort(key=lambda x: -x['error_count'])

    if not rows:
        print(f"[Evaluator Debug] Parent Confusion Matrix: No errors found")
        return

    # 计算列宽
    col_widths = {
        'child': max(13, max(len(row['child_name']) for row in rows) + 2) if rows else 13,
        'gt': max(13, max(len(row['gt_name']) for row in rows) + 2) if rows else 13,
        'acc': max(10, 12),  # "90% (587/652)"
        'errors': max(25, max(len(row['errors_detail']) for row in rows) + 2) if rows else 25,
    }

    # 打印表格
    print(f"\n[Evaluator Debug] Parent Confusion Matrix:")

    # 上边框
    total_width = sum(col_widths.values()) + 7  # 3 separators + 2 edges
    print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')

    # 表头
    print('| ' + 'Child Class'.ljust(col_widths['child']) + ' | ' + 'GT Parent'.ljust(col_widths['gt']) + ' | ' + 'Accuracy'.ljust(col_widths['acc']) + ' | ' + 'Mispredictions'.ljust(col_widths['errors']) + ' |')

    # 中间分隔线
    print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')

    # 数据行
    for row in rows:
        acc_str = f"{row['acc_pct']:.0f}% ({row['correct']}/{row['total']})"
        child_str = row['child_name'].ljust(col_widths['child'])
        gt_str = row['gt_name'].ljust(col_widths['gt'])
        acc_str = acc_str.ljust(col_widths['acc'])
        errors_str = row['errors_detail'].ljust(col_widths['errors'])

        print(f"| {child_str} | {gt_str} | {acc_str} | {errors_str} |")

    # 下边框
    print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')


if __name__ == '__main__':
    # 模拟统计数据
    stats = [
        # fstline -> fstline 的情况：587 正确，54 误判为 section，11 误判为 paraline
        *([{'child_class': 0, 'gt_parent_class': 0, 'pred_parent_class': 0, 'is_correct': True}] * 587),
        *([{'child_class': 0, 'gt_parent_class': 0, 'pred_parent_class': 1, 'is_correct': False}] * 54),
        *([{'child_class': 0, 'gt_parent_class': 0, 'pred_parent_class': 2, 'is_correct': False}] * 11),

        # fstline -> section 的情况：51 正确，3 误判为 fstline，1 误判为 table
        *([{'child_class': 0, 'gt_parent_class': 1, 'pred_parent_class': 1, 'is_correct': True}] * 51),
        *([{'child_class': 0, 'gt_parent_class': 1, 'pred_parent_class': 0, 'is_correct': False}] * 3),
        *([{'child_class': 0, 'gt_parent_class': 1, 'pred_parent_class': 3, 'is_correct': False}] * 1),

        # paraline -> fstline 的情况：319 正确，2 误判为 paraline，1 误判为 section
        *([{'child_class': 2, 'gt_parent_class': 0, 'pred_parent_class': 0, 'is_correct': True}] * 319),
        *([{'child_class': 2, 'gt_parent_class': 0, 'pred_parent_class': 2, 'is_correct': False}] * 2),
        *([{'child_class': 2, 'gt_parent_class': 0, 'pred_parent_class': 1, 'is_correct': False}] * 1),

        # paraline -> paraline 的情况：284 正确，9 误判为 fstline
        *([{'child_class': 2, 'gt_parent_class': 2, 'pred_parent_class': 2, 'is_correct': True}] * 284),
        *([{'child_class': 2, 'gt_parent_class': 2, 'pred_parent_class': 0, 'is_correct': False}] * 9),

        # section -> section 的情况：74 正确，4 误判为 fstline，1 误判为 table
        *([{'child_class': 1, 'gt_parent_class': 1, 'pred_parent_class': 1, 'is_correct': True}] * 74),
        *([{'child_class': 1, 'gt_parent_class': 1, 'pred_parent_class': 0, 'is_correct': False}] * 4),
        *([{'child_class': 1, 'gt_parent_class': 1, 'pred_parent_class': 3, 'is_correct': False}] * 1),

        # table -> section 的情况：11 正确，2 误判为 table
        *([{'child_class': 3, 'gt_parent_class': 1, 'pred_parent_class': 1, 'is_correct': True}] * 11),
        *([{'child_class': 3, 'gt_parent_class': 1, 'pred_parent_class': 3, 'is_correct': False}] * 2),
    ]

    print("=" * 100)
    print("Parent Confusion Matrix - 新格式演示")
    print("=" * 100)

    print_parent_confusion_matrix(stats, ID2LABEL)

    print("\n\n说明：")
    print("1. 只显示有错误的行（correct < total）")
    print("2. 按错误数量从大到小排序")
    print("3. 准确率用百分比显示，同时显示 (correct/total)")
    print("4. 使用 ASCII 表格边框（| 和 -），确保兼容性")
