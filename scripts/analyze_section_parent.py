import json
from pathlib import Path
import numpy as np

runs_base = Path('/data/LLM_group/layoutlmft/examples/stage/runs')

target_dirs = sorted([
    d for d in runs_base.iterdir()
    if d.name.startswith('20251229_') and d.name >= '20251229_190000'
])

def analyze_section_parent(run_dir):
    json_files = list(run_dir.glob('*_infer.json'))
    if not json_files:
        return None

    section_stats = {
        'gt_section_total': 0,
        'gt_section_parent_correct': 0,
    }

    section_classes = ['sec1', 'sec2', 'sec3', 'section']
    per_class_stats = {cls: {'total': 0, 'correct': 0} for cls in section_classes}

    for json_file in json_files:
        with open(json_file) as f:
            lines = json.load(f)

        for line in lines:
            gt_cls = line['gt_class']
            gt_parent = line.get('gt_parent_id', -1)
            pred_parent = line.get('pred_parent_id', -1)

            # GT 是 section 类
            if gt_cls in section_classes or 'sec' in gt_cls.lower():
                section_stats['gt_section_total'] += 1
                if gt_parent == pred_parent:
                    section_stats['gt_section_parent_correct'] += 1

                for cls in section_classes:
                    if gt_cls == cls:
                        per_class_stats[cls]['total'] += 1
                        if gt_parent == pred_parent:
                            per_class_stats[cls]['correct'] += 1
                        break

    return {
        'name': run_dir.name,
        'num_files': len(json_files),
        'section_stats': section_stats,
        'per_class_stats': per_class_stats,
    }

hrdh_results = []
tender_results = []
other_results = []

for run_dir in target_dirs:
    result = analyze_section_parent(run_dir)
    if result is None:
        continue

    if result['num_files'] == 110:
        hrdh_results.append(result)
    elif result['num_files'] == 2:
        tender_results.append(result)
    else:
        other_results.append(result)

def print_results(results, dataset_name):
    print('=' * 120)
    print(dataset_name + ' - Section Parent Accuracy')
    print('=' * 120)
    header = "Run                  Total   Correct      Acc  |      sec1         sec2         sec3      section"
    print(header)
    print('-' * 120)

    all_total, all_correct = 0, 0
    all_per_class = {cls: {'total': 0, 'correct': 0} for cls in ['sec1', 'sec2', 'sec3', 'section']}

    for r in results:
        stats = r['section_stats']
        per_cls = r['per_class_stats']
        total = stats['gt_section_total']
        correct = stats['gt_section_parent_correct']
        acc = correct / total * 100 if total > 0 else 0

        all_total += total
        all_correct += correct

        cls_strs = []
        for cls in ['sec1', 'sec2', 'sec3', 'section']:
            t = per_cls[cls]['total']
            c = per_cls[cls]['correct']
            all_per_class[cls]['total'] += t
            all_per_class[cls]['correct'] += c
            if t > 0:
                pct = c / t * 100
                cls_strs.append("{}/{}={:.1f}%".format(c, t, pct))
            else:
                cls_strs.append("-")

        print("{:<20} {:>6} {:>9} {:>8.2f}%  |  {:>12} {:>12} {:>12} {:>12}".format(
            r['name'], total, correct, acc, cls_strs[0], cls_strs[1], cls_strs[2], cls_strs[3]))

    if results:
        print('-' * 120)
        avg_acc = all_correct / all_total * 100 if all_total > 0 else 0
        cls_strs = []
        for cls in ['sec1', 'sec2', 'sec3', 'section']:
            t = all_per_class[cls]['total']
            c = all_per_class[cls]['correct']
            if t > 0:
                pct = c / t * 100
                cls_strs.append("{}/{}={:.1f}%".format(c, t, pct))
            else:
                cls_strs.append("-")
        print("{:<20} {:>6} {:>9} {:>8.2f}%  |  {:>12} {:>12} {:>12} {:>12}".format(
            'TOTAL', all_total, all_correct, avg_acc, cls_strs[0], cls_strs[1], cls_strs[2], cls_strs[3]))

print_results(hrdh_results, 'HRDH Dataset (110 files)')
print()
print_results(tender_results, 'Tender Dataset (2 files)')
print()
print_results(other_results, 'Other (3 files)')
