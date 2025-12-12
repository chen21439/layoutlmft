#!/usr/bin/env python
# coding=utf-8
"""
Dev Split Creator with Coverage + Distribution Matching

Two-stage greedy algorithm:
  Stage 1 (Coverage): Select docs until all Tier-1 classes meet minimum thresholds
  Stage 2 (Distribution): Continue selecting to match overall distribution

Output naming: covmatch/doc_covmatch_dev{pct}_seed{seed}/
  - train_doc_ids.json
  - dev_doc_ids.json
  - split_report.json

Usage:
    python scripts/create_dev_covmatch.py --env dev --dataset hrds
    python scripts/create_dev_covmatch.py --env dev --dataset hrdh
    python scripts/create_dev_covmatch.py --env test --dataset hrds --dev_ratio 0.10 --seed 42
"""

import os
import sys
import json
import argparse
import random
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config


# =============================================================================
# Tier Configuration
# =============================================================================

TIER1_CLASSES = {
    "mail", "affili", "fig", "tab", "figcap", "tabcap",
    "sec1", "sec2", "sec3"
}

TIER2_CLASSES = {
    "title", "author", "foot", "fnote", "header"
}

TIER3_CLASSES = {
    "secx", "alg"
}

# Thresholds
TIER1_MIN_LINES = 30
TIER1_MIN_DOCS = 5

TIER2_MIN_LINES = 10
TIER2_MIN_DOCS = 3

TIER3_MIN_LINES = 5


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_stats(data_dir: str) -> Tuple[Dict, Dict, List[str]]:
    """
    Load dataset and compute per-doc statistics.

    Returns:
        doc_stats: {doc_id: {class_counts, relation_counts, parent_density, page_count, line_count}}
        global_stats: {class_counts, relation_counts, ...}
        doc_ids: list of all doc ids
    """
    doc_stats = {}
    global_class_counts = Counter()
    global_relation_counts = Counter()
    global_parent_edges = 0
    global_lines = 0

    doc_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    doc_ids = [f.replace('.json', '') for f in doc_files]

    for fname in doc_files:
        doc_id = fname.replace('.json', '')
        filepath = os.path.join(data_dir, fname)

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = json.load(f)

        class_counts = Counter()
        relation_counts = Counter()
        parent_edges = 0
        max_page = 0

        for line in lines:
            cls = line['class']
            class_counts[cls] += 1
            global_class_counts[cls] += 1

            if 'relation' in line:
                rel = line['relation']
                relation_counts[rel] += 1
                global_relation_counts[rel] += 1

            if line.get('parent_id', -1) != -1:
                parent_edges += 1
                global_parent_edges += 1

            max_page = max(max_page, line.get('page', 0))

        line_count = len(lines)
        global_lines += line_count

        doc_stats[doc_id] = {
            'class_counts': dict(class_counts),
            'relation_counts': dict(relation_counts),
            'parent_density': parent_edges / line_count if line_count > 0 else 0,
            'page_count': max_page + 1,
            'line_count': line_count,
        }

    global_stats = {
        'class_counts': dict(global_class_counts),
        'relation_counts': dict(global_relation_counts),
        'parent_density': global_parent_edges / global_lines if global_lines > 0 else 0,
        'total_lines': global_lines,
        'total_docs': len(doc_ids),
    }

    return doc_stats, global_stats, doc_ids


# =============================================================================
# Coverage Check
# =============================================================================

def check_tier1_coverage(
    dev_doc_ids: Set[str],
    doc_stats: Dict,
    label_set: Set[str]
) -> Tuple[bool, Dict]:
    """
    Check if Tier-1 coverage is met.

    Returns:
        (all_met, {class: {lines, docs, lines_met, docs_met}})
    """
    class_lines = Counter()
    class_docs = defaultdict(set)

    for doc_id in dev_doc_ids:
        for cls, cnt in doc_stats[doc_id]['class_counts'].items():
            class_lines[cls] += cnt
            class_docs[cls].add(doc_id)

    tier1_in_dataset = TIER1_CLASSES & label_set

    coverage = {}
    all_met = True

    for cls in tier1_in_dataset:
        lines = class_lines.get(cls, 0)
        docs = len(class_docs.get(cls, set()))
        lines_met = lines >= TIER1_MIN_LINES
        docs_met = docs >= TIER1_MIN_DOCS

        coverage[cls] = {
            'lines': lines,
            'docs': docs,
            'lines_met': lines_met,
            'docs_met': docs_met,
            'met': lines_met and docs_met,
        }

        if not (lines_met and docs_met):
            all_met = False

    return all_met, coverage


def get_unmet_classes(coverage: Dict) -> List[str]:
    """Get list of Tier-1 classes that haven't met coverage."""
    return [cls for cls, info in coverage.items() if not info['met']]


# =============================================================================
# Distribution Matching
# =============================================================================

def compute_distribution_vector(
    doc_ids: Set[str],
    doc_stats: Dict,
    all_classes: List[str],
    all_relations: List[str]
) -> np.ndarray:
    """
    Compute normalized distribution vector for a set of docs.
    """
    class_counts = Counter()
    relation_counts = Counter()
    total_lines = 0
    total_parent_density = 0
    total_pages = 0

    for doc_id in doc_ids:
        stats = doc_stats[doc_id]
        for cls, cnt in stats['class_counts'].items():
            class_counts[cls] += cnt
        for rel, cnt in stats['relation_counts'].items():
            relation_counts[rel] += cnt
        total_lines += stats['line_count']
        total_parent_density += stats['parent_density']
        total_pages += stats['page_count']

    n_docs = len(doc_ids)
    if n_docs == 0 or total_lines == 0:
        return np.zeros(len(all_classes) + len(all_relations) + 2)

    class_vec = np.array([class_counts.get(cls, 0) / total_lines for cls in all_classes])

    rel_total = sum(relation_counts.values())
    if rel_total > 0:
        rel_vec = np.array([relation_counts.get(rel, 0) / rel_total for rel in all_relations])
    else:
        rel_vec = np.zeros(len(all_relations))

    parent_density = total_parent_density / n_docs
    avg_pages = total_pages / n_docs / 20.0

    return np.concatenate([class_vec, rel_vec, [parent_density], [avg_pages]])


def compute_distribution_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L1 distance between two distribution vectors."""
    return np.sum(np.abs(vec1 - vec2))


# =============================================================================
# Two-Stage Greedy Selection
# =============================================================================

def select_dev_docs(
    doc_stats: Dict,
    global_stats: Dict,
    doc_ids: List[str],
    target_dev_count: int,
    seed: int = 42
) -> List[str]:
    """
    Two-stage greedy selection:
    Stage 1: Coverage - ensure Tier-1 classes meet thresholds
    Stage 2: Distribution - match overall distribution
    """
    random.seed(seed)
    np.random.seed(seed)

    label_set = set(global_stats['class_counts'].keys())
    all_classes = sorted(label_set)
    all_relations = sorted(global_stats['relation_counts'].keys())

    all_doc_set = set(doc_ids)
    target_vec = compute_distribution_vector(all_doc_set, doc_stats, all_classes, all_relations)

    remaining = set(doc_ids)
    selected = set()

    print(f"\n{'='*60}")
    print("Stage 1: Coverage (Tier-1 classes)")
    print(f"{'='*60}")

    tier1_in_dataset = TIER1_CLASSES & label_set

    iteration = 0
    while len(selected) < target_dev_count:
        all_met, coverage = check_tier1_coverage(selected, doc_stats, label_set)

        if all_met:
            print(f"  All Tier-1 coverage met after {len(selected)} docs")
            break

        unmet = get_unmet_classes(coverage)

        best_doc = None
        best_score = -1

        for doc_id in remaining:
            score = 0
            doc_classes = doc_stats[doc_id]['class_counts']
            for cls in unmet:
                if cls in doc_classes:
                    current = coverage[cls]['lines']
                    needed = TIER1_MIN_LINES - current
                    contribution = min(doc_classes[cls], max(0, needed))
                    score += contribution

            if score > best_score:
                best_score = score
                best_doc = doc_id

        if best_doc is None or best_score == 0:
            best_doc = random.choice(list(remaining))

        selected.add(best_doc)
        remaining.remove(best_doc)
        iteration += 1

        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: {len(selected)} docs selected, unmet: {unmet}")

    _, coverage = check_tier1_coverage(selected, doc_stats, label_set)
    print(f"\nStage 1 complete: {len(selected)} docs")
    print("Tier-1 coverage status:")
    for cls in sorted(tier1_in_dataset):
        info = coverage.get(cls, {'lines': 0, 'docs': 0, 'met': False})
        status = "✓" if info.get('met', False) else "✗"
        print(f"  {cls:<10} lines={info['lines']:>4} docs={info['docs']:>3} {status}")

    print(f"\n{'='*60}")
    print("Stage 2: Distribution Matching")
    print(f"{'='*60}")

    while len(selected) < target_dev_count and remaining:
        current_vec = compute_distribution_vector(selected, doc_stats, all_classes, all_relations)
        current_dist = compute_distribution_distance(current_vec, target_vec)

        best_doc = None
        best_dist = float('inf')

        candidates = list(remaining)
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)

        for doc_id in candidates:
            test_set = selected | {doc_id}
            test_vec = compute_distribution_vector(test_set, doc_stats, all_classes, all_relations)
            test_dist = compute_distribution_distance(test_vec, target_vec)

            if test_dist < best_dist:
                best_dist = test_dist
                best_doc = doc_id

        if best_doc:
            selected.add(best_doc)
            remaining.remove(best_doc)

        if len(selected) % 20 == 0:
            print(f"  {len(selected)} docs selected, L1 distance: {best_dist:.4f}")

    print(f"\nStage 2 complete: {len(selected)} docs")
    final_vec = compute_distribution_vector(selected, doc_stats, all_classes, all_relations)
    final_dist = compute_distribution_distance(final_vec, target_vec)
    print(f"Final L1 distance from target: {final_dist:.4f}")

    return sorted(selected)


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    train_ids: List[str],
    dev_ids: List[str],
    doc_stats: Dict,
    global_stats: Dict,
    dataset_name: str
) -> Dict:
    """Generate comprehensive split report."""

    label_set = set(global_stats['class_counts'].keys())
    all_classes = sorted(label_set)
    all_relations = sorted(global_stats['relation_counts'].keys())

    def compute_split_stats(doc_ids):
        class_counts = Counter()
        class_docs = defaultdict(set)
        relation_counts = Counter()
        total_lines = 0
        total_parent_density = 0
        total_pages = 0

        for doc_id in doc_ids:
            stats = doc_stats[doc_id]
            for cls, cnt in stats['class_counts'].items():
                class_counts[cls] += cnt
                class_docs[cls].add(doc_id)
            for rel, cnt in stats['relation_counts'].items():
                relation_counts[rel] += cnt
            total_lines += stats['line_count']
            total_parent_density += stats['parent_density']
            total_pages += stats['page_count']

        n_docs = len(doc_ids)
        return {
            'n_docs': n_docs,
            'n_lines': total_lines,
            'class_counts': dict(class_counts),
            'class_docs': {k: len(v) for k, v in class_docs.items()},
            'relation_counts': dict(relation_counts),
            'parent_density': total_parent_density / n_docs if n_docs > 0 else 0,
            'avg_pages': total_pages / n_docs if n_docs > 0 else 0,
        }

    train_stats = compute_split_stats(train_ids)
    dev_stats = compute_split_stats(dev_ids)

    class_comparison = {}
    for cls in all_classes:
        train_lines = train_stats['class_counts'].get(cls, 0)
        train_docs = train_stats['class_docs'].get(cls, 0)
        dev_lines = dev_stats['class_counts'].get(cls, 0)
        dev_docs = dev_stats['class_docs'].get(cls, 0)

        train_pct = train_lines / train_stats['n_lines'] * 100 if train_stats['n_lines'] > 0 else 0
        dev_pct = dev_lines / dev_stats['n_lines'] * 100 if dev_stats['n_lines'] > 0 else 0

        tier = "Tier-1" if cls in TIER1_CLASSES else ("Tier-2" if cls in TIER2_CLASSES else ("Tier-3" if cls in TIER3_CLASSES else "Other"))

        class_comparison[cls] = {
            'tier': tier,
            'train_lines': train_lines,
            'train_docs': train_docs,
            'train_pct': round(train_pct, 2),
            'dev_lines': dev_lines,
            'dev_docs': dev_docs,
            'dev_pct': round(dev_pct, 2),
            'pct_diff': round(abs(train_pct - dev_pct), 3),
        }

    tier1_coverage = {}
    tier1_all_met = True
    for cls in TIER1_CLASSES & label_set:
        info = class_comparison[cls]
        lines_met = info['dev_lines'] >= TIER1_MIN_LINES
        docs_met = info['dev_docs'] >= TIER1_MIN_DOCS
        met = lines_met and docs_met
        tier1_coverage[cls] = {
            'lines': info['dev_lines'],
            'docs': info['dev_docs'],
            'lines_met': lines_met,
            'docs_met': docs_met,
            'met': met,
        }
        if not met:
            tier1_all_met = False

    relation_comparison = {}
    for rel in all_relations:
        train_cnt = train_stats['relation_counts'].get(rel, 0)
        dev_cnt = dev_stats['relation_counts'].get(rel, 0)
        train_total = sum(train_stats['relation_counts'].values())
        dev_total = sum(dev_stats['relation_counts'].values())

        train_pct = train_cnt / train_total * 100 if train_total > 0 else 0
        dev_pct = dev_cnt / dev_total * 100 if dev_total > 0 else 0

        relation_comparison[rel] = {
            'train_count': train_cnt,
            'train_pct': round(train_pct, 2),
            'dev_count': dev_cnt,
            'dev_pct': round(dev_pct, 2),
        }

    train_vec = compute_distribution_vector(set(train_ids), doc_stats, all_classes, all_relations)
    dev_vec = compute_distribution_vector(set(dev_ids), doc_stats, all_classes, all_relations)
    l1_distance = compute_distribution_distance(train_vec, dev_vec)

    report = {
        'dataset': dataset_name,
        'split_summary': {
            'train_docs': len(train_ids),
            'dev_docs': len(dev_ids),
            'train_lines': train_stats['n_lines'],
            'dev_lines': dev_stats['n_lines'],
            'dev_ratio': round(len(dev_ids) / (len(train_ids) + len(dev_ids)), 3),
        },
        'tier1_coverage': {
            'all_met': tier1_all_met,
            'details': tier1_coverage,
        },
        'class_comparison': class_comparison,
        'relation_comparison': relation_comparison,
        'structure_comparison': {
            'train_parent_density': round(train_stats['parent_density'], 4),
            'dev_parent_density': round(dev_stats['parent_density'], 4),
            'train_avg_pages': round(train_stats['avg_pages'], 2),
            'dev_avg_pages': round(dev_stats['avg_pages'], 2),
        },
        'distribution_l1_distance': round(l1_distance, 4),
    }

    return report


def print_report_summary(report: Dict):
    """Print human-readable report summary."""
    print(f"\n{'='*70}")
    print(f"Split Report: {report['dataset']}")
    print(f"{'='*70}")

    summary = report['split_summary']
    print(f"\nSplit Summary:")
    print(f"  Train: {summary['train_docs']} docs, {summary['train_lines']} lines")
    print(f"  Dev:   {summary['dev_docs']} docs, {summary['dev_lines']} lines")
    print(f"  Dev ratio: {summary['dev_ratio']:.1%}")

    print(f"\nTier-1 Coverage (all met: {report['tier1_coverage']['all_met']}):")
    for cls, info in sorted(report['tier1_coverage']['details'].items()):
        status = "✓" if info['met'] else "✗"
        print(f"  {cls:<10} lines={info['lines']:>4} (≥{TIER1_MIN_LINES}) "
              f"docs={info['docs']:>3} (≥{TIER1_MIN_DOCS}) {status}")

    print(f"\nClass Distribution Comparison (sorted by train %):")
    print(f"  {'Class':<12} {'Tier':<7} {'Train%':>8} {'Dev%':>8} {'Diff':>8}")
    print(f"  {'-'*50}")
    for cls, info in sorted(report['class_comparison'].items(),
                           key=lambda x: -x[1]['train_pct']):
        print(f"  {cls:<12} {info['tier']:<7} {info['train_pct']:>7.2f}% "
              f"{info['dev_pct']:>7.2f}% {info['pct_diff']:>7.3f}")

    print(f"\nRelation Distribution:")
    for rel, info in report['relation_comparison'].items():
        print(f"  {rel:<12} train={info['train_pct']:>5.1f}% dev={info['dev_pct']:>5.1f}%")

    struct = report['structure_comparison']
    print(f"\nStructure Comparison:")
    print(f"  Parent density: train={struct['train_parent_density']:.2%} dev={struct['dev_parent_density']:.2%}")
    print(f"  Avg pages/doc:  train={struct['train_avg_pages']:.1f} dev={struct['dev_avg_pages']:.1f}")

    print(f"\nOverall Distribution L1 Distance: {report['distribution_l1_distance']:.4f}")
    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Create dev split with coverage + distribution matching")

    # Environment selection
    parser.add_argument("--env", type=str, required=True,
                        help="Environment: dev or test")

    # Dataset selection (optional, defaults to config)
    parser.add_argument("--dataset", type=str, default=None, choices=["hrds", "hrdh"],
                        help="Dataset to split (default: from config)")

    # Split parameters
    parser.add_argument("--dev_ratio", type=float, default=0.10,
                        help="Dev set ratio (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without splitting")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.env)

    # Get dataset name (from args or config)
    dataset_name_lower = args.dataset or config.dataset.name
    dataset_name = dataset_name_lower.upper()
    data_dir = config.dataset.get_data_dir(dataset_name_lower)
    train_dir = os.path.join(data_dir, "train")

    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist")
        return

    # Output directory naming
    dev_pct = int(args.dev_ratio * 100)
    split_name = f"doc_covmatch_dev{dev_pct}_seed{args.seed}"
    output_dir = os.path.join(data_dir, "covmatch", split_name)

    print(f"Environment:  {args.env}")
    print(f"Dataset:      {dataset_name}")
    print(f"Data dir:     {data_dir}")
    print(f"Train dir:    {train_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Dev ratio:    {args.dev_ratio}")
    print(f"Seed:         {args.seed}")

    if args.dry_run:
        print("\n[Dry run mode - exiting without splitting]")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading dataset statistics...")
    doc_stats, global_stats, doc_ids = load_dataset_stats(train_dir)

    print(f"Loaded {len(doc_ids)} docs, {global_stats['total_lines']} lines")
    print(f"Classes: {sorted(global_stats['class_counts'].keys())}")
    print(f"Relations: {sorted(global_stats['relation_counts'].keys())}")

    # Calculate target dev size
    target_dev_count = int(len(doc_ids) * args.dev_ratio)
    print(f"\nTarget dev size: {target_dev_count} docs ({args.dev_ratio:.0%})")

    # Run selection
    dev_ids = select_dev_docs(doc_stats, global_stats, doc_ids, target_dev_count, args.seed)
    train_ids = sorted(set(doc_ids) - set(dev_ids))

    print(f"\nFinal split: train={len(train_ids)}, dev={len(dev_ids)}")

    # Generate report
    report = generate_report(train_ids, dev_ids, doc_stats, global_stats, dataset_name)
    print_report_summary(report)

    # Save outputs
    train_file = os.path.join(output_dir, "train_doc_ids.json")
    dev_file = os.path.join(output_dir, "dev_doc_ids.json")
    report_file = os.path.join(output_dir, "split_report.json")

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_ids, f, indent=2)

    with open(dev_file, 'w', encoding='utf-8') as f:
        json.dump(dev_ids, f, indent=2)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {train_file}")
    print(f"  {dev_file}")
    print(f"  {report_file}")


if __name__ == "__main__":
    main()
