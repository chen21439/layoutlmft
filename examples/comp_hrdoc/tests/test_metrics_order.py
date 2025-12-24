#!/usr/bin/env python
"""Test Order Module (4.3) Metrics

Tests:
- ReadingOrderMetric: Inter-region reading order TEDS
- sequence_edit_distance: Sequence comparison
- min_edit_distance_between_groups: Floating element grouping
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment
setup_environment()


def test_sequence_edit_distance():
    """Test sequence edit distance computation"""
    from examples.comp_hrdoc.metrics import sequence_edit_distance

    print("Test sequence_edit_distance")
    print("-" * 40)

    # Identical sequences
    seq1 = ["a", "b", "c"]
    seq2 = ["a", "b", "c"]
    dist, sim = sequence_edit_distance(seq1, seq2)
    assert dist == 0
    assert sim == 1.0
    print(f"[PASS] Identical: dist={dist}, sim={sim:.4f}")

    # One substitution
    seq1 = ["a", "b", "c"]
    seq2 = ["a", "x", "c"]
    dist, sim = sequence_edit_distance(seq1, seq2)
    assert dist == 1
    print(f"[PASS] One substitution: dist={dist}, sim={sim:.4f}")

    # Insertion/deletion
    seq1 = ["a", "b", "c"]
    seq2 = ["a", "c"]
    dist, sim = sequence_edit_distance(seq1, seq2)
    assert dist == 1
    print(f"[PASS] One deletion: dist={dist}, sim={sim:.4f}")

    # Completely different
    seq1 = ["a", "b", "c"]
    seq2 = ["x", "y", "z"]
    dist, sim = sequence_edit_distance(seq1, seq2)
    assert dist == 3
    assert sim == 0.0
    print(f"[PASS] Completely different: dist={dist}, sim={sim:.4f}")

    # Empty sequences
    dist, sim = sequence_edit_distance([], [])
    assert dist == 0
    assert sim == 1.0
    print(f"[PASS] Empty: dist={dist}, sim={sim:.4f}")


def test_split_chain_by_tag():
    """Test chain splitting"""
    from examples.comp_hrdoc.metrics import split_chain_by_tag

    print("\nTest split_chain_by_tag")
    print("-" * 40)

    chain = ["a", "b", "<p>", "c", "d", "e", "<p>", "f"]
    groups = split_chain_by_tag(chain, "<p>")

    assert len(groups) == 3
    assert groups[0] == ["a", "b"]
    assert groups[1] == ["c", "d", "e"]
    assert groups[2] == ["f"]
    print(f"[PASS] Split into {len(groups)} groups: {groups}")

    # No tags
    chain = ["a", "b", "c"]
    groups = split_chain_by_tag(chain, "<p>")
    assert len(groups) == 1
    assert groups[0] == ["a", "b", "c"]
    print(f"[PASS] No tags: {groups}")


def test_min_edit_distance_between_groups():
    """Test group matching with Hungarian algorithm"""
    from examples.comp_hrdoc.metrics import min_edit_distance_between_groups

    print("\nTest min_edit_distance_between_groups")
    print("-" * 40)

    # Identical groups
    groups1 = [["a", "b"], ["c", "d"]]
    groups2 = [["a", "b"], ["c", "d"]]
    dist, sim = min_edit_distance_between_groups(groups1, groups2)
    assert dist == 0
    assert sim == 1.0
    print(f"[PASS] Identical groups: dist={dist}, sim={sim:.4f}")

    # Permuted groups (should still match well)
    groups1 = [["a", "b"], ["c", "d"]]
    groups2 = [["c", "d"], ["a", "b"]]
    dist, sim = min_edit_distance_between_groups(groups1, groups2)
    assert dist == 0
    assert sim == 1.0
    print(f"[PASS] Permuted groups: dist={dist}, sim={sim:.4f}")

    # Partially different
    groups1 = [["a", "b"], ["c", "d"]]
    groups2 = [["a", "x"], ["c", "d"]]
    dist, sim = min_edit_distance_between_groups(groups1, groups2)
    assert 0 < dist < 4
    print(f"[PASS] Partial difference: dist={dist}, sim={sim:.4f}")


def test_reading_order_metric():
    """Test ReadingOrderMetric"""
    from examples.comp_hrdoc.metrics import ReadingOrderMetric

    print("\nTest ReadingOrderMetric")
    print("-" * 40)

    metric = ReadingOrderMetric()

    # Perfect prediction using chains directly
    gt_main = ["title:Title", "section:Sec1", "para:Para1", "<p>"]
    pred_main = ["title:Title", "section:Sec1", "para:Para1", "<p>"]

    gt_floating = [["figure:Fig1", "caption:Cap1"]]
    pred_floating = [["figure:Fig1", "caption:Cap1"]]

    metric.update_from_chains(
        pred_main_chain=pred_main,
        gt_main_chain=gt_main,
        pred_floating_groups=pred_floating,
        gt_floating_groups=gt_floating,
        sample_id="sample1"
    )

    result = metric.compute()

    assert result.macro_teds == 1.0
    assert result.macro_teds_floating == 1.0
    print(f"[PASS] Perfect prediction:")
    print(f"  Main TEDS: {result.macro_teds:.4f}")
    print(f"  Floating TEDS: {result.macro_teds_floating:.4f}")

    # Test with errors
    metric.reset()
    pred_main_wrong = ["title:Title", "para:Para1", "section:Sec1", "<p>"]  # Wrong order

    metric.update_from_chains(
        pred_main_chain=pred_main_wrong,
        gt_main_chain=gt_main,
        sample_id="sample2"
    )

    result = metric.compute()
    assert result.macro_teds < 1.0
    print(f"[PASS] With order errors: TEDS={result.macro_teds:.4f}")


def test_reading_order_from_tree():
    """Test ReadingOrderMetric with tree input"""
    from examples.comp_hrdoc.metrics import ReadingOrderMetric, generate_doc_tree

    print("\nTest ReadingOrderMetric from tree")
    print("-" * 40)

    metric = ReadingOrderMetric()

    # Simple document structure
    texts = ["title:Doc Title", "section:Introduction", "paraline:Paragraph 1"]
    parent_ids = [0, 0, 1]  # title->ROOT, section->ROOT, para->section
    relations = ["contain", "contain", "contain"]

    # Identical prediction
    metric.update(
        pred_texts=texts,
        pred_parent_ids=parent_ids,
        pred_relations=relations,
        gt_texts=texts,
        gt_parent_ids=parent_ids,
        gt_relations=relations,
        sample_id="tree_sample"
    )

    result = metric.compute()
    assert result.macro_teds == 1.0
    print(f"[PASS] Tree-based evaluation: TEDS={result.macro_teds:.4f}")


if __name__ == "__main__":
    test_sequence_edit_distance()
    test_split_chain_by_tag()
    test_min_edit_distance_between_groups()
    test_reading_order_metric()
    test_reading_order_from_tree()
    print("\n" + "=" * 40)
    print("All Order metrics tests passed!")
