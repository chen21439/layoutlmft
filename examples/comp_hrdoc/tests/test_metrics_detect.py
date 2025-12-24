#!/usr/bin/env python
"""Test Detect Module (4.2) Metrics

Tests:
- ClassificationMetric: Logical role classification F1
- IntraRegionOrderMetric: Intra-region reading order accuracy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment
setup_environment()


def test_classification_metric():
    """Test ClassificationMetric for logical role classification"""
    from examples.comp_hrdoc.metrics import (
        ClassificationMetric, ClassificationResult,
        CLASS2ID, ID2CLASS, normalize_class, class_to_id
    )

    print("Test ClassificationMetric")
    print("-" * 40)

    # Test class normalization
    assert normalize_class("sec1") == "section"
    assert normalize_class("sec2") == "section"
    assert normalize_class("figcap") == "caption"
    assert normalize_class("tab") == "table"
    assert normalize_class("para") == "paraline"
    print("[PASS] Class normalization works correctly")

    # Test class_to_id
    assert class_to_id("title") == 0
    assert class_to_id("sec1") == 4  # section
    assert class_to_id("figcap") == 9  # caption
    print("[PASS] class_to_id works correctly")

    # Test metric computation
    metric = ClassificationMetric(num_classes=14)

    # Simulate predictions
    preds = [0, 1, 2, 3, 4, 4, 5, 6, 6, 7]  # some correct, some wrong
    labels = [0, 1, 2, 3, 4, 5, 5, 6, 7, 7]

    metric.update(preds, labels)
    result = metric.compute()

    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Micro F1: {result.micro_f1:.4f}")

    assert result.num_samples == 10
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.macro_f1 <= 1
    print("[PASS] ClassificationMetric computes correctly")

    # Test from class names
    metric.reset()
    pred_classes = ["title", "author", "sec1", "para", "figcap"]
    gt_classes = ["title", "author", "sec2", "paraline", "caption"]

    metric.update_from_classes(pred_classes, gt_classes)
    result = metric.compute()

    print(f"From classes - Accuracy: {result.accuracy:.4f}")
    assert result.accuracy == 1.0  # All should match after normalization
    print("[PASS] ClassificationMetric handles class names correctly")


def test_intra_region_order_metric():
    """Test IntraRegionOrderMetric for text line grouping"""
    from examples.comp_hrdoc.metrics import IntraRegionOrderMetric

    print("\nTest IntraRegionOrderMetric")
    print("-" * 40)

    metric = IntraRegionOrderMetric()

    # Test successor accuracy
    # Lines: 0->1->2, 3->4, 5 (standalone)
    gt_successors = [1, 2, -1, 4, -1, -1]
    pred_successors = [1, 2, -1, 4, -1, -1]  # Perfect prediction

    metric.update(pred_successors, gt_successors)
    result = metric.compute()

    assert result['successor_accuracy'] == 1.0
    print(f"[PASS] Perfect successor prediction: {result['successor_accuracy']:.4f}")

    # Test with some errors
    metric.reset()
    pred_successors = [1, 3, -1, 4, -1, -1]  # Error: 1->3 instead of 1->2
    metric.update(pred_successors, gt_successors)
    result = metric.compute()

    assert result['successor_accuracy'] < 1.0
    print(f"[PASS] With errors: {result['successor_accuracy']:.4f}")

    # Test group accuracy
    metric.reset()
    gt_groups = [[0, 1, 2], [3, 4], [5]]
    pred_groups = [[0, 1, 2], [3, 4], [5]]  # Perfect

    metric.update_groups(pred_groups, gt_groups)
    result = metric.compute()

    assert result['group_accuracy'] == 1.0
    print(f"[PASS] Perfect group prediction: {result['group_accuracy']:.4f}")

    # Test with group errors
    metric.reset()
    pred_groups = [[0, 1], [2, 3, 4], [5]]  # Wrong grouping
    metric.update_groups(pred_groups, gt_groups)
    result = metric.compute()

    assert result['group_accuracy'] < 1.0
    print(f"[PASS] Group errors detected: {result['group_accuracy']:.4f}")


def test_confusion_matrix():
    """Test confusion matrix generation"""
    from examples.comp_hrdoc.metrics import ClassificationMetric, ID2CLASS

    print("\nTest Confusion Matrix")
    print("-" * 40)

    metric = ClassificationMetric(num_classes=14)

    # Create predictions with known pattern
    # 10 correct title, 5 title predicted as section
    preds = [0] * 10 + [4] * 5 + [1] * 10  # 10 title, 5 section, 10 author
    labels = [0] * 15 + [1] * 10  # 15 title, 10 author

    metric.update(preds, labels)
    result = metric.compute()

    cm = result.confusion_matrix
    assert cm[0, 0] == 10  # 10 correct title
    assert cm[0, 4] == 5   # 5 title predicted as section
    assert cm[1, 1] == 10  # 10 correct author

    print(f"[PASS] Confusion matrix correct")
    print(f"  Title precision: {result.per_class_precision['title']:.4f}")
    print(f"  Title recall: {result.per_class_recall['title']:.4f}")
    print(f"  Title F1: {result.per_class_f1['title']:.4f}")


if __name__ == "__main__":
    test_classification_metric()
    test_intra_region_order_metric()
    test_confusion_matrix()
    print("\n" + "=" * 40)
    print("All Detect metrics tests passed!")
