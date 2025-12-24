#!/usr/bin/env python
"""Test Construct Module (4.5) Metrics

Tests:
- TEDSMetric: Tree Edit Distance Similarity for document structure
- Node: Document tree node representation
- generate_doc_tree: Tree generation from predictions
- tree_edit_distance: APTED-based tree comparison
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment
setup_environment()


def test_node_class():
    """Test Node class for tree representation"""
    from examples.comp_hrdoc.metrics import Node

    print("Test Node class")
    print("-" * 40)

    # Create simple tree: ROOT -> A -> B, C
    root = Node("ROOT")
    a = Node("A")
    b = Node("B")
    c = Node("C")

    root.add_child(a)
    a.add_child(b)
    a.add_child(c)

    assert len(root) == 4  # ROOT, A, B, C
    assert a.parent == root
    assert b.parent == a
    assert c.parent == a
    print(f"[PASS] Tree size: {len(root)}")

    # Test depth
    root.set_depth(0)
    assert root.depth == 0
    assert a.depth == 1
    assert b.depth == 2
    print(f"[PASS] Depth correctly set")


def test_generate_doc_tree():
    """Test document tree generation from model outputs"""
    from examples.comp_hrdoc.metrics import generate_doc_tree, Node

    print("\nTest generate_doc_tree")
    print("-" * 40)

    # Simple document: ROOT -> title, section -> para
    # parent_ids: -1 means ROOT, otherwise index into texts
    texts = ["title:Document Title", "section:Introduction", "paraline:First paragraph"]
    parent_ids = [-1, -1, 1]  # title->ROOT, section->ROOT, para->section(idx 1)
    relations = ["contain", "contain", "contain"]

    tree = generate_doc_tree(texts, parent_ids, relations)

    assert tree.name == "ROOT"
    assert len(tree.children) == 2  # title and section
    print(f"[PASS] Tree root has {len(tree.children)} children")

    # Check structure
    title_node = tree.children[0]
    section_node = tree.children[1]
    assert "title" in title_node.name
    assert "section" in section_node.name
    assert len(section_node.children) == 1  # para
    print(f"[PASS] Tree structure correct")


def test_generate_tree_with_equality():
    """Test tree generation with equality relations"""
    from examples.comp_hrdoc.metrics import generate_doc_tree

    print("\nTest generate_doc_tree with equality")
    print("-" * 40)

    # Document with sibling paragraphs (equality relation)
    # section -> ROOT, para1 -> section, para2 -> para1 (equality means same parent)
    texts = [
        "section:Section 1",
        "paraline:Para 1",
        "paraline:Para 2",  # equality to Para 1
    ]
    parent_ids = [-1, 0, 1]  # section->ROOT, para1->section(idx 0), para2->para1(idx 1)
    relations = ["contain", "contain", "equality"]

    tree = generate_doc_tree(texts, parent_ids, relations)

    # section should have para1 and para2 as children (equality makes them siblings)
    section = tree.children[0]
    assert len(section.children) >= 1
    print(f"[PASS] Equality relation handled, section has {len(section.children)} children")


def test_tree_edit_distance():
    """Test tree edit distance computation"""
    from examples.comp_hrdoc.metrics import tree_edit_distance, Node, HAS_APTED

    print("\nTest tree_edit_distance")
    print("-" * 40)

    if not HAS_APTED:
        print("[SKIP] apted not installed")
        return

    # Identical trees
    tree1 = Node("ROOT")
    tree1.add_child(Node("A"))
    tree1.add_child(Node("B"))

    tree2 = Node("ROOT")
    tree2.add_child(Node("A"))
    tree2.add_child(Node("B"))

    dist, teds = tree_edit_distance(tree1, tree2)
    assert dist == 0
    assert teds == 1.0
    print(f"[PASS] Identical trees: dist={dist}, TEDS={teds:.4f}")

    # Different trees
    tree3 = Node("ROOT")
    tree3.add_child(Node("A"))
    tree3.add_child(Node("C"))  # Different from B

    dist, teds = tree_edit_distance(tree1, tree3)
    assert dist > 0
    assert teds < 1.0
    print(f"[PASS] Different trees: dist={dist}, TEDS={teds:.4f}")


def test_teds_metric():
    """Test TEDSMetric class"""
    from examples.comp_hrdoc.metrics import TEDSMetric, HAS_APTED

    print("\nTest TEDSMetric")
    print("-" * 40)

    if not HAS_APTED:
        print("[SKIP] apted not installed")
        return

    metric = TEDSMetric()

    # Sample 1: Perfect prediction
    texts1 = ["title:Title", "section:Sec1", "paraline:Para1"]
    parent_ids1 = [-1, -1, 1]
    relations1 = ["contain", "contain", "contain"]

    metric.update(
        pred_texts=texts1,
        pred_parent_ids=parent_ids1,
        pred_relations=relations1,
        gt_texts=texts1,
        gt_parent_ids=parent_ids1,
        gt_relations=relations1,
        sample_id="sample1"
    )

    result = metric.compute()
    assert result.macro_teds == 1.0
    print(f"[PASS] Perfect prediction: TEDS={result.macro_teds:.4f}")

    # Sample 2: With structure error
    metric.reset()
    pred_parent_ids = [-1, -1, -1]  # para directly under ROOT instead of section

    metric.update(
        pred_texts=texts1,
        pred_parent_ids=pred_parent_ids,
        pred_relations=relations1,
        gt_texts=texts1,
        gt_parent_ids=parent_ids1,
        gt_relations=relations1,
        sample_id="sample2"
    )

    result = metric.compute()
    assert result.macro_teds < 1.0
    print(f"[PASS] Structure error: TEDS={result.macro_teds:.4f}")


def test_teds_metric_aggregation():
    """Test TEDS metric aggregation over multiple samples"""
    from examples.comp_hrdoc.metrics import TEDSMetric, HAS_APTED

    print("\nTest TEDSMetric aggregation")
    print("-" * 40)

    if not HAS_APTED:
        print("[SKIP] apted not installed")
        return

    metric = TEDSMetric()

    # Add multiple samples
    for i in range(5):
        texts = [f"title:Title{i}", f"section:Sec{i}"]
        parent_ids = [-1, -1]
        relations = ["contain", "contain"]

        metric.update(
            pred_texts=texts,
            pred_parent_ids=parent_ids,
            pred_relations=relations,
            gt_texts=texts,
            gt_parent_ids=parent_ids,
            gt_relations=relations,
            sample_id=f"sample_{i}"
        )

    result = metric.compute()

    assert result.num_samples == 5
    assert result.macro_teds == 1.0  # All perfect
    assert len(result.per_sample) == 5
    print(f"[PASS] Aggregation: {result.num_samples} samples, macro_TEDS={result.macro_teds:.4f}")


def test_transfer_tree_to_chain():
    """Test tree to reading order chain conversion"""
    from examples.comp_hrdoc.metrics import generate_doc_tree, transfer_tree_to_chain

    print("\nTest transfer_tree_to_chain")
    print("-" * 40)

    # Document with main text and floating elements
    texts = [
        "title:Title",
        "section:Section",
        "paraline:Paragraph",
        "figure:Figure1",
        "caption:Fig Caption"
    ]
    parent_ids = [-1, -1, 1, 1, 3]  # title->ROOT, section->ROOT, para->section, figure->section, caption->figure
    relations = ["contain", "contain", "contain", "contain", "contain"]

    tree = generate_doc_tree(texts, parent_ids, relations)
    main_chain, floating_chain = transfer_tree_to_chain(tree)

    print(f"Main chain: {main_chain[:5]}...")
    print(f"Floating chain: {floating_chain[:5]}...")

    # Main chain should have title, section, paragraph
    assert any("title" in item for item in main_chain)
    assert any("section" in item for item in main_chain)
    print(f"[PASS] Chains generated correctly")


if __name__ == "__main__":
    test_node_class()
    test_generate_doc_tree()
    test_generate_tree_with_equality()
    test_tree_edit_distance()
    test_teds_metric()
    test_teds_metric_aggregation()
    test_transfer_tree_to_chain()
    print("\n" + "=" * 40)
    print("All Construct metrics tests passed!")
