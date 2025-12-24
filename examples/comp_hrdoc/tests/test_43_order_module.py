#!/usr/bin/env python
"""Test Complete OrderModule (4.3)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import OrderModule, OrderLoss


def test_order_module():
    print("Test OrderModule (complete 4.3)")
    device = get_device()
    print_gpu_info()

    batch_size, num_lines, hidden_size = 2, 30, 768
    model = OrderModule(
        hidden_size=hidden_size, num_categories=10, num_heads=12,
        num_layers=3, ffn_dim=2048, proj_size=2048, mlp_hidden=1024,
        num_relations=3, use_spatial=True
    ).to(device)

    line_features = torch.randn(batch_size, num_lines, hidden_size, device=device)
    regions = [
        [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9, 10, 11], [12, 13]],
        [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10]],
    ]
    region_roles = [[1, 2, 3, 4, 5], [1, 2, 3, 4]]
    max_regions = 5
    region_bboxes = torch.rand(batch_size, max_regions, 4, device=device) * 1000
    line_mask = torch.ones(batch_size, num_lines, dtype=torch.bool, device=device)

    outputs = model(line_features, regions, region_roles, region_bboxes, line_mask)

    expected_keys = ['region_features', 'enhanced_features', 'order_logits',
                     'relation_logits', 'object_mask', 'object_bboxes']
    for key in expected_keys:
        assert key in outputs, f"Missing: {key}"

    print(f"[PASS] OrderModule outputs: {list(outputs.keys())}")


def test_order_loss():
    print("Test OrderLoss")
    device = get_device()

    batch_size, num_regions, num_relations = 2, 5, 3
    loss_fn = OrderLoss(order_weight=1.0, relation_weight=0.5)

    order_logits = torch.randn(batch_size, num_regions, num_regions, device=device)
    relation_logits = torch.randn(batch_size, num_regions, num_regions, num_relations, device=device)
    order_labels = torch.tensor([[1, 2, 3, 4, -1], [1, 2, 3, -1, -1]], device=device)
    relation_labels = torch.zeros(batch_size, num_regions, num_regions, dtype=torch.long, device=device)
    mask = torch.ones(batch_size, num_regions, dtype=torch.bool, device=device)
    mask[1, 4] = False

    loss_dict = loss_fn(order_logits, relation_logits, order_labels, relation_labels, mask)

    assert loss_dict['loss'] > 0
    print(f"[PASS] Loss={loss_dict['loss'].item():.4f}")


if __name__ == "__main__":
    test_order_module()
    test_order_loss()
