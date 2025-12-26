#!/usr/bin/env python
"""Test DOCPipeline (4.2 + 4.3 integration)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import DOCPipeline, predict_reading_order


def test_predict_reading_order():
    print("Test predict_reading_order")
    device = get_device()

    num_regions = 5
    order_logits = torch.full((1, num_regions, num_regions), -10.0, device=device)
    order_logits[0, 0, 1] = 5.0  # 0->1
    order_logits[0, 1, 2] = 5.0  # 1->2
    order_logits[0, 2, 3] = 5.0  # 2->3
    order_logits[0, 3, 4] = 5.0  # 3->4
    order_logits[0, 4, 4] = 5.0  # 4->END

    mask = torch.ones(1, num_regions, dtype=torch.bool, device=device)
    reading_order = predict_reading_order(order_logits, mask)

    assert reading_order[0].tolist() == [0, 1, 2, 3, 4]
    print(f"[PASS] Order={reading_order[0].tolist()}")


def test_doc_pipeline():
    print("Test DOCPipeline (Detect + Order)")
    device = get_device()
    print_gpu_info()

    batch_size, num_lines, input_size = 2, 25, 768
    model = DOCPipeline(
        input_size=input_size, hidden_size=768, proj_size=2048, mlp_hidden=1024,
        detect_num_heads=12, detect_num_layers=1, num_roles=10,
        order_num_heads=12, order_num_layers=3, num_relations=3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    line_features = torch.randn(batch_size, num_lines, input_size, device=device)
    line_bboxes = torch.rand(batch_size, num_lines, 4, device=device) * 1000
    line_mask = torch.ones(batch_size, num_lines, dtype=torch.bool, device=device)

    successor_labels = torch.zeros(batch_size, num_lines, dtype=torch.long, device=device)
    for b in range(batch_size):
        for i in range(num_lines - 1):
            successor_labels[b, i] = i + 1
        successor_labels[b, -1] = -1
    role_labels = torch.randint(0, 10, (batch_size, num_lines), device=device)

    outputs = model(line_features, line_bboxes, line_mask, successor_labels, role_labels)
    print(f"Loss: {outputs['loss'].item():.4f}")

    predictions = model.predict(line_features, line_bboxes, line_mask)
    print(f"Regions: {[len(r) for r in predictions['regions']]}")
    print(f"[PASS] DOCPipeline test passed")


if __name__ == "__main__":
    test_predict_reading_order()
    test_doc_pipeline()
