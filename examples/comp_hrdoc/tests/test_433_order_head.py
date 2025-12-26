#!/usr/bin/env python
"""Test 4.3.3: InterRegionOrderHead"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import InterRegionOrderHead


def test_order_head():
    print("Test 4.3.3: InterRegionOrderHead")
    device = get_device()
    print_gpu_info()

    batch_size, num_regions, hidden_size = 2, 6, 768
    model = InterRegionOrderHead(
        hidden_size=hidden_size, proj_size=2048, mlp_hidden=1024, use_spatial=True
    ).to(device)

    features = torch.randn(batch_size, num_regions, hidden_size, device=device)
    bbox = torch.rand(batch_size, num_regions, 4, device=device) * 1000
    mask = torch.ones(batch_size, num_regions, dtype=torch.bool, device=device)

    order_logits = model(features, bbox, mask)

    assert model.head_proj.out_features == 2048, f"FC_q should be 2048"
    assert model.dep_proj.out_features == 2048, f"FC_k should be 2048"
    assert torch.all(order_logits[0].diag() < -1e8), "Diagonal should be masked"
    print(f"[PASS] FC_q=FC_k=2048, output={order_logits.shape}")


if __name__ == "__main__":
    test_order_head()
