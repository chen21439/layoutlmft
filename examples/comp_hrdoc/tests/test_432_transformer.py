#!/usr/bin/env python
"""Test 4.3.2: OrderTransformerEncoder (3-layer)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import OrderTransformerEncoder


def test_transformer_encoder():
    print("Test 4.3.2: OrderTransformerEncoder")
    device = get_device()
    print_gpu_info()

    batch_size, num_regions, hidden_size = 2, 8, 768
    model = OrderTransformerEncoder(
        hidden_size=hidden_size, num_heads=12, num_layers=3, ffn_dim=2048
    ).to(device)

    features = torch.randn(batch_size, num_regions, hidden_size, device=device)
    mask = torch.ones(batch_size, num_regions, dtype=torch.bool, device=device)

    enhanced = model(features, mask)

    num_layers = len(model.encoder.layers)
    num_heads = model.encoder.layers[0].self_attn.num_heads

    assert num_layers == 3, f"Should have 3 layers, got {num_layers}"
    assert num_heads == 12, f"Should have 12 heads, got {num_heads}"
    print(f"[PASS] layers=3, heads=12, output={enhanced.shape}")


if __name__ == "__main__":
    test_transformer_encoder()
