#!/usr/bin/env python
"""Test 4.3.1: TextRegionAttentionFusion (Eq. 10-12)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import TextRegionAttentionFusion


def test_attention_fusion():
    print("Test 4.3.1: TextRegionAttentionFusion")
    device = get_device()
    print_gpu_info()

    batch_size, num_lines, hidden_size = 2, 20, 768
    model = TextRegionAttentionFusion(hidden_size=hidden_size, attention_hidden=1024).to(device)

    line_features = torch.randn(batch_size, num_lines, hidden_size, device=device)
    regions = [
        [[0, 1, 2], [3, 4], [5, 6, 7, 8], [9]],
        [[0, 1], [2, 3, 4, 5], [6, 7], [8, 9, 10]],
    ]
    line_mask = torch.ones(batch_size, num_lines, dtype=torch.bool, device=device)

    region_features, region_mask = model(line_features, regions, line_mask)

    assert model.fc2.out_features == 1024, f"FC2 should be 1024, got {model.fc2.out_features}"
    assert region_features.shape == (batch_size, 4, hidden_size)
    print(f"[PASS] FC2=1024, output={region_features.shape}")


if __name__ == "__main__":
    test_attention_fusion()
