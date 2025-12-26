#!/usr/bin/env python
"""Test 4.3.4: RelationTypeHead (Eq. 16)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.utils.config import setup_environment, get_device, print_gpu_info
setup_environment()

import torch
from examples.comp_hrdoc.models.order import RelationTypeHead


def test_relation_head():
    print("Test 4.3.4: RelationTypeHead")
    device = get_device()
    print_gpu_info()

    batch_size, num_regions, hidden_size = 2, 6, 768
    num_relations = 3
    model = RelationTypeHead(
        hidden_size=hidden_size, proj_size=2048, num_relations=num_relations
    ).to(device)

    features = torch.randn(batch_size, num_regions, hidden_size, device=device)
    relation_logits = model(features)

    assert model.head_proj.out_features == 2048, f"FC_q should be 2048"
    assert model.tail_proj.out_features == 2048, f"FC_k should be 2048"
    assert relation_logits.shape == (batch_size, num_regions, num_regions, num_relations)
    print(f"[PASS] FC_q=FC_k=2048, output={relation_logits.shape}")


if __name__ == "__main__":
    test_relation_head()
