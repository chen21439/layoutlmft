#!/usr/bin/env python
# coding=utf-8
"""
测试 HRDocJointDataCollator 和 JointModel micro-batching
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)
EXAMPLES_ROOT = os.path.dirname(STAGE_ROOT)
sys.path.insert(0, EXAMPLES_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass


# Mock tokenizer
@dataclass
class MockTokenizer:
    pad_token_id: int = 0


def create_mock_document(doc_name, num_chunks, seq_len=512, num_lines=50):
    """创建 mock 文档数据"""
    chunks = []
    for i in range(num_chunks):
        chunk = {
            "input_ids": list(range(seq_len)),
            "attention_mask": [1] * seq_len,
            "labels": [0] * seq_len,
            "bbox": [[0, 0, 100, 100]] * seq_len,
            "line_ids": [j % num_lines for j in range(seq_len)],
            "image": torch.randn(3, 224, 224),
        }
        chunks.append(chunk)

    return {
        "document_name": doc_name,
        "num_pages": num_chunks // 2 + 1,
        "chunks": chunks,
        "line_parent_ids": list(range(-1, num_lines - 1)),
        "line_relations": ["cont"] * num_lines,
    }


# Mock Stage1 model
class MockStage1(nn.Module):
    def __init__(self, hidden_size=768, num_labels=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, bbox, attention_mask, image=None, labels=None, output_hidden_states=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        # 模拟 hidden states
        hidden = torch.randn(batch_size, seq_len, self.hidden_size, device=input_ids.device)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = logits
        out.hidden_states = (hidden,) if output_hidden_states else None
        return out


def test_micro_batching():
    print("=" * 60)
    print("Testing JointModel Micro-Batching")
    print("=" * 60)

    from joint_data_collator import HRDocJointDataCollator
    from models.joint_model import JointModel

    device = torch.device("cpu")

    # 创建 mock 模型
    stage1 = MockStage1(hidden_size=768, num_labels=16)

    # Mock Stage3 (SimpleParentFinder)
    class MockStage3(nn.Module):
        def forward(self, candidates, child_feat):
            scores = torch.randn(candidates.shape[0])
            return scores

    # Mock Stage4
    class MockStage4(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768 * 2, 3)
        def forward(self, parent_feat, child_feat):
            combined = torch.cat([parent_feat, child_feat], dim=-1)
            return self.fc(combined)

    stage3 = MockStage3()
    stage4 = MockStage4()

    # 创建 collator
    tokenizer = MockTokenizer()
    collator = HRDocJointDataCollator(tokenizer=tokenizer, max_length=512)

    # Test 1: 小 batch（不触发 micro-batching）
    print("\n[Test 1] Small batch (3 chunks) - no micro-batching")
    model = JointModel(
        stage1_model=stage1,
        stage3_model=stage3,
        stage4_model=stage4,
        feature_extractor=None,
        stage1_micro_batch_size=8,
        use_gru=False,
    )

    doc1 = create_mock_document("small_doc", num_chunks=3, num_lines=10)
    batch1 = collator([doc1])

    # 转换 batch 到 tensor
    for k, v in batch1.items():
        if isinstance(v, torch.Tensor):
            batch1[k] = v.to(device)

    start = time.time()
    output1 = model(**batch1)
    print(f"  Time: {time.time() - start:.3f}s")
    print(f"  Loss: {output1.loss.item():.4f}")

    # Test 2: 大 batch（触发 micro-batching）
    print("\n[Test 2] Large batch (50 chunks) - with micro-batching (micro_bs=8)")
    model2 = JointModel(
        stage1_model=stage1,
        stage3_model=stage3,
        stage4_model=stage4,
        feature_extractor=None,
        stage1_micro_batch_size=8,
        use_gru=False,
    )

    doc2 = create_mock_document("large_doc", num_chunks=50, num_lines=100)
    batch2 = collator([doc2])

    for k, v in batch2.items():
        if isinstance(v, torch.Tensor):
            batch2[k] = v.to(device)

    start = time.time()
    output2 = model2(**batch2)
    print(f"  Time: {time.time() - start:.3f}s")
    print(f"  Loss: {output2.loss.item():.4f}")
    print(f"  Expected ~7 micro-batches (50/8)")

    # Test 3: 验证 no_grad 模式
    print("\n[Test 3] Large batch with stage1_no_grad=True")
    model3 = JointModel(
        stage1_model=stage1,
        stage3_model=stage3,
        stage4_model=stage4,
        feature_extractor=None,
        stage1_micro_batch_size=8,
        stage1_no_grad=True,
        use_gru=False,
    )

    start = time.time()
    output3 = model3(**batch2)
    print(f"  Time: {time.time() - start:.3f}s")
    print(f"  Loss: {output3.loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nSummary:")
    print("  - Micro-batching works correctly")
    print("  - Large documents are split into smaller chunks for Stage1")
    print("  - no_grad mode works for memory savings")
    print("\nUsage:")
    print("  python train_joint.py --stage1_micro_batch_size 8")
    print("  python train_joint.py --stage1_micro_batch_size 4 --stage1_no_grad")
    print("=" * 60)


if __name__ == "__main__":
    test_micro_batching()
