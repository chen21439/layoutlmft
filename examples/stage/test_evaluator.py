#!/usr/bin/env python
# coding=utf-8
"""
快速测试 Evaluator 和 Batch 抽象层

验证：
1. Batch 抽象层能正确包装 page/doc 级别数据
2. Predictor 能正确进行推理
3. Evaluator 能正确计算指标
"""

import os
import sys
import torch

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)

def test_batch_abstraction():
    """测试 Batch 抽象层"""
    print("=" * 60)
    print("Testing Batch Abstraction")
    print("=" * 60)

    from data.batch import Sample, PageLevelBatch, DocumentLevelBatch, wrap_batch

    # 测试页面级别 batch
    page_batch_raw = {
        "input_ids": torch.randint(0, 1000, (4, 512)),
        "bbox": torch.randint(0, 1000, (4, 512, 4)),
        "attention_mask": torch.ones(4, 512, dtype=torch.long),
        "line_ids": torch.randint(-1, 50, (4, 512)),
        "labels": torch.randint(0, 14, (4, 512)),
    }

    page_batch = wrap_batch(page_batch_raw)
    assert isinstance(page_batch, PageLevelBatch)
    assert len(page_batch) == 4
    assert not page_batch.is_document_level

    sample = page_batch[0]
    assert isinstance(sample, Sample)
    assert sample.input_ids.shape == (512,)
    assert not sample.is_document_level
    print("  PageLevelBatch: OK")

    # 测试文档级别 batch
    doc_batch_raw = {
        "input_ids": torch.randint(0, 1000, (10, 512)),  # 10 chunks total
        "bbox": torch.randint(0, 1000, (10, 512, 4)),
        "attention_mask": torch.ones(10, 512, dtype=torch.long),
        "line_ids": torch.randint(-1, 100, (10, 512)),
        "labels": torch.randint(0, 14, (10, 512)),
        "num_docs": 3,
        "chunks_per_doc": [3, 5, 2],  # 3 documents with 3, 5, 2 chunks
        "line_parent_ids": torch.randint(-1, 50, (3, 100)),
        "line_relations": torch.randint(-1, 3, (3, 100)),
    }

    doc_batch = wrap_batch(doc_batch_raw)
    assert isinstance(doc_batch, DocumentLevelBatch)
    assert len(doc_batch) == 3  # 3 documents
    assert doc_batch.is_document_level

    sample0 = doc_batch[0]
    assert isinstance(sample0, Sample)
    assert sample0.input_ids.shape == (3, 512)  # 3 chunks
    assert sample0.is_document_level
    assert sample0.num_chunks == 3

    sample1 = doc_batch[1]
    assert sample1.input_ids.shape == (5, 512)  # 5 chunks
    assert sample1.num_chunks == 5

    sample2 = doc_batch[2]
    assert sample2.input_ids.shape == (2, 512)  # 2 chunks
    assert sample2.num_chunks == 2

    print("  DocumentLevelBatch: OK")

    # 测试迭代
    count = 0
    for sample in doc_batch:
        count += 1
    assert count == 3
    print("  Iteration: OK")

    print("Batch Abstraction: PASSED")
    return True


def test_predictor_mock():
    """测试 Predictor（使用 mock 模型）"""
    print("=" * 60)
    print("Testing Predictor (mock)")
    print("=" * 60)

    from data.batch import Sample
    from engines.predictor import Predictor, PredictionOutput

    # 创建 mock 模型
    class MockStage1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, input_ids, bbox, attention_mask, image=None, output_hidden_states=False):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, 14)
            # 模拟真实 LayoutXLM：hidden_states 包含视觉 tokens (7x7=49)
            visual_tokens = 49
            hidden_states = [torch.randn(batch_size, seq_len + visual_tokens, 768)]

            class Output:
                pass
            out = Output()
            out.logits = logits
            out.hidden_states = hidden_states
            return out

    class MockFeatureExtractor(torch.nn.Module):
        def extract_line_features(self, hidden, line_ids, pooling="mean"):
            batch_size = hidden.shape[0]
            max_lines = 50
            features = torch.randn(batch_size, max_lines, 768)
            mask = torch.ones(batch_size, max_lines)
            mask[:, 30:] = 0  # 假设只有 30 行
            return features, mask

    class MockStage3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, features, mask, return_gru_hidden=False):
            batch_size, max_lines, hidden_size = features.shape
            # parent_logits: [B, L+1, L+1]
            parent_logits = torch.randn(batch_size, max_lines + 1, max_lines + 1)
            if return_gru_hidden:
                gru_hidden = torch.randn(batch_size, max_lines + 1, 512)
                return parent_logits, gru_hidden
            return parent_logits

    class MockStage4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, parent_feat, child_feat):
            batch_size = parent_feat.shape[0]
            return torch.randn(batch_size, 3)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = MockStage1()
            self.feature_extractor = MockFeatureExtractor()
            self.stage3 = MockStage3()
            self.stage4 = MockStage4()
            self.use_gru = True

    model = MockModel()
    predictor = Predictor(model, device=torch.device("cpu"))

    # 测试页面级别推理
    sample = Sample(
        input_ids=torch.randint(0, 1000, (512,)),
        bbox=torch.randint(0, 1000, (512, 4)),
        attention_mask=torch.ones(512, dtype=torch.long),
        line_ids=torch.randint(0, 30, (512,)),
        is_document_level=False,
    )

    output = predictor.predict(sample)
    assert isinstance(output, PredictionOutput)
    assert output.num_lines > 0
    print(f"  Page-level prediction: {output.num_lines} lines")

    # 测试文档级别推理
    sample_doc = Sample(
        input_ids=torch.randint(0, 1000, (3, 512)),  # 3 chunks
        bbox=torch.randint(0, 1000, (3, 512, 4)),
        attention_mask=torch.ones(3, 512, dtype=torch.long),
        line_ids=torch.randint(0, 50, (3, 512)),
        num_chunks=3,
        is_document_level=True,
    )

    output_doc = predictor.predict(sample_doc)
    assert isinstance(output_doc, PredictionOutput)
    assert output_doc.num_lines > 0
    print(f"  Document-level prediction: {output_doc.num_lines} lines")

    print("Predictor: PASSED")
    return True


def test_evaluator_mock():
    """测试 Evaluator（使用 mock 模型和数据）"""
    print("=" * 60)
    print("Testing Evaluator (mock)")
    print("=" * 60)

    from data.batch import wrap_batch
    from engines.evaluator import Evaluator, EvaluationOutput

    # 创建 mock 模型（同上）
    class MockStage1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, input_ids, bbox, attention_mask, image=None, output_hidden_states=False):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, 14)
            # 模拟真实 LayoutXLM：hidden_states 包含视觉 tokens (7x7=49)
            visual_tokens = 49
            hidden_states = [torch.randn(batch_size, seq_len + visual_tokens, 768)]

            class Output:
                pass
            out = Output()
            out.logits = logits
            out.hidden_states = hidden_states
            return out

    class MockFeatureExtractor(torch.nn.Module):
        def extract_line_features(self, hidden, line_ids, pooling="mean"):
            batch_size = hidden.shape[0]
            max_lines = 50
            features = torch.randn(batch_size, max_lines, 768)
            mask = torch.ones(batch_size, max_lines)
            mask[:, 20:] = 0
            return features, mask

    class MockStage3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, features, mask, return_gru_hidden=False):
            batch_size, max_lines, hidden_size = features.shape
            parent_logits = torch.randn(batch_size, max_lines + 1, max_lines + 1)
            if return_gru_hidden:
                gru_hidden = torch.randn(batch_size, max_lines + 1, 512)
                return parent_logits, gru_hidden
            return parent_logits

    class MockStage4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(10, 10)

        def forward(self, parent_feat, child_feat):
            batch_size = parent_feat.shape[0]
            return torch.randn(batch_size, 3)

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stage1 = MockStage1()
            self.feature_extractor = MockFeatureExtractor()
            self.stage3 = MockStage3()
            self.stage4 = MockStage4()
            self.use_gru = True

    # 创建 mock dataloader
    def mock_dataloader():
        for _ in range(2):  # 2 batches
            yield {
                "input_ids": torch.randint(0, 1000, (2, 512)),
                "bbox": torch.randint(0, 1000, (2, 512, 4)),
                "attention_mask": torch.ones(2, 512, dtype=torch.long),
                "line_ids": torch.randint(0, 20, (2, 512)),
                "labels": torch.randint(0, 14, (2, 512)),
                "line_parent_ids": torch.randint(0, 20, (2, 20)),
                "line_relations": torch.randint(0, 3, (2, 20)),
            }

    model = MockModel()
    evaluator = Evaluator(model, device=torch.device("cpu"))

    output = evaluator.evaluate(mock_dataloader(), verbose=False)

    assert isinstance(output, EvaluationOutput)
    assert output.num_samples > 0
    print(f"  Samples evaluated: {output.num_samples}")
    print(f"  Lines evaluated: {output.num_lines}")
    print(f"  Line accuracy: {output.line_accuracy:.2%}")
    print(f"  Parent accuracy: {output.parent_accuracy:.2%}")
    print(f"  Relation accuracy: {output.relation_accuracy:.2%}")

    print("Evaluator: PASSED")
    return True


def main():
    print("\n" + "=" * 60)
    print("Running Evaluator and Batch Abstraction Tests")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Batch Abstraction", test_batch_abstraction()))
    except Exception as e:
        print(f"Batch Abstraction: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("Batch Abstraction", False))

    try:
        results.append(("Predictor", test_predictor_mock()))
    except Exception as e:
        print(f"Predictor: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("Predictor", False))

    try:
        results.append(("Evaluator", test_evaluator_mock()))
    except Exception as e:
        print(f"Evaluator: FAILED - {e}")
        import traceback
        traceback.print_exc()
        results.append(("Evaluator", False))

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
