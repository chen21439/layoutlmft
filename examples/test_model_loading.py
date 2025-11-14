#!/usr/bin/env python
# coding=utf-8
"""
测试模型加载功能
"""

import torch
from layoutlmft.models.relation_classifier import SimpleRelationClassifier
from layoutlmft.models.model_utils import (
    load_best_model,
    load_checkpoint,
    list_checkpoints,
    print_checkpoint_info
)


def main():
    print("=" * 60)
    print("关系分类模型加载测试")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "./output/relation_classifier"

    # 1. 列出所有checkpoints
    print("\n[1] 查看所有checkpoints:")
    print_checkpoint_info(checkpoint_dir)

    # 2. 加载最佳模型
    print("\n[2] 加载最佳模型:")
    model = SimpleRelationClassifier(hidden_size=768, use_geometry=False)
    model, metrics = load_best_model(
        model,
        checkpoint_dir=checkpoint_dir,
        device=device,
        return_metrics=True
    )
    print(f"模型已加载！")
    print(f"  - Epoch: {metrics['epoch']}")
    print(f"  - 验证F1: {metrics['val_f1']:.4f}")
    print(f"  - 精确率: {metrics.get('precision', 0):.4f}")
    print(f"  - 召回率: {metrics.get('recall', 0):.4f}")

    # 3. 演示如何使用加载的模型
    print("\n[3] 测试模型推理:")
    model.eval()
    with torch.no_grad():
        # 创建虚拟输入
        parent_feat = torch.randn(1, 768).to(device)
        child_feat = torch.randn(1, 768).to(device)

        logits = model(parent_feat, child_feat)
        probs = torch.softmax(logits, dim=1)

        print(f"  输入形状: parent={parent_feat.shape}, child={child_feat.shape}")
        print(f"  输出logits: {logits[0].cpu().numpy()}")
        print(f"  预测概率: None={probs[0,0]:.4f}, Parent={probs[0,1]:.4f}")

    # 4. 演示如何从特定checkpoint恢复训练
    print("\n[4] 从特定checkpoint恢复训练:")
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) > 1:
        # 选择第一个非best的checkpoint
        for ckpt in checkpoints:
            if 'checkpoint-epoch' in ckpt['filename']:
                print(f"  加载: {ckpt['filename']}")
                new_model = SimpleRelationClassifier(hidden_size=768)
                optimizer = torch.optim.Adam(new_model.parameters())

                new_model, optimizer, epoch, metrics = load_checkpoint(
                    new_model,
                    optimizer,
                    checkpoint_path=ckpt['path'],
                    device=device
                )
                print(f"  ✓ 成功恢复到Epoch {epoch}")
                print(f"  ✓ 可以从Epoch {epoch+1} 继续训练")
                break

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
