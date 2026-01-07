#!/bin/bash

mkdir -p /data/LLM_group/layoutlmft/examples/comp_hrdoc/logs
LOG=/data/LLM_group/layoutlmft/examples/comp_hrdoc/logs/train_doc_$(date +%Y%m%d_%H%M%S).log

cmd=(python examples/comp_hrdoc/scripts/train_doc.py
  --env test
  --use-stage-features
  --stage-checkpoint  /data/LLM_group/layoutlmft/artifact/joint_train/stage1_hrds/checkpoint-1300
  --dataset hrds
  --max-regions 128
  --batch-size 1
  --log-steps 50
  --eval-steps 200
  --save-steps 100
  --new-exp
  --exp-name construct_train
)

{
  echo "===== $(date) ====="
  echo "CMD: nohup ${cmd[*]}"
  echo "===================="
} >> "$LOG"

nohup "${cmd[@]}" >> "$LOG" 2>&1 &
echo "Started. Log: $LOG"
tail -f "$LOG"

# 可选参数:
#   --fp16                        # 混合精度训练
#   --quick                       # 快速测试模式
#   --document-level              # 文档级别训练 (多页)
#   --toc-only                    # 仅 TOC 模式 (过滤 section)
#   --section-label-id 4          # section 标签 ID
#   --use-stage-features          # 使用 Stage 特征
#   --stage-checkpoint <path>     # Stage 模型检查点
#   --no-construct                # 禁用 Construct 模块 (仅 Order)
#   --use-semantic                # 启用语义分类头
#   --cls-weight 1.0              # 分类损失权重
#   --order-weight 1.0            # Order 损失权重
#   --construct-weight 1.0        # Construct 损失权重
