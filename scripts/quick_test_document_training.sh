#!/bin/bash
# 小规模文档级别训练测试
# 目的：快速验证代码正确性，不用等待完整训练
# 测试规模：10个文档，1个epoch

set -e  # 遇到错误立即退出

echo "======================================================"
echo "小规模文档级别训练测试"
echo "======================================================"
echo ""

# 激活环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate layoutlmv2

cd /root/code/layoutlmft

# ========================================
# 步骤1: 提取10个文档的特征
# ========================================
echo "步骤 1/3: 提取文档级别特征 (10个文档)"
echo "------------------------------------------------------"

export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc_quick_test"
export LAYOUTLMFT_NUM_SAMPLES="10"  # 只处理10个文档
export LAYOUTLMFT_DOCS_PER_CHUNK="5"  # 每个chunk 5个文档

# 清理旧数据
rm -rf "$LAYOUTLMFT_FEATURES_DIR"
mkdir -p "$LAYOUTLMFT_FEATURES_DIR"

# 使用新路径：examples/stage/util/
python examples/stage/util/extract_line_features_document_level.py

echo ""
echo "✓ 特征提取完成"
echo ""

# ========================================
# 步骤2: 训练 ParentFinder (1 epoch)
# ========================================
echo "步骤 2/3: 训练 ParentFinder (1 epoch, 文档级别)"
echo "------------------------------------------------------"

# 使用新路径：examples/stage/
python examples/stage/train_parent_finder.py \
    --mode full \
    --level document \
    --batch_size 1 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --max_chunks -1 \
    --features_dir "$LAYOUTLMFT_FEATURES_DIR" \
    --output_dir "$LAYOUTLMFT_FEATURES_DIR"

echo ""
echo "✓ ParentFinder 训练完成"
echo ""

# ========================================
# 步骤3: 训练关系分类器 (50 steps)
# ========================================
echo "步骤 3/3: 训练关系分类器 (50 steps, 文档级别)"
echo "------------------------------------------------------"

# 设置环境变量
export MAX_CHUNKS="-1"  # 使用全部chunk
export MAX_STEPS="50"   # 快速测试：只训练50步

# 使用新路径：examples/stage/
python examples/stage/train_multiclass_relation.py

echo ""
echo "✓ 关系分类器训练完成"
echo ""

# 下面是旧的临时脚本方案（已废弃，保留作为参考）
: << 'OLD_APPROACH'
cat > /tmp/train_relation_quick.py << 'SCRIPT'
import os
import sys

# 添加项目路径
sys.path.insert(0, '/root/code/layoutlmft')

# 设置环境变量
os.environ['LAYOUTLMFT_FEATURES_DIR'] = '/mnt/e/models/train_data/layoutlmft/line_features_doc_quick_test'
os.environ['LAYOUTLMFT_OUTPUT_DIR'] = '/mnt/e/models/train_data/layoutlmft/line_features_doc_quick_test'
os.environ['MAX_CHUNKS'] = '-1'

# 导入并修改配置
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle

# 加载数据集和模型定义
from examples.train_multiclass_relation import (
    MultiClassRelationDataset,
    train_step,
    evaluate,
    logger
)
from layoutlmft.models.relation_classifier import (
    MultiClassRelationClassifier,
    FocalLoss
)

# 配置
features_dir = "/mnt/e/models/train_data/layoutlmft/line_features_doc_quick_test"
output_dir = "/mnt/e/models/train_data/layoutlmft/line_features_doc_quick_test/multiclass_relation"
max_steps = 50  # 快速测试：只训练50步
batch_size = 32
learning_rate = 5e-4
neg_ratio = 1.5

os.makedirs(output_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 创建数据集
train_dataset = MultiClassRelationDataset(
    features_dir=features_dir,
    split="train",
    neg_ratio=neg_ratio,
    max_chunks=None
)

val_dataset = MultiClassRelationDataset(
    features_dir=features_dir,
    split="validation",
    neg_ratio=neg_ratio,
    max_chunks=None
)

logger.info(f"训练集大小: {len(train_dataset)}")
logger.info(f"验证集大小: {len(val_dataset)}")

# 创建dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# 创建模型
model = MultiClassRelationClassifier(
    hidden_size=768,
    num_relations=4,
    use_geometry=True,
    dropout=0.1
).to(device)

logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = FocalLoss(alpha=1.0, gamma=2.0)

# 训练
logger.info(f"\n开始训练 (最多 {max_steps} 步)...")
best_f1 = 0
global_step = 0

for epoch in range(100):  # 最多100个epoch（但会在max_steps停止）
    logger.info(f"\n===== Epoch {epoch + 1} =====")

    model.train()
    epoch_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}"):
        child_feats = batch["child_feat"].to(device)
        parent_feats = batch["parent_feat"].to(device)
        geom_feats = batch["geom_feat"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(child_feats, parent_feats, geom_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        if global_step >= max_steps:
            break

    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        logger.info(f"训练 - Loss: {avg_loss:.4f}")

    # 评估
    logger.info("评估中...")
    val_loss, val_acc, val_prec, val_rec, val_f1, cm, report = evaluate(
        model, val_loader, criterion, device
    )
    logger.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    # 保存最佳模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1
        }, os.path.join(output_dir, "best_model.pt"))
        logger.info(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")

    if global_step >= max_steps:
        logger.info(f"\n达到最大步数 {max_steps}，训练结束")
        break

logger.info(f"\n训练完成！最佳 F1: {best_f1:.4f}")
SCRIPT

python /tmp/train_relation_quick.py
OLD_APPROACH
# 旧的临时脚本方案结束

# ========================================
# 总结
# ========================================
echo "======================================================"
echo "小规模测试完成！"
echo "======================================================"
echo ""
echo "测试结果保存在: $LAYOUTLMFT_FEATURES_DIR"
echo ""
echo "模型文件:"
echo "  - ParentFinder: $LAYOUTLMFT_FEATURES_DIR/parent_finder_full/best_model.pt"
echo "  - 关系分类器: $LAYOUTLMFT_FEATURES_DIR/multiclass_relation/best_model.pt"
echo ""
echo "下一步："
echo "  1. 检查日志确认训练正常"
echo "  2. 如果测试成功，运行完整训练"
echo "  3. 完整训练: 删除 LAYOUTLMFT_NUM_SAMPLES 限制，增加 epoch"
echo ""
