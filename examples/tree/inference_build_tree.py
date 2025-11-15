#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 完整推理 Pipeline
实现论文的 Overall Task：将三个子任务串联，输出文档结构树

Pipeline 流程：
1. SubTask 1: 语义单元分类 (LayoutLMv2) → line_labels
2. SubTask 2: 父节点查找 (ParentFinder) → parent_indices
3. SubTask 3: 关系分类 (RelationClassifier) → relation_types
4. Overall Task: 构建文档树 (DocumentTree)

使用方法：
    python examples/tree/inference_build_tree.py \\
        --subtask1_model /path/to/layoutlmv2/checkpoint \\
        --subtask2_model /path/to/parent_finder/best_model.pt \\
        --subtask3_model /path/to/relation_classifier/best_model.pt \\
        --features_dir /path/to/line_features \\
        --output_dir ./outputs/trees \\
        --max_samples 10
"""

import logging
import os
import sys
import torch
import pickle
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutlmft.models.relation_classifier import (
    MultiClassRelationClassifier,
    compute_geometry_features,
)

# 导入树结构
from document_tree import DocumentTree, LABEL_MAP, RELATION_MAP

logger = logging.getLogger(__name__)


# ==================== 模型定义 ====================

class SimpleParentFinder(torch.nn.Module):
    """父节点查找器（从train_parent_finder_simple.py复制）"""

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + 4, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, child_feat, parent_feats, geom_feats):
        batch_size, num_candidates, hidden_size = parent_feats.shape
        child_feat_expanded = child_feat.unsqueeze(1).expand(batch_size, num_candidates, hidden_size)
        combined = torch.cat([child_feat_expanded, parent_feats, geom_feats], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores


# ==================== 推理函数 ====================

def load_models(subtask2_path: str, subtask3_path: str, device: torch.device):
    """
    加载训练好的模型

    Args:
        subtask2_path: SubTask 2模型路径（父节点查找）
        subtask3_path: SubTask 3模型路径（关系分类）
        device: 设备

    Returns:
        subtask2_model, subtask3_model
    """
    logger.info(f"加载SubTask 2模型: {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)

    # 尝试判断模型类型
    state_dict = subtask2_checkpoint.get("model_state_dict", subtask2_checkpoint)

    # 检查是否是SimpleParentFinder（几何特征是4维）
    if any("score_head" in k for k in state_dict.keys()):
        subtask2_model = SimpleParentFinder(hidden_size=768, dropout=0.1)
        logger.info("  检测到SimpleParentFinder模型")
    else:
        # 可能是其他类型的模型
        raise ValueError("不支持的SubTask 2模型类型")

    subtask2_model.load_state_dict(state_dict)
    subtask2_model = subtask2_model.to(device)
    subtask2_model.eval()
    logger.info(f"✓ SubTask 2模型加载成功")

    logger.info(f"加载SubTask 3模型: {subtask3_path}")
    subtask3_checkpoint = torch.load(subtask3_path, map_location=device)
    subtask3_model = MultiClassRelationClassifier(
        hidden_size=768,
        num_relations=4,
        use_geometry=True,
        dropout=0.1,
    )
    subtask3_model.load_state_dict(subtask3_checkpoint["model_state_dict"])
    subtask3_model = subtask3_model.to(device)
    subtask3_model.eval()
    logger.info(f"✓ SubTask 3模型加载成功")

    return subtask2_model, subtask3_model


def predict_parents(
    subtask2_model,
    line_features: torch.Tensor,
    line_mask: torch.Tensor,
    line_bboxes: list,
    device: torch.device,
) -> list:
    """
    SubTask 2: 预测每个语义单元的父节点

    Args:
        subtask2_model: 父节点查找模型
        line_features: 行级特征 [num_lines, hidden_size]
        line_mask: 有效行mask [num_lines]
        line_bboxes: 行边界框列表 [num_lines, 4]
        device: 设备

    Returns:
        parent_indices: 每个行的父节点索引列表 [num_lines]
    """
    num_lines = line_features.shape[0]
    parent_indices = []

    with torch.no_grad():
        for child_idx in range(num_lines):
            # 检查有效性
            if child_idx >= line_mask.shape[0]:
                parent_indices.append(-1)
                continue

            # 安全地获取mask值
            child_mask_val = line_mask[child_idx]
            if hasattr(child_mask_val, 'item'):
                child_mask_val = child_mask_val.item()
            if not child_mask_val:
                parent_indices.append(-1)
                continue

            # 候选父节点：0 到 child_idx-1
            if child_idx == 0:
                parent_indices.append(-1)  # 第一个节点的父节点是ROOT
                continue

            best_score = -float('inf')
            pred_parent_idx = -1

            # 遍历所有候选父节点
            for candidate_idx in range(child_idx):
                if candidate_idx >= line_mask.shape[0]:
                    continue

                # 安全地获取mask值
                cand_mask_val = line_mask[candidate_idx]
                if hasattr(cand_mask_val, 'item'):
                    cand_mask_val = cand_mask_val.item()
                if not cand_mask_val:
                    continue

                if candidate_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                    continue

                # 提取特征
                child_feat = line_features[child_idx].unsqueeze(0)
                parent_feat = line_features[candidate_idx].unsqueeze(0)

                parent_bbox = torch.tensor(line_bboxes[candidate_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                # 预测得分
                scores = subtask2_model(child_feat, parent_feat.unsqueeze(1), geom_feat.unsqueeze(1))
                score = scores[0, 0].item()

                if score > best_score:
                    best_score = score
                    pred_parent_idx = candidate_idx

            parent_indices.append(pred_parent_idx)

    return parent_indices


def predict_relations(
    subtask3_model,
    line_features: torch.Tensor,
    parent_indices: list,
    line_bboxes: list,
    device: torch.device,
) -> list:
    """
    SubTask 3: 预测每个父子对之间的关系类型

    Args:
        subtask3_model: 关系分类模型
        line_features: 行级特征 [num_lines, hidden_size]
        parent_indices: 父节点索引列表 [num_lines]
        line_bboxes: 行边界框列表 [num_lines, 4]
        device: 设备

    Returns:
        relation_types: 关系类型列表 [num_lines]
        relation_confidences: 关系置信度列表 [num_lines]
    """
    num_lines = line_features.shape[0]
    relation_types = []
    relation_confidences = []

    with torch.no_grad():
        for child_idx in range(num_lines):
            parent_idx = parent_indices[child_idx]

            # 如果没有父节点，关系为none
            if parent_idx < 0 or parent_idx >= num_lines:
                relation_types.append(0)  # none
                relation_confidences.append(1.0)
                continue

            # 检查有效性
            if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                relation_types.append(0)
                relation_confidences.append(0.0)
                continue

            # 提取特征
            parent_feat = line_features[parent_idx].unsqueeze(0)
            child_feat = line_features[child_idx].unsqueeze(0)

            parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
            child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
            geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

            # 预测关系
            logits = subtask3_model(parent_feat, child_feat, geom_feat)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()

            relation_types.append(pred_label)
            relation_confidences.append(confidence)

    return relation_types, relation_confidences


def tree_to_hrds_format(tree, page_num=0):
    """
    将DocumentTree转换为HRDS平铺格式

    Args:
        tree: DocumentTree实例
        page_num: 页码

    Returns:
        list: HRDS格式的平铺列表
    """
    flat_list = []

    def traverse(node, parent_id=-1):
        if node.idx >= 0:  # 跳过ROOT
            # 转换为HRDS格式
            hrds_item = {
                "line_id": node.idx,
                "text": node.text if node.text else "",
                "box": node.bbox,
                "class": node.label.lower().replace("-", "_"),
                "page": page_num,
                "parent_id": parent_id,
                "relation": node.relation_to_parent if node.relation_to_parent else "none",
                "is_meta": (node.relation_to_parent == "meta"),
            }
            flat_list.append(hrds_item)

        # 递归遍历子节点
        for child in node.children:
            traverse(child, node.idx)

    traverse(tree.root)
    return flat_list


def inference_single_page(
    page_data: dict,
    subtask2_model,
    subtask3_model,
    device: torch.device,
) -> DocumentTree:
    """
    对单个页面进行完整推理

    Args:
        page_data: 页面数据（包含line_features, line_mask, line_bboxes等）
        subtask2_model: 父节点查找模型
        subtask3_model: 关系分类模型
        device: 设备

    Returns:
        DocumentTree实例
    """
    # 提取数据
    line_features = page_data["line_features"].squeeze(0).to(device)
    line_mask = page_data["line_mask"].squeeze(0)
    line_bboxes = page_data["line_bboxes"]

    # 如果有预先提取的line_labels（SubTask 1的结果），使用它们
    # 否则，需要用LayoutLMv2模型重新推理
    if "line_labels" in page_data and page_data["line_labels"] is not None:
        line_labels = page_data["line_labels"]
        # 如果是numpy数组或tensor，转换为list
        if hasattr(line_labels, 'tolist'):
            line_labels = line_labels.tolist()
        elif not isinstance(line_labels, list):
            line_labels = list(line_labels)
    else:
        # 这里需要SubTask 1模型进行推理
        # 为了简化，暂时使用占位符
        logger.warning("page_data中没有line_labels，使用占位符")
        num_lines = line_features.shape[0]
        line_labels = [3] * num_lines  # 默认为Para-Line

    # SubTask 2: 预测父节点
    parent_indices = predict_parents(
        subtask2_model, line_features, line_mask, line_bboxes, device
    )

    # SubTask 3: 预测关系
    relation_types, relation_confidences = predict_relations(
        subtask3_model, line_features, parent_indices, line_bboxes, device
    )

    # 构建树
    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_bboxes=line_bboxes,
        line_texts=None,  # 如果有文本数据可以传入
        label_confidences=None,
        relation_confidences=relation_confidences,
    )

    return tree


def main():
    parser = argparse.ArgumentParser(description="HRDoc完整推理Pipeline")
    parser.add_argument(
        "--subtask2_model",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt",
        help="SubTask 2模型路径（父节点查找）",
    )
    parser.add_argument(
        "--subtask3_model",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt",
        help="SubTask 3模型路径（关系分类）",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/line_features",
        help="特征文件目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/trees",
        help="输出目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="数据集分割",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="最多处理多少个样本（用于快速测试）",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=1,
        help="最多加载多少个chunk",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="是否保存JSON格式的树",
    )
    parser.add_argument(
        "--save_markdown",
        action="store_true",
        help="是否保存Markdown格式的树",
    )
    parser.add_argument(
        "--save_ascii",
        action="store_true",
        help="是否保存ASCII格式的树",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="hrds",
        choices=["tree", "hrds", "both"],
        help="输出格式：tree=嵌套树，hrds=HRDS平铺格式，both=两种都输出",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"特征目录: {args.features_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 加载模型
    logger.info("\n" + "="*80)
    logger.info("加载模型")
    logger.info("="*80)
    subtask2_model, subtask3_model = load_models(
        args.subtask2_model,
        args.subtask3_model,
        device
    )

    # 加载数据
    logger.info("\n" + "="*80)
    logger.info("加载数据")
    logger.info("="*80)
    import glob
    pattern = os.path.join(args.features_dir, f"{args.split}_line_features_chunk_*.pkl")
    chunk_files = sorted(glob.glob(pattern))[:args.max_chunks]

    if len(chunk_files) == 0:
        raise ValueError(f"没有找到特征文件: {pattern}")

    logger.info(f"找到 {len(chunk_files)} 个chunk文件")
    page_features = []
    for chunk_file in chunk_files:
        logger.info(f"  加载 {os.path.basename(chunk_file)}...")
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)
        page_features.extend(chunk_data)

    logger.info(f"总共加载了 {len(page_features)} 页")

    # 限制样本数量
    if args.max_samples > 0:
        page_features = page_features[:args.max_samples]
        logger.info(f"限制处理前 {args.max_samples} 个样本")

    # 推理
    logger.info("\n" + "="*80)
    logger.info("开始推理")
    logger.info("="*80)

    trees = []
    for i, page_data in enumerate(tqdm(page_features, desc="推理进度")):
        try:
            tree = inference_single_page(
                page_data, subtask2_model, subtask3_model, device
            )
            trees.append(tree)

            # 保存单个树
            page_idx = page_data.get("page_idx", i)

            # 根据output_format保存
            if args.output_format in ["hrds", "both"]:
                # 保存HRDS格式
                hrds_data = tree_to_hrds_format(tree, page_num=page_idx)
                hrds_path = output_dir / f"page_{page_idx:04d}.json"
                with open(hrds_path, 'w', encoding='utf-8') as f:
                    json.dump(hrds_data, f, indent=2, ensure_ascii=False)

            if args.output_format in ["tree", "both"]:
                # 保存树格式
                if args.save_json:
                    json_path = output_dir / f"tree_{page_idx:04d}.json"
                    tree.to_json(str(json_path))

                if args.save_markdown:
                    md_path = output_dir / f"tree_{page_idx:04d}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(tree.to_markdown())

                if args.save_ascii:
                    ascii_path = output_dir / f"tree_{page_idx:04d}_ascii.txt"
                    with open(ascii_path, 'w', encoding='utf-8') as f:
                        f.write(tree.visualize_ascii())

        except Exception as e:
            import traceback
            logger.error(f"处理页面 {i} 时出错: {str(e)}")
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            continue

    # 统计
    logger.info("\n" + "="*80)
    logger.info("推理完成！")
    logger.info("="*80)
    logger.info(f"成功处理: {len(trees)}/{len(page_features)} 个页面")
    logger.info(f"输出目录: {output_dir}")

    # 汇总统计
    if trees:
        total_nodes = sum(tree.get_statistics()["total_nodes"] for tree in trees)
        avg_nodes = total_nodes / len(trees)
        logger.info(f"\n平均每页节点数: {avg_nodes:.1f}")

        # 保存汇总
        summary = {
            "total_pages": len(trees),
            "total_nodes": total_nodes,
            "average_nodes_per_page": avg_nodes,
            "trees": [tree.to_dict() for tree in trees[:5]],  # 只保存前5个完整树
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 汇总信息保存到: {summary_path}")


if __name__ == "__main__":
    main()
