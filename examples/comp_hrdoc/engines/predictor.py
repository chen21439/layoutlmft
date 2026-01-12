"""Predictor - 推理管线

封装 Construct 模型的推理逻辑：
1. 解码模型输出（格式B: 自指向方案）- 使用论文 Algorithm 1 联合解码
2. 转换为格式A（parent_id + relation）
3. 构建标准化输出

复用说明：
- scripts/infer.py 调用本模块进行推理
- scripts/train_doc.py 验证时调用本模块 + metrics/teds.py
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor

from ..utils.tree_utils import (
    resolve_ref_parents_and_relations,
    build_tree_from_parents,
    format_toc_tree,
    tree_insertion_decode,
)


def decode_construct_outputs(
    outputs: Dict[str, Tensor],
    mask: Tensor,
) -> Tuple[List[int], List[int]]:
    """解码 Construct 模型输出（论文 Algorithm 1: Tree Insertion Algorithm）

    使用联合解码保证树结构的合法性（sibling 约束）。

    Args:
        outputs: 模型输出，包含 parent_logits 和 sibling_logits
        mask: [N] 有效区域掩码

    Returns:
        pred_parents: 预测的层级父节点（格式B，自指向方案）
        pred_siblings: 预测的左兄弟（格式B，自指向方案）
    """
    parent_logits = outputs["parent_logits"]  # [N, N]
    sibling_logits = outputs.get("sibling_logits")  # [N, N] or None

    # 联合解码
    if sibling_logits is not None:
        pred_parents, pred_siblings = tree_insertion_decode(parent_logits, sibling_logits)
    else:
        # 无 sibling_logits 时退化为 argmax
        pred_parents = parent_logits.argmax(dim=-1).cpu().tolist()
        pred_siblings = list(range(len(pred_parents)))  # 全部自指向

    # 只保留有效节点
    mask_list = mask.cpu().tolist()
    valid_parents = [pred_parents[i] for i, m in enumerate(mask_list) if m]
    valid_siblings = [pred_siblings[i] for i, m in enumerate(mask_list) if m]

    return valid_parents, valid_siblings


def convert_to_format_a(
    pred_parents: List[int],
    pred_siblings: Optional[List[int]],
) -> Tuple[List[int], List[str]]:
    """将格式B转换为格式A

    格式B: hierarchical_parent + sibling（自指向方案）
    格式A: ref_parent + relation（顶层节点 parent=-1）

    Args:
        pred_parents: 格式B的层级父节点
        pred_siblings: 格式B的左兄弟

    Returns:
        ref_parents: 格式A的引用父节点
        relations: 关系类型列表
    """
    return resolve_ref_parents_and_relations(pred_parents, pred_siblings)


def build_predictions(
    ref_parents: List[int],
    relations: List[str],
    texts: Optional[List[str]] = None,
    line_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """构建标准化预测结果

    Args:
        ref_parents: 格式A的父节点索引
        relations: 关系类型
        texts: 文本列表
        line_ids: 原始 line_id 列表

    Returns:
        predictions: 标准化预测结果列表
    """
    n = len(ref_parents)
    predictions = []

    for i in range(n):
        pred = {
            "line_id": line_ids[i] if line_ids else i,
            "parent_id": ref_parents[i],
            "relation": relations[i],
        }
        if texts:
            pred["text"] = texts[i] if i < len(texts) else ""
        predictions.append(pred)

    return predictions


class ConstructPredictor:
    """Construct 模型推理器

    封装完整的推理流程：
    1. 模型前向传播
    2. 解码输出（格式B）
    3. 转换为格式A
    4. 构建标准化结果

    Usage:
        predictor = ConstructPredictor(model, device)
        result = predictor.predict(features, mask, texts)
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Args:
            model: Construct 模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.model.eval()

    def predict(
        self,
        region_features: Tensor,
        region_mask: Tensor,
        texts: Optional[List[str]] = None,
        line_ids: Optional[List[int]] = None,
        categories: Optional[Tensor] = None,
        reading_orders: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """运行推理

        Args:
            region_features: [N, H] 区域特征
            region_mask: [N] 有效区域掩码
            texts: 文本列表
            line_ids: 原始 line_id 列表
            categories: [N] 类别
            reading_orders: [N] 阅读顺序

        Returns:
            result: {
                "predictions": [...],  # 格式A预测结果
                "toc_tree": [...],     # 嵌套树结构
                "format_b": {...},     # 原始格式B输出（可选）
            }
        """
        # 确保输入在正确设备上
        region_features = region_features.to(self.device)
        region_mask = region_mask.to(self.device)
        if categories is not None:
            categories = categories.to(self.device)
        if reading_orders is not None:
            reading_orders = reading_orders.to(self.device)

        # 添加 batch 维度（如果需要）
        if region_features.dim() == 2:
            region_features = region_features.unsqueeze(0)  # [1, N, H]
            region_mask = region_mask.unsqueeze(0)  # [1, N]
            if categories is not None:
                categories = categories.unsqueeze(0)
            if reading_orders is not None:
                reading_orders = reading_orders.unsqueeze(0)

        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                region_features=region_features,
                region_mask=region_mask,
                categories=categories,
                reading_orders=reading_orders,
            )

        # 取第一个样本（batch_size=1）
        single_outputs = {
            "parent_logits": outputs["parent_logits"][0],
            "sibling_logits": outputs["sibling_logits"][0] if "sibling_logits" in outputs else None,
        }
        single_mask = region_mask[0]

        # 解码（格式B）
        pred_parents, pred_siblings = decode_construct_outputs(single_outputs, single_mask)

        # 转换为格式A
        ref_parents, relations = convert_to_format_a(pred_parents, pred_siblings)

        # 构建预测结果
        predictions = build_predictions(ref_parents, relations, texts, line_ids)

        # 构建嵌套树
        toc_tree = build_tree_from_parents(predictions, id_key="line_id", parent_key="parent_id")

        return {
            "predictions": predictions,
            "toc_tree": toc_tree,
            "num_nodes": len(predictions),
            # 保留格式B（用于调试或其他用途）
            "format_b": {
                "hierarchical_parents": pred_parents,
                "siblings": pred_siblings,
            },
        }

    def predict_batch(
        self,
        outputs: Dict[str, Tensor],
        masks: Tensor,
        texts_batch: Optional[List[List[str]]] = None,
        line_ids_batch: Optional[List[List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """批量解码模型输出

        用于训练时的验证，模型已经前向传播完成。

        Args:
            outputs: 模型输出（batch）
            masks: [B, N] 掩码
            texts_batch: 每个样本的文本列表
            line_ids_batch: 每个样本的 line_id 列表

        Returns:
            results: 每个样本的预测结果
        """
        batch_size = masks.size(0)
        results = []

        for b in range(batch_size):
            single_outputs = {
                "parent_logits": outputs["parent_logits"][b],
            }
            if "sibling_logits" in outputs:
                single_outputs["sibling_logits"] = outputs["sibling_logits"][b]

            single_mask = masks[b]

            # 解码
            pred_parents, pred_siblings = decode_construct_outputs(single_outputs, single_mask)

            # 转换
            ref_parents, relations = convert_to_format_a(pred_parents, pred_siblings)

            # 构建结果
            texts = texts_batch[b] if texts_batch else None
            line_ids = line_ids_batch[b] if line_ids_batch else None
            predictions = build_predictions(ref_parents, relations, texts, line_ids)
            toc_tree = build_tree_from_parents(predictions, id_key="line_id", parent_key="parent_id")

            results.append({
                "predictions": predictions,
                "toc_tree": toc_tree,
                "num_nodes": len(predictions),
                "format_b": {
                    "hierarchical_parents": pred_parents,
                    "siblings": pred_siblings,
                },
            })

        return results


def format_result_as_tree(result: Dict[str, Any], max_text_len: int = 40) -> List[str]:
    """将预测结果格式化为树形字符串

    Args:
        result: predict() 返回的结果
        max_text_len: 文本最大长度

    Returns:
        树形字符串列表
    """
    return format_toc_tree(result["toc_tree"], max_text_len=max_text_len)
