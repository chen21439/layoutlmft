"""TEDS (Tree Edit Distance Similarity) 指标

用于评估层级文档结构重建 (Construct) 和阅读顺序预测 (Order)。
基于 apted 库计算树编辑距离。
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np

try:
    from apted import APTED, Config
    HAS_APTED = True
except ImportError:
    HAS_APTED = False

# 从 tree_utils 导入 Node 和 generate_doc_tree（避免重复实现）
from ..utils.tree_utils import Node, generate_doc_tree


@dataclass
class TEDSResult:
    """TEDS 评估结果"""
    macro_teds: float = 0.0
    micro_teds: float = 0.0
    total_distance: int = 0
    total_gt_nodes: int = 0
    total_pred_nodes: int = 0
    num_samples: int = 0
    per_sample: Dict[str, Dict] = field(default_factory=dict)


# Node 类已移至 utils/tree_utils.py


def tree_edit_distance(pred_tree: Node, gt_tree: Node) -> Tuple[int, float]:
    """计算树编辑距离和TEDS

    Args:
        pred_tree: 预测的文档树
        gt_tree: 真实的文档树

    Returns:
        (distance, teds): 编辑距离和TEDS分数
    """
    if not HAS_APTED:
        raise ImportError("需要安装 apted: pip install apted")

    distance = APTED(pred_tree, gt_tree, Config()).compute_edit_distance()
    max_nodes = max(len(pred_tree), len(gt_tree))
    teds = 1.0 - (float(distance) / max_nodes) if max_nodes > 0 else 1.0
    return distance, teds


# generate_doc_tree 函数已移至 utils/tree_utils.py


def sequence_edit_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, float]:
    """计算序列编辑距离

    用于阅读顺序链的比较。

    Args:
        seq1: 序列1
        seq2: 序列2

    Returns:
        (distance, similarity): 编辑距离和相似度分数
    """
    m, n = len(seq1), len(seq2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )

    distance = dp[m][n]
    max_len = max(m, n)
    similarity = 1.0 - distance / max_len if max_len > 0 else 1.0
    return distance, similarity


def transfer_tree_to_chain(tree: Node) -> Tuple[List[str], List[str]]:
    """将文档树转换为阅读顺序链

    Args:
        tree: 文档树根节点

    Returns:
        (main_chain, floating_chain): 主体文本链和浮动元素链
    """
    main_chain = []
    floating_chain = []

    def dfs_main(node: Node):
        main_chain.append(node.name)
        if len(node.children) == 0 and 'section' not in node.name:
            main_chain.append('<p>')
            return
        elif len(node.children) == 0 and 'section' in node.name:
            return
        for child in node.children:
            # 跳过浮动元素 (figure, table, caption)
            if child.name.startswith('figure') or child.name.startswith('table') or child.name.startswith('caption'):
                continue
            dfs_main(child)

    def dfs_floating(node: Node, level: int = 1):
        floating_chain.append(node.name)
        for child in node.children:
            if child.name.startswith('figure') or child.name.startswith('table') or child.name.startswith('caption'):
                dfs_floating(child, level + 1)
        if level == 2:
            floating_chain.append('<p>')

    dfs_main(tree)
    dfs_floating(tree)

    return main_chain, floating_chain


def split_chain_by_tag(chain: List[str], tag: str = '<p>') -> List[List[str]]:
    """按标签分割链

    Args:
        chain: 阅读顺序链
        tag: 分割标签

    Returns:
        分组后的链列表
    """
    groups = []
    current = []

    for item in chain:
        if item == tag:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(item)

    if current:
        groups.append(current)

    return groups


def min_edit_distance_between_groups(
    groups1: List[List[str]],
    groups2: List[List[str]],
) -> Tuple[float, float]:
    """计算两组链之间的最小编辑距离

    使用匈牙利算法进行最优匹配。

    Args:
        groups1: 链组1
        groups2: 链组2

    Returns:
        (total_distance, similarity): 总距离和相似度
    """
    from scipy.optimize import linear_sum_assignment

    total_items1 = sum(len(g) for g in groups1)
    total_items2 = sum(len(g) for g in groups2)
    max_items = max(total_items1, total_items2)

    if max_items == 0:
        return 0.0, 1.0

    if len(groups1) == 0 or len(groups2) == 0:
        return float(max_items), 0.0

    # 构建距离矩阵
    dist_matrix = np.zeros((len(groups1), len(groups2)))
    for i, g1 in enumerate(groups1):
        for j, g2 in enumerate(groups2):
            dist, _ = sequence_edit_distance(g1, g2)
            dist_matrix[i, j] = dist

    # 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    total_distance = dist_matrix[row_ind, col_ind].sum()

    similarity = 1.0 - total_distance / max_items
    return total_distance, similarity


class TEDSMetric:
    """TEDS 评估指标类

    用于评估层级文档结构重建质量。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置指标状态"""
        self.teds_list = []
        self.distance_list = []
        self.gt_nodes_list = []
        self.pred_nodes_list = []
        self.sample_ids = []

    def update(
        self,
        pred_texts: List[str],
        pred_parent_ids: List[int],
        pred_relations: List[str],
        gt_texts: List[str],
        gt_parent_ids: List[int],
        gt_relations: List[str],
        sample_id: str = None,
    ):
        """更新指标

        Args:
            pred_*: 预测结果
            gt_*: 真实标签
            sample_id: 样本ID
        """
        try:
            gt_tree = generate_doc_tree(gt_texts, gt_parent_ids, gt_relations)
            pred_tree = generate_doc_tree(pred_texts, pred_parent_ids, pred_relations)
            distance, teds = tree_edit_distance(pred_tree, gt_tree)

            self.teds_list.append(teds)
            self.distance_list.append(distance)
            self.gt_nodes_list.append(len(gt_tree))
            self.pred_nodes_list.append(len(pred_tree))
            self.sample_ids.append(sample_id or str(len(self.teds_list)))
        except Exception as e:
            print(f"TEDS 计算错误 ({sample_id}): {e}")

    def compute(self) -> TEDSResult:
        """计算最终指标"""
        result = TEDSResult()

        if len(self.teds_list) == 0:
            return result

        result.num_samples = len(self.teds_list)
        result.macro_teds = sum(self.teds_list) / len(self.teds_list)

        total_distance = sum(self.distance_list)
        total_max_nodes = sum(
            max(g, p) for g, p in zip(self.gt_nodes_list, self.pred_nodes_list)
        )
        result.micro_teds = 1.0 - total_distance / total_max_nodes if total_max_nodes > 0 else 1.0

        result.total_distance = total_distance
        result.total_gt_nodes = sum(self.gt_nodes_list)
        result.total_pred_nodes = sum(self.pred_nodes_list)

        for i, sid in enumerate(self.sample_ids):
            result.per_sample[sid] = {
                'teds': self.teds_list[i],
                'distance': self.distance_list[i],
                'gt_nodes': self.gt_nodes_list[i],
                'pred_nodes': self.pred_nodes_list[i],
            }

        return result
