"""TOC 子图压缩工具

将完整的 line-level 树结构压缩到 section-only 子图，
用于对齐论文 4.4 Construct 模块的训练目标。

核心问题：
- 原始 parent_ids 是基于所有 lines 的索引
- section 的 parent 可能指向非 section 节点
- 需要"向上追溯"到最近的 section ancestor

使用方式：
    from examples.comp_hrdoc.utils.toc_compress import compress_to_sections_batch

    # 在 train loop 中
    compressed = compress_to_sections_batch(
        line_features=line_features,      # [B, L, H]
        line_mask=line_mask,              # [B, L]
        parent_ids=parent_ids,            # [B, L]
        line_labels=line_labels,          # [B, L] category labels
        section_label_id=SECTION_LABEL_ID,
    )
    # compressed["features"]: [B, S, H]  (S = max sections)
    # compressed["mask"]: [B, S]
    # compressed["parent_ids"]: [B, S]  (remapped to section-only space)
"""

from typing import Dict, List, Tuple, Optional
import torch


def compress_to_sections(
    parent_ids: torch.Tensor,  # [L] full lines, -1=root
    section_mask: torch.Tensor,  # [L] bool, True for section lines
) -> Tuple[torch.Tensor, Dict[int, int], List[int]]:
    """
    将 line-level parent_ids 压缩到 section-only 子图。

    采用论文自指向方案：root 节点的 parent 指向自己。

    Args:
        parent_ids: [L] 每个 line 的 parent index，-1 表示 root
        section_mask: [L] bool，True 表示该 line 是 section

    Returns:
        new_parent_ids: [S] section-only 的 parent indices，root 自指向
        old2new: dict，old_idx -> new_idx 映射
        kept_old_indices: List[int]，保留的原始索引列表
    """
    L = len(parent_ids)
    device = parent_ids.device

    # 找出所有 section 节点
    kept_old_indices = [i for i in range(L) if section_mask[i].item()]
    old2new = {old_i: new_i for new_i, old_i in enumerate(kept_old_indices)}

    def climb_to_section(p: int) -> int:
        """向上追溯到最近的 section ancestor 或 root"""
        visited = set()
        while p != -1 and 0 <= p < L:
            if p in visited:  # 避免循环
                return -1
            visited.add(p)
            if section_mask[p].item():
                return p
            p = parent_ids[p].item()
        return -1  # root

    # 论文自指向方案：root 节点的 parent 指向自己
    new_parent_ids = []
    for new_i, old_i in enumerate(kept_old_indices):
        p = parent_ids[old_i].item()
        if p == -1:
            new_parent_ids.append(new_i)  # root 自指向
        else:
            section_ancestor = climb_to_section(p)
            if section_ancestor == -1:
                new_parent_ids.append(new_i)  # 无 section ancestor，视为 root（自指向）
            else:
                new_parent_ids.append(old2new[section_ancestor])

    return (
        torch.tensor(new_parent_ids, dtype=torch.long, device=device),
        old2new,
        kept_old_indices,
    )


def compress_to_sections_batch(
    line_features: torch.Tensor,  # [B, L, H]
    line_mask: torch.Tensor,      # [B, L]
    parent_ids: torch.Tensor,     # [B, L]
    line_labels: torch.Tensor,    # [B, L] category labels
    reading_orders: Optional[torch.Tensor] = None,  # [B, L]
    section_label_id: int = 1,    # section 的 label id
) -> Dict[str, torch.Tensor]:
    """
    批量压缩到 section-only 子图。

    Args:
        line_features: [B, L, H] line-level 特征
        line_mask: [B, L] 有效 line mask
        parent_ids: [B, L] parent indices
        line_labels: [B, L] category labels (用于识别 section)
        reading_orders: [B, L] 可选，阅读顺序
        section_label_id: section 类别的 label id

    Returns:
        Dict with:
            features: [B, S, H] section-only 特征
            mask: [B, S] section mask
            parent_ids: [B, S] remapped parent indices
            categories: [B, S] section categories (全是 section_label_id)
            reading_orders: [B, S] remapped reading orders
            num_sections: List[int] 每个样本的 section 数量
    """
    B, L, H = line_features.shape
    device = line_features.device

    # 找出每个样本的 section mask
    section_masks = (line_labels == section_label_id) & line_mask

    # 计算每个样本的 section 数量和最大值
    num_sections_list = [section_masks[b].sum().item() for b in range(B)]
    max_sections = max(num_sections_list) if num_sections_list else 1
    max_sections = max(max_sections, 1)  # 至少 1

    # 初始化输出
    new_features = torch.zeros(B, max_sections, H, device=device)
    new_mask = torch.zeros(B, max_sections, dtype=torch.bool, device=device)
    new_parent_ids = torch.full((B, max_sections), -1, dtype=torch.long, device=device)
    new_categories = torch.full((B, max_sections), section_label_id, dtype=torch.long, device=device)
    new_reading_orders = torch.zeros(B, max_sections, dtype=torch.long, device=device)

    for b in range(B):
        sample_section_mask = section_masks[b]
        sample_parent_ids = parent_ids[b]

        # 压缩到 section 子图
        new_pids, old2new, kept_indices = compress_to_sections(
            sample_parent_ids, sample_section_mask
        )

        S = len(kept_indices)
        if S == 0:
            continue

        # 复制特征
        for new_i, old_i in enumerate(kept_indices):
            new_features[b, new_i] = line_features[b, old_i]

        # 设置 mask 和 parent_ids
        new_mask[b, :S] = True
        new_parent_ids[b, :S] = new_pids

        # 处理 reading_orders
        if reading_orders is not None:
            # 按原始 reading_order 排序 section，然后重新编号
            section_orders = [(reading_orders[b, old_i].item(), new_i)
                             for new_i, old_i in enumerate(kept_indices)]
            section_orders.sort(key=lambda x: x[0])
            for rank, (_, new_i) in enumerate(section_orders):
                new_reading_orders[b, new_i] = rank
        else:
            # 默认按出现顺序
            new_reading_orders[b, :S] = torch.arange(S, device=device)

    return {
        "features": new_features,
        "mask": new_mask,
        "parent_ids": new_parent_ids,
        "categories": new_categories,
        "reading_orders": new_reading_orders,
        "num_sections": num_sections_list,
    }


def generate_sibling_labels_from_parents(
    parent_ids: torch.Tensor,     # [B, N]
    mask: torch.Tensor,           # [B, N]
    reading_orders: torch.Tensor = None,  # [B, N] reading order positions
) -> torch.Tensor:
    """
    从 parent_ids 和 reading_orders 生成 left sibling labels。

    每个节点的 left sibling 是同一 parent 下，reading order 在它之前的兄弟节点。

    Args:
        parent_ids: [B, N] parent index for each node (-1 for root)
        mask: [B, N] valid node mask
        reading_orders: [B, N] reading order positions (default: use indices)

    Returns:
        sibling_labels: [B, N] index of left sibling (-1 if no left sibling)
    """
    B, N = parent_ids.shape
    device = parent_ids.device

    # Default reading orders: use indices
    if reading_orders is None:
        reading_orders = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

    sibling_labels = torch.full((B, N), fill_value=-1, dtype=torch.long, device=device)

    for b in range(B):
        # Group nodes by parent
        parent_to_children = {}
        for i in range(N):
            if not mask[b, i]:
                continue
            parent_i = parent_ids[b, i].item()
            # 论文自指向方案：root 自指向，所以 parent_i == i 时是 root，跳过
            if parent_i == i:  # root (self-pointing) has no siblings
                continue
            if parent_i not in parent_to_children:
                parent_to_children[parent_i] = []
            parent_to_children[parent_i].append(i)

        # For each group of siblings, sort by reading order and assign left sibling
        for parent, children in parent_to_children.items():
            if len(children) < 2:
                continue
            # Sort by reading order
            children_sorted = sorted(children, key=lambda x: reading_orders[b, x].item())
            # Assign left sibling for each node (except the first one)
            for idx in range(1, len(children_sorted)):
                curr_node = children_sorted[idx]
                left_sibling = children_sorted[idx - 1]
                sibling_labels[b, curr_node] = left_sibling

    return sibling_labels
