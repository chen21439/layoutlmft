# data - 数据处理模块
#
# 数据集格式:
# - HRDS/HRDH: 每个文档一个 JSON 文件 (hrdoc_loader.py)
#
# 加载器:
# - hrdoc_loader.py: HRDS/HRDH 数据加载（统一的加载器，支持 Construct 训练）
# - line_level_loader.py: 行级别数据加载 (手动拼接方式)
# - line_collator_v2.py: 行级别数据加载 V2 (is_split_into_words=True)

from .line_level_loader import (
    LineLevelDataset,
    LineLevelCollator,
    LineLevelLayoutXLMCollator,
    create_line_level_dataloaders,
    create_layoutxlm_line_dataloaders,
)

from .line_collator_v2 import (
    LineLevelCollatorV2,
    create_dataloaders_v2,
)

# 统一的 HRDoc 加载器 (支持 HRDS/HRDH)
from .hrdoc_loader import (
    HRDocDataset,
    HRDocCollator,
    HRDocLayoutXLMCollator,
    create_hrdoc_dataloaders,
    create_hrdoc_layoutxlm_dataloaders,
    # 向后兼容别名
    HRDSDataset,
    HRDSCollator,
    HRDSLayoutXLMCollator,
    create_hrds_dataloaders,
    create_hrds_layoutxlm_dataloaders,
)

