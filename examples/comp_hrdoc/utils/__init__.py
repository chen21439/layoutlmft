# utils - 横切能力与基础设施
#
# config.py: 配置解析/GPU设置/环境检测
# experiment_manager.py: 实验管理
# stage_feature_extractor.py: 使用 stage 模型提取 line-level 特征
# toc_compress.py: TOC 子图压缩（对齐论文 4.4）
# tree_utils.py: 树构建工具（层级父节点和兄弟分组计算）

from .config import (
    setup_environment,
    setup_gpu,
    get_device,
    load_config,
    get_gpu_config,
    get_artifact_path,
    get_data_path,
    print_gpu_info,
    print_config,
)

from .experiment_manager import (
    ExperimentManager,
    ensure_experiment,
    get_artifact_path as get_artifact_path_legacy,
)

def StageFeatureExtractor(*args, **kwargs):
    """延迟导入 StageFeatureExtractor 类"""
    from .stage_feature_extractor import StageFeatureExtractor as _cls
    return _cls(*args, **kwargs)


from .tree_utils import (
    Node,
    RELATION_STR_TO_INT,
    RELATION_INT_TO_STR,
    normalize_relation,
    generate_doc_tree,
    resolve_parent_and_sibling_from_tree,
    build_sibling_matrix,
)


__all__ = [
    'setup_environment',
    'setup_gpu',
    'get_device',
    'load_config',
    'get_gpu_config',
    'get_artifact_path',
    'get_data_path',
    'print_gpu_info',
    'print_config',
    'ExperimentManager',
    'ensure_experiment',
    # tree 构建工具
    'Node',
    'RELATION_STR_TO_INT',
    'RELATION_INT_TO_STR',
    'normalize_relation',
    'generate_doc_tree',
    'resolve_parent_and_sibling_from_tree',
    'build_sibling_matrix',
    # stage 特征提取
    'StageFeatureExtractor',
]
