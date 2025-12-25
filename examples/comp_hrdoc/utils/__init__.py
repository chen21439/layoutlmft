# utils - 横切能力与基础设施
#
# config.py: 配置解析/GPU设置/环境检测
# experiment_manager.py: 实验管理

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
]
