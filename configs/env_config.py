#!/usr/bin/env python
# coding=utf-8
"""
环境配置管理
支持：local（本机）、cloud（云服务器）、auto（自动检测）
"""

import os
import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    model_name_or_path: str = "microsoft/layoutlmv2-base-uncased"
    output_dir: str = "./output/hrdoc"

    # 本地模型路径（用于离线加载，避免 transformers 版本 bug）
    local_model_path: Optional[str] = None

    # 训练参数
    max_steps: int = 1000
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # 优化器参数
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 评估和保存
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"

    # 性能优化
    fp16: bool = False
    dataloader_num_workers: int = 4

    # 其他
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save_json(self, path: str):
        """保存为JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str):
        """从JSON加载"""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(**config)


class EnvironmentDetector:
    """环境检测器"""

    @staticmethod
    def get_gpu_memory() -> Optional[float]:
        """获取GPU显存（GB）"""
        if not torch.cuda.is_available():
            return None

        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return gpu_memory
        except:
            return None

    @staticmethod
    def get_gpu_count() -> int:
        """获取GPU数量"""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def is_cloud_environment() -> bool:
        """判断是否为云环境（启发式方法）"""
        # 检查环境变量
        cloud_indicators = [
            'AWS_EXECUTION_ENV',
            'AZURE_VM',
            'GOOGLE_CLOUD_PROJECT',
            'KUBERNETES_SERVICE_HOST',
            'CLOUD_ENV'  # 自定义标记
        ]

        for indicator in cloud_indicators:
            if os.getenv(indicator):
                return True

        # 检查主机名
        hostname = os.uname().nodename.lower()
        cloud_hosts = ['aws', 'azure', 'gcp', 'cloud', 'gpu-server']
        if any(h in hostname for h in cloud_hosts):
            return True

        return False

    @classmethod
    def detect_environment(cls) -> str:
        """自动检测环境类型"""
        gpu_memory = cls.get_gpu_memory()
        gpu_count = cls.get_gpu_count()
        is_cloud = cls.is_cloud_environment()

        # 判断逻辑
        if is_cloud or (gpu_memory and gpu_memory >= 20):  # 20GB+认为是云服务器
            return "cloud"
        else:
            return "local"

    @classmethod
    def print_environment_info(cls):
        """打印环境信息"""
        gpu_count = cls.get_gpu_count()
        gpu_memory = cls.get_gpu_memory()
        is_cloud = cls.is_cloud_environment()
        detected_env = cls.detect_environment()

        print("=" * 60)
        print("环境信息")
        print("=" * 60)
        print(f"GPU 数量: {gpu_count}")
        print(f"GPU 显存: {gpu_memory:.2f} GB" if gpu_memory else "GPU 显存: N/A")
        print(f"云环境标记: {is_cloud}")
        print(f"检测到的环境: {detected_env}")
        print(f"主机名: {os.uname().nodename}")
        print("=" * 60)


def get_config(env: str = "auto", dataset: str = "simple") -> TrainingConfig:
    """
    获取指定环境的配置

    Args:
        env: 环境类型 ("local", "cloud", "auto")

    Returns:
        TrainingConfig 对象
    """
    # 从环境变量读取输出根目录（可选）
    output_root = os.getenv("LAYOUTLMFT_OUTPUT_DIR", None)

    # 自动检测环境
    if env == "auto":
        env = EnvironmentDetector.detect_environment()
        print(f"✓ 自动检测环境: {env}")

    # HRDoc-Hard配置（40k步）
    if env == "cloud" and dataset == "hard":
        return TrainingConfig(
            output_dir="./output/hrdoc_hard_full",
            max_steps=40000,  # HRDoc-Hard论文值
            per_device_train_batch_size=3,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=100,
            eval_steps=1000,
            save_steps=1000,
            save_total_limit=3,
            evaluation_strategy="steps",
            fp16=True,
            dataloader_num_workers=4,
        )

    # 环境特定配置
    configs = {
        "local": TrainingConfig(
            # 本机：小规模快速测试（输出到 E 盘节省系统盘空间）
            output_dir=output_root + "/hrdoc_local" if output_root else "/mnt/e/models/train_data/layoutlmft/hrdoc_local",
            local_model_path="/mnt/e/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350c668f88ec580046e35c670df6ec616c1",
            max_steps=500,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,  # 有效batch=4
            learning_rate=5e-5,
            warmup_ratio=0.1,
            logging_steps=20,
            eval_steps=100,
            save_steps=100,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
        ),

        "cloud": TrainingConfig(
            # 云服务器：完整训练（严格对齐HRDoc论文）
            # 论文参考: "batch size of 3 (page-level) for 30,000 steps"
            output_dir=output_root + "/hrdoc_simple_full" if output_root else "./output/hrdoc_simple_full",
            local_model_path="/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350c668f88ec580046e35c670df6ec616c1",
            max_steps=30000,  # HRDoc-Simple 论文精确值
            per_device_train_batch_size=3,  # 论文精确值
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,  # 不累积，直接batch=3
            learning_rate=5e-5,  # LayoutLMv2常用finetune lr
            warmup_ratio=0.1,  # 前10%步数warmup
            weight_decay=0.01,  # BERT/LayoutLM标准值
            logging_steps=100,
            eval_steps=1000,
            save_steps=1000,
            save_total_limit=3,
            evaluation_strategy="steps",
            fp16=True,  # 对齐论文硬件环境(V100-24G)
            dataloader_num_workers=4,
        ),

        "quick": TrainingConfig(
            # 快速测试（几分钟，输出到 E 盘节省系统盘空间）
            output_dir=output_root + "/hrdoc_quick" if output_root else "/mnt/e/models/train_data/layoutlmft/hrdoc_quick",
            local_model_path="/mnt/e/models/HuggingFace/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350c668f88ec580046e35c670df6ec616c1",
            max_steps=50,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            logging_steps=10,
            eval_steps=25,
            save_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
        ),
    }

    if env not in configs:
        raise ValueError(f"未知环境: {env}，可选: {list(configs.keys())}")

    return configs[env]


def create_default_configs():
    """创建默认配置文件"""
    os.makedirs("./configs", exist_ok=True)

    for env in ["local", "cloud", "quick"]:
        config = get_config(env)
        config_path = f"./configs/{env}_config.json"
        config.save_json(config_path)
        print(f"✓ 创建配置: {config_path}")


if __name__ == "__main__":
    # 打印环境信息
    EnvironmentDetector.print_environment_info()

    # 创建默认配置
    create_default_configs()

    # 测试加载
    print("\n测试配置加载:")
    for env in ["local", "cloud", "quick"]:
        config = get_config(env)
        print(f"\n{env} 配置:")
        print(f"  - max_steps: {config.max_steps}")
        print(f"  - batch_size: {config.per_device_train_batch_size}")
        print(f"  - learning_rate: {config.learning_rate}")
