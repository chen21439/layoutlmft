#!/usr/bin/env python
# coding=utf-8
"""
训练监控配置
提供实时指标观测、异常检测和自动报警
"""

import os
import json
import time
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# GPUtil是可选依赖
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


@dataclass
class MonitoringThresholds:
    """监控阈值配置"""

    # Loss相关
    max_loss_spike: float = 2.0  # loss突然上升超过2倍触发警告
    min_loss_decrease_rate: float = 0.01  # 连续100步loss下降不足1%触发警告
    loss_nan_check: bool = True  # 检测NaN loss

    # 学习率相关
    lr_min: float = 1e-7  # 学习率过低警告
    lr_max: float = 1e-3  # 学习率过高警告

    # F1相关
    min_eval_f1: float = 0.5  # 验证F1低于50%触发警告
    f1_drop_threshold: float = 0.1  # F1下降超过10%触发警告

    # GPU相关
    gpu_memory_threshold: float = 0.95  # GPU显存使用超过95%警告
    gpu_utilization_low: float = 0.3  # GPU利用率低于30%警告（可能有问题）

    # 训练速度相关
    min_samples_per_sec: float = 0.05  # 训练速度过慢警告
    max_step_time: float = 120.0  # 单步超过2分钟警告

    # 梯度相关
    max_grad_norm: float = 10.0  # 梯度范数过大（可能爆炸）
    min_grad_norm: float = 1e-6  # 梯度范数过小（可能消失）


@dataclass
class MonitoringConfig:
    """监控配置"""

    # 基础配置
    enable_tensorboard: bool = True
    tensorboard_dir: str = "./runs"

    enable_wandb: bool = False  # Weights & Biases（可选）
    wandb_project: str = "hrdoc-training"
    wandb_entity: Optional[str] = None

    # 日志配置
    log_file: str = "./training.log"
    save_metrics_every: int = 10  # 每10步保存一次指标

    # 实时监控
    enable_live_monitoring: bool = True
    monitoring_port: int = 6006  # TensorBoard端口

    # 异常检测
    enable_anomaly_detection: bool = True
    thresholds: MonitoringThresholds = field(default_factory=MonitoringThresholds)

    # 检查点配置
    save_best_only: bool = False  # 是否只保存最佳模型
    metric_for_best_model: str = "eval_f1"  # 用于选择最佳模型的指标

    # 早停配置
    enable_early_stopping: bool = False
    early_stopping_patience: int = 5  # 连续5次eval无改善则停止
    early_stopping_threshold: float = 0.001  # 改善阈值


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = []
        self.best_metric = None
        self.no_improvement_count = 0

    def log_metrics(self, metrics: Dict, step: int):
        """记录指标"""
        metrics["step"] = step
        metrics["timestamp"] = time.time()

        # 添加系统指标
        metrics.update(self._get_system_metrics())

        self.metrics_history.append(metrics)

        # 异常检测
        if self.config.enable_anomaly_detection:
            warnings = self._check_anomalies(metrics)
            if warnings:
                self._log_warnings(warnings, step)

        return metrics

    def _get_system_metrics(self) -> Dict:
        """获取系统指标"""
        metrics = {}

        # CPU
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["memory_percent"] = psutil.virtual_memory().percent

        # GPU（可选）
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics["gpu_memory_used"] = gpu.memoryUsed
                    metrics["gpu_memory_total"] = gpu.memoryTotal
                    metrics["gpu_memory_percent"] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    metrics["gpu_utilization"] = gpu.load * 100
                    metrics["gpu_temperature"] = gpu.temperature
            except:
                pass

        return metrics

    def _check_anomalies(self, metrics: Dict) -> List[str]:
        """检测异常"""
        warnings = []
        thresholds = self.config.thresholds

        # 检查loss
        if "loss" in metrics:
            loss = metrics["loss"]

            # NaN检测
            if thresholds.loss_nan_check and (loss != loss):  # NaN检测
                warnings.append("⚠️ Loss is NaN! 训练可能已崩溃")

            # Loss突增检测
            if len(self.metrics_history) > 0:
                prev_loss = self.metrics_history[-1].get("loss")
                if prev_loss and loss > prev_loss * thresholds.max_loss_spike:
                    warnings.append(f"⚠️ Loss突然上升: {prev_loss:.4f} → {loss:.4f}")

        # 检查学习率
        if "learning_rate" in metrics:
            lr = metrics["learning_rate"]
            if lr < thresholds.lr_min:
                warnings.append(f"⚠️ 学习率过低: {lr:.2e}")
            if lr > thresholds.lr_max:
                warnings.append(f"⚠️ 学习率过高: {lr:.2e}")

        # 检查F1
        if "eval_f1" in metrics:
            f1 = metrics["eval_f1"]
            if f1 < thresholds.min_eval_f1:
                warnings.append(f"⚠️ F1过低: {f1:.4f}")

            # F1下降检测
            if self.best_metric and f1 < self.best_metric - thresholds.f1_drop_threshold:
                warnings.append(f"⚠️ F1显著下降: {self.best_metric:.4f} → {f1:.4f}")

        # 检查GPU显存
        if "gpu_memory_percent" in metrics:
            mem_pct = metrics["gpu_memory_percent"]
            if mem_pct > thresholds.gpu_memory_threshold * 100:
                warnings.append(f"⚠️ GPU显存接近上限: {mem_pct:.1f}%")

        # 检查GPU利用率
        if "gpu_utilization" in metrics:
            gpu_util = metrics["gpu_utilization"]
            if gpu_util < thresholds.gpu_utilization_low * 100:
                warnings.append(f"⚠️ GPU利用率过低: {gpu_util:.1f}% (可能IO瓶颈)")

        # 检查训练速度
        if "train_samples_per_second" in metrics:
            speed = metrics["train_samples_per_second"]
            if speed < thresholds.min_samples_per_sec:
                warnings.append(f"⚠️ 训练速度过慢: {speed:.4f} samples/sec")

        return warnings

    def _log_warnings(self, warnings: List[str], step: int):
        """记录警告"""
        print(f"\n{'='*60}")
        print(f"[Step {step}] 检测到异常:")
        for warning in warnings:
            print(f"  {warning}")
        print(f"{'='*60}\n")

        # 写入日志文件
        if self.config.log_file:
            with open(self.config.log_file, "a") as f:
                f.write(f"\n[Step {step}] Warnings:\n")
                for warning in warnings:
                    f.write(f"  {warning}\n")

    def check_early_stopping(self, eval_metric: float) -> bool:
        """检查是否应该早停"""
        if not self.config.enable_early_stopping:
            return False

        if self.best_metric is None:
            self.best_metric = eval_metric
            return False

        # 检查是否有改善
        improvement = eval_metric - self.best_metric
        if improvement > self.config.early_stopping_threshold:
            self.best_metric = eval_metric
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.config.early_stopping_patience:
                print(f"\n⚠️ 早停触发: {self.config.early_stopping_patience}次eval无改善")
                return True

        return False

    def save_metrics(self, filepath: str):
        """保存指标历史"""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def generate_report(self) -> str:
        """生成训练报告"""
        if not self.metrics_history:
            return "暂无训练数据"

        report = []
        report.append("=" * 60)
        report.append("训练监控报告")
        report.append("=" * 60)

        # 最新指标
        latest = self.metrics_history[-1]
        report.append("\n最新指标:")
        for key, value in latest.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")

        # 最佳指标
        if self.best_metric:
            report.append(f"\n最佳 {self.config.metric_for_best_model}: {self.best_metric:.4f}")

        # GPU状态
        if "gpu_memory_percent" in latest:
            report.append(f"\nGPU状态:")
            report.append(f"  显存使用: {latest['gpu_memory_percent']:.1f}%")
            report.append(f"  GPU利用率: {latest.get('gpu_utilization', 0):.1f}%")

        report.append("=" * 60)
        return "\n".join(report)


def get_monitoring_config(env: str = "cloud") -> MonitoringConfig:
    """获取监控配置"""

    if env == "cloud":
        # 云服务器：全面监控
        return MonitoringConfig(
            enable_tensorboard=True,
            tensorboard_dir="./runs",
            enable_wandb=False,  # 可选开启
            log_file="./training_cloud.log",
            save_metrics_every=10,
            enable_live_monitoring=True,
            enable_anomaly_detection=True,
            enable_early_stopping=False,  # 论文复现不建议早停
            thresholds=MonitoringThresholds(
                max_loss_spike=2.0,
                min_eval_f1=0.5,
                gpu_memory_threshold=0.95,
            )
        )

    elif env == "local":
        # 本机：轻量监控
        return MonitoringConfig(
            enable_tensorboard=True,
            tensorboard_dir="./runs",
            enable_wandb=False,
            log_file="./training_local.log",
            save_metrics_every=20,
            enable_live_monitoring=True,
            enable_anomaly_detection=True,
            enable_early_stopping=True,  # 本机测试可早停
            early_stopping_patience=3,
        )

    else:
        # 默认配置
        return MonitoringConfig()


if __name__ == "__main__":
    # 测试监控系统
    config = get_monitoring_config("cloud")
    monitor = TrainingMonitor(config)

    # 模拟训练指标
    for step in range(10):
        metrics = {
            "loss": 2.0 - step * 0.15,
            "learning_rate": 5e-5 * (1 - step/10),
            "eval_f1": 0.6 + step * 0.03,
            "train_samples_per_second": 0.1,
        }

        monitor.log_metrics(metrics, step)

    # 生成报告
    print(monitor.generate_report())
