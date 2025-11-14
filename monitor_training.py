#!/usr/bin/env python
# coding=utf-8
"""
实时监控训练进度
在云服务器上运行训练时，使用此脚本查看实时状态
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


def parse_trainer_state(state_file: str) -> dict:
    """解析trainer_state.json"""
    if not os.path.exists(state_file):
        return None

    with open(state_file, 'r') as f:
        state = json.load(f)

    return state


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining_time(state: dict) -> str:
    """估算剩余时间"""
    if "train_runtime" not in state or "global_step" not in state:
        return "未知"

    runtime = state["train_runtime"]
    current_step = state["global_step"]
    max_steps = state.get("max_steps", 0)

    if max_steps == 0 or current_step == 0:
        return "未知"

    avg_time_per_step = runtime / current_step
    remaining_steps = max_steps - current_step
    remaining_seconds = avg_time_per_step * remaining_steps

    return format_time(remaining_seconds)


def get_latest_metrics(state: dict) -> dict:
    """获取最新指标"""
    if "log_history" not in state or len(state["log_history"]) == 0:
        return {}

    return state["log_history"][-1]


def get_best_metrics(state: dict) -> dict:
    """获取最佳指标"""
    log_history = state.get("log_history", [])

    best_metrics = {}
    best_f1 = -1
    best_loss = float('inf')

    for log in log_history:
        # 查找最佳F1
        for key in ["eval_f1", "eval_overall_f1"]:
            if key in log and log[key] > best_f1:
                best_f1 = log[key]
                best_metrics["best_f1"] = best_f1
                best_metrics["best_f1_step"] = log.get("step", 0)

        # 查找最低loss
        if "loss" in log and log["loss"] < best_loss:
            best_loss = log["loss"]
            best_metrics["best_loss"] = best_loss
            best_metrics["best_loss_step"] = log.get("step", 0)

    return best_metrics


def print_status(output_dir: str, clear_screen: bool = True):
    """打印训练状态"""
    state_file = os.path.join(output_dir, "trainer_state.json")

    if clear_screen:
        os.system('clear' if os.name != 'nt' else 'cls')

    print("=" * 80)
    print(f"HRDoc 训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    state = parse_trainer_state(state_file)

    if state is None:
        print("\n⚠️  训练尚未开始或状态文件不存在")
        print(f"   监控目录: {output_dir}")
        return False

    # 基本信息
    print(f"\n📊 训练进度:")
    current_step = state.get("global_step", 0)
    max_steps = state.get("max_steps", 0)
    epoch = state.get("epoch", 0)

    if max_steps > 0:
        progress = (current_step / max_steps) * 100
        print(f"   当前步数: {current_step:,} / {max_steps:,} ({progress:.1f}%)")
    else:
        print(f"   当前步数: {current_step:,}")

    print(f"   当前Epoch: {epoch:.2f}")

    # 时间信息
    if "train_runtime" in state:
        runtime = state["train_runtime"]
        print(f"\n⏱️  时间信息:")
        print(f"   已运行时间: {format_time(runtime)}")
        remaining = estimate_remaining_time(state)
        print(f"   预计剩余: {remaining}")

        # 训练速度
        if current_step > 0:
            samples_per_sec = state.get("train_samples_per_second", 0)
            steps_per_sec = current_step / runtime
            print(f"   训练速度: {samples_per_sec:.4f} samples/sec, {steps_per_sec:.4f} steps/sec")

    # 最新指标
    latest_metrics = get_latest_metrics(state)
    if latest_metrics:
        print(f"\n📈 最新指标 (Step {latest_metrics.get('step', 0)}):")

        # Loss
        if "loss" in latest_metrics:
            print(f"   Loss: {latest_metrics['loss']:.4f}")

        # 学习率
        if "learning_rate" in latest_metrics:
            print(f"   Learning Rate: {latest_metrics['learning_rate']:.2e}")

        # 评估指标
        for key in ["eval_f1", "eval_precision", "eval_recall", "eval_accuracy"]:
            if key in latest_metrics:
                metric_name = key.replace("eval_", "").upper()
                print(f"   {metric_name}: {latest_metrics[key]:.4f}")

    # 最佳指标
    best_metrics = get_best_metrics(state)
    if best_metrics:
        print(f"\n🏆 最佳指标:")
        if "best_f1" in best_metrics:
            print(f"   Best F1: {best_metrics['best_f1']:.4f} (Step {best_metrics['best_f1_step']})")
        if "best_loss" in best_metrics:
            print(f"   Best Loss: {best_metrics['best_loss']:.4f} (Step {best_metrics['best_loss_step']})")

    # 检查异常
    warnings = check_anomalies(state, latest_metrics)
    if warnings:
        print(f"\n⚠️  异常检测:")
        for warning in warnings:
            print(f"   {warning}")

    print("\n" + "=" * 80)
    print("按 Ctrl+C 退出监控")
    print("=" * 80)

    return True


def check_anomalies(state: dict, latest_metrics: dict) -> list:
    """检测训练异常"""
    warnings = []

    # 检查loss是否为NaN
    if "loss" in latest_metrics:
        loss = latest_metrics["loss"]
        if loss != loss:  # NaN检测
            warnings.append("⚠️ Loss is NaN! 训练可能已崩溃")
        elif loss > 10.0:
            warnings.append(f"⚠️ Loss过高: {loss:.4f}")

    # 检查F1是否过低
    if "eval_f1" in latest_metrics:
        f1 = latest_metrics["eval_f1"]
        current_step = state.get("global_step", 0)
        if current_step > 1000 and f1 < 0.5:
            warnings.append(f"⚠️ F1过低: {f1:.4f} (Step {current_step})")

    # 检查训练速度
    if "train_samples_per_second" in latest_metrics:
        speed = latest_metrics["train_samples_per_second"]
        if speed < 0.05:
            warnings.append(f"⚠️ 训练速度过慢: {speed:.4f} samples/sec")

    # 检查是否卡住
    current_step = state.get("global_step", 0)
    max_steps = state.get("max_steps", 0)
    if max_steps > 0 and current_step == 0:
        warnings.append("⚠️ 训练尚未开始")

    return warnings


def monitor_loop(output_dir: str, interval: int = 10):
    """监控循环"""
    try:
        while True:
            success = print_status(output_dir, clear_screen=True)
            if not success:
                print(f"\n等待训练开始... (每{interval}秒刷新)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")


def main():
    parser = argparse.ArgumentParser(description="实时监控HRDoc训练进度")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/hrdoc_simple_full",
        help="训练输出目录"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="刷新间隔（秒）"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只显示一次，不循环"
    )

    args = parser.parse_args()

    if args.once:
        print_status(args.output_dir, clear_screen=False)
    else:
        monitor_loop(args.output_dir, args.interval)


if __name__ == "__main__":
    main()
