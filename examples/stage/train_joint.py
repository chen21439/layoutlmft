#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练脚本 (基于 HuggingFace Trainer)

支持三种训练模式：
- stage1: 只训练 Stage 1 分类模型
- stage34: 冻结 Stage 1，只训练 Stage 3/4
- joint: 端到端联合训练所有阶段（默认）

端到端训练流程:
1. Stage 1: LayoutXLM 分类 (产生分类 loss + hidden states)
2. Stage 2: 从 hidden states 提取 line-level 特征
3. Stage 3: ParentFinder 训练 (产生 parent loss) - 使用 ParentFinderGRU + Soft-Mask
4. Stage 4: RelationClassifier 训练 (产生 relation loss)

总 Loss = λ1 * L_cls + λ2 * L_par + λ3 * L_rel (论文公式)

Usage:
    # 联合训练（默认）
    python examples/stage/train_joint.py --env test --dataset hrds

    # 只训练 Stage 1
    python examples/stage/train_joint.py --env test --dataset hrds --mode stage1

    # 只训练 Stage 3/4（需要预训练好的 Stage 1）
    python examples/stage/train_joint.py --env test --dataset hrds --mode stage34

    # 快速测试
    python examples/stage/train_joint.py --env test --dataset hrds --quick
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ==================== GPU 设置（必须在 import torch 之前）====================
def _setup_gpu_early():
    """在 import torch 之前设置 GPU，避免 DataParallel 问题"""
    gpu_id = None
    env = "test"
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return

    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, PROJECT_ROOT)
        from configs.config_loader import load_config
        config = load_config(env)
        if config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
    except Exception:
        pass

_setup_gpu_early()

import torch
from transformers import HfArgumentParser, TrainingArguments, set_seed

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)

import layoutlmft.data.datasets.hrdoc
from layoutlmft.data.labels import LABEL_LIST, NUM_LABELS, get_id2label, get_label2id
from layoutlmft.models.layoutxlm import (
    LayoutXLMForTokenClassification,
    LayoutXLMConfig,
    LayoutXLMTokenizerFast,
)
from layoutlmft.models.relation_classifier import (
    LineFeatureExtractor,
    MultiClassRelationClassifier,
    NUM_RELATIONS,
)

from joint_data_collator import HRDocJointDataCollator, HRDocDocumentLevelCollator
from train_parent_finder import SimpleParentFinder, ParentFinderGRU
from tasks.parent_finding import build_child_parent_matrix_from_dataset
from models.joint_model import JointModel
from engines import (
    JointTrainer,
    AMPDiagnosticCallback,
    JointLoggingCallback,
    E2EEvaluationCallback,
    Stage1EvaluationCallback,
)
from util.config_setup import load_config_and_setup
from util.experiment_manager import ensure_experiment
from data import HRDocDataLoader, HRDocDataLoaderConfig

logger = logging.getLogger(__name__)


# ==================== 参数定义 ====================

@dataclass
class JointModelArguments:
    """联合模型参数"""
    model_name_or_path: str = field(
        default=None, metadata={"help": "Stage1 base model or joint checkpoint path"}
    )
    mode: str = field(
        default="joint",
        metadata={"help": "Training mode: stage1 (only Stage1), stage34 (freeze Stage1), joint (all stages)"}
    )
    use_gru: bool = field(default=True, metadata={"help": "Use GRU decoder (paper method)"})
    use_soft_mask: bool = field(default=True, metadata={"help": "Use Soft-Mask (paper method)"})
    use_focal_loss: bool = field(default=True, metadata={"help": "Use Focal Loss"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Classification loss weight"})
    lambda_parent: float = field(default=1.0, metadata={"help": "Parent loss weight"})
    lambda_rel: float = field(default=1.0, metadata={"help": "Relation loss weight"})
    section_parent_weight: float = field(default=1.0, metadata={"help": "Weight for section type in parent loss"})
    stage1_micro_batch_size: int = field(default=1, metadata={"help": "Stage1 micro-batch size"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing for Stage1"})
    freeze_visual: bool = field(default=False, metadata={"help": "Freeze visual encoder (ResNet)"})
    use_gt_class: bool = field(default=False, metadata={"help": "Use GT class labels for Stage3/4"})


@dataclass
class JointDataArguments:
    """联合训练数据参数"""
    env: str = field(default="test", metadata={"help": "Environment: dev, test"})
    dataset: str = field(default="hrds", metadata={"help": "Dataset: hrds, hrdh, tender"})
    gpu: Optional[str] = field(default=None, metadata={"help": "GPU ID to use"})
    covmatch: Optional[str] = field(default=None, metadata={"help": "Covmatch split name"})
    max_train_samples: int = field(default=-1, metadata={"help": "Max train samples (-1 for all)"})
    max_eval_samples: int = field(default=-1, metadata={"help": "Max eval samples (-1 for all)"})
    force_rebuild: bool = field(default=False, metadata={"help": "Force rebuild dataset"})
    document_level: bool = field(default=True, metadata={"help": "Use document-level batching"})


@dataclass
class JointTrainingArguments(TrainingArguments):
    """扩展 HuggingFace TrainingArguments"""
    output_dir: str = field(default="./output/joint", metadata={"help": "Output directory"})
    report_to: str = field(default="none", metadata={"help": "Disable TensorBoard"})
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation steps"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum checkpoints to keep"})
    logging_steps: int = field(default=100, metadata={"help": "Logging steps"})

    # 实验管理
    exp: str = field(default=None, metadata={"help": "Experiment ID"})
    new_exp: str = field(default="", metadata={"help": "Create new experiment"})
    exp_name: str = field(default="", metadata={"help": "Experiment name"})
    artifact_dir: str = field(default="", metadata={"help": "Artifact root directory (overrides config.paths.output_dir)"})

    # 快速测试
    quick: bool = field(default=False, metadata={"help": "Quick test mode"})
    min_steps: int = field(default=1000, metadata={"help": "Minimum training steps"})

    # Stage 3/4 学习率
    learning_rate_stage34: float = field(default=5e-4, metadata={"help": "Learning rate for Stage 3/4"})

    # 功能开关
    eval_before_train: bool = field(default=False, metadata={"help": "Run evaluation before training"})
    save_predictions: bool = field(default=False, metadata={"help": "Save prediction results"})
    dry_run: bool = field(default=False, metadata={"help": "Dry run mode"})

    # DataLoader 设置
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of dataloader workers"})
    dataloader_pin_memory: bool = field(default=False, metadata={"help": "Pin memory"})
    dataloader_drop_last: bool = field(default=False, metadata={"help": "Drop last incomplete batch"})


# ==================== 数据准备 ====================

def prepare_datasets(tokenizer, data_args, training_args, model_args):
    """准备训练和评估数据集"""
    num_workers = 1 if data_args.max_train_samples and data_args.max_train_samples < 100 else 4

    # 所有模式都需要 line info（line-level 分类）
    include_line_info = True

    loader_config = HRDocDataLoaderConfig(
        data_dir=os.environ.get("HRDOC_DATA_DIR"),
        dataset_name=data_args.dataset,
        max_length=512,
        preprocessing_num_workers=num_workers,
        max_train_samples=data_args.max_train_samples if data_args.max_train_samples > 0 else None,
        max_val_samples=data_args.max_eval_samples if data_args.max_eval_samples > 0 else None,
        force_rebuild=data_args.force_rebuild,
        document_level=data_args.document_level,
    )

    data_loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=loader_config,
        include_line_info=include_line_info,
    )

    data_loader.load_raw_datasets()
    tokenized_datasets = data_loader.prepare_datasets()

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    raw_train_dataset = data_loader._raw_datasets.get("train")

    logger.info(f"Train dataset: {len(train_dataset) if train_dataset else 0} samples")
    logger.info(f"Eval dataset: {len(eval_dataset) if eval_dataset else 0} samples")
    logger.info(f"include_line_info: {include_line_info} (mode={model_args.mode})")

    return train_dataset, eval_dataset, raw_train_dataset


def compute_line_level_metrics(eval_pred):
    """计算 line-level 评估指标"""
    from sklearn.metrics import accuracy_score, f1_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    all_preds, all_labels = [], []
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:
                all_preds.append(p)
                all_labels.append(l)

    if not all_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
    }


# ==================== 模型构建 ====================

def build_model(model_args, config, raw_train_dataset, device):
    """构建联合模型"""
    # 检测 checkpoint 类型
    joint_model_path = None
    specified_path = model_args.model_name_or_path

    stage1_subdir = os.path.join(specified_path, "stage1")
    joint_pytorch_model = os.path.join(specified_path, "pytorch_model.bin")

    if os.path.isfile(os.path.join(stage1_subdir, "config.json")) and os.path.isfile(joint_pytorch_model):
        stage1_path = stage1_subdir
        joint_model_path = specified_path
        logger.info(f"Loading from joint checkpoint: {specified_path}")
    else:
        stage1_path = specified_path
        logger.info(f"Loading Stage 1 from: {stage1_path}")

    # Stage 1: LayoutXLM
    stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    if joint_model_path:
        stage1_model = LayoutXLMForTokenClassification(config=stage1_config)
    else:
        stage1_model = LayoutXLMForTokenClassification.from_pretrained(stage1_path, config=stage1_config)

    if model_args.gradient_checkpointing:
        stage1_model.config.gradient_checkpointing = True
        logger.info("Gradient Checkpointing ENABLED for Stage1")

    # Stage 2: Feature Extractor
    feature_extractor = LineFeatureExtractor()

    # Stage 3: ParentFinder
    if model_args.use_gru:
        logger.info("Using ParentFinderGRU (paper method)")
        stage3_model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=NUM_LABELS,
            dropout=0.1,
            use_soft_mask=model_args.use_soft_mask,
        )

        # 只在非 joint checkpoint 加载且 mode 需要 stage3 时构建 M_cp
        if model_args.use_soft_mask and not joint_model_path and model_args.mode != "stage1":
            logger.info("Building Child-Parent Distribution Matrix (M_cp)...")
            cp_matrix = build_child_parent_matrix_from_dataset(raw_train_dataset, num_classes=NUM_LABELS)
            stage3_model.set_child_parent_matrix(cp_matrix.get_tensor(device))
            logger.info("M_cp initialized successfully")
    else:
        logger.info("Using SimpleParentFinder")
        stage3_model = SimpleParentFinder(hidden_size=768, dropout=0.1)

    # Stage 4: RelationClassifier
    gru_hidden_size = 512 if model_args.use_gru else 768
    stage4_model = MultiClassRelationClassifier(
        hidden_size=gru_hidden_size,
        num_relations=NUM_RELATIONS,
        use_geometry=False,
        dropout=0.1,
    )

    # 根据 mode 设置参数
    stage1_no_grad = model_args.mode == "stage34"
    disable_stage34 = model_args.mode == "stage1"

    if disable_stage34:
        # stage1 模式：只训练 Stage1，禁用 Stage3/4
        lambda_cls = model_args.lambda_cls
        lambda_parent = 0.0
        lambda_rel = 0.0
    elif stage1_no_grad:
        # stage34 模式：冻结 Stage1，只训练 Stage3/4
        lambda_cls = 0.0  # 不计算分类 loss
        lambda_parent = model_args.lambda_parent
        lambda_rel = model_args.lambda_rel
    else:
        # joint 模式：全部训练
        lambda_cls = model_args.lambda_cls
        lambda_parent = model_args.lambda_parent
        lambda_rel = model_args.lambda_rel

    # 创建联合模型
    model = JointModel(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        feature_extractor=feature_extractor,
        lambda_cls=lambda_cls,
        lambda_parent=lambda_parent,
        lambda_rel=lambda_rel,
        section_parent_weight=model_args.section_parent_weight,
        use_focal_loss=model_args.use_focal_loss,
        use_gru=model_args.use_gru,
        stage1_micro_batch_size=model_args.stage1_micro_batch_size,
        freeze_visual=model_args.freeze_visual,
        stage1_no_grad=stage1_no_grad,
        use_gt_class=model_args.use_gt_class,
    )

    if model_args.mode == "stage1":
        logger.info(f"Mode: stage1 (Stage 3/4 disabled)")
    elif model_args.mode == "stage34":
        logger.info(f"Mode: stage34 (Stage 1 frozen, no_grad=True)")
    else:
        logger.info(f"Mode: joint (all stages enabled)")

    # 加载 joint checkpoint 权重
    if joint_model_path:
        joint_weights_path = os.path.join(joint_model_path, "pytorch_model.bin")
        logger.info(f"Loading joint model weights from: {joint_weights_path}")
        state_dict = torch.load(joint_weights_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

    # 打印参数量
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage3_params = sum(p.numel() for p in model.stage3.parameters())
    stage4_params = sum(p.numel() for p in model.stage4.parameters())
    logger.info(f"Parameters: Stage1={stage1_params:,}, Stage3={stage3_params:,}, Stage4={stage4_params:,}")

    return model, joint_model_path


# ==================== 主函数 ====================

def main():
    # 解析参数
    parser = HfArgumentParser((JointModelArguments, JointDataArguments, JointTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 验证 mode 参数
    if model_args.mode not in ("stage1", "stage34", "joint"):
        raise ValueError(f"Invalid mode: {model_args.mode}. Must be one of: stage1, stage34, joint")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # 加载配置
    stage_name = "stage1" if model_args.mode == "stage1" else "joint"
    config, data_dir, exp_manager = load_config_and_setup(data_args, training_args, model_args, stage_name)

    # Quick 模式
    if training_args.quick:
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE")
        logger.info("=" * 60)
        if data_args.max_train_samples == -1:
            data_args.max_train_samples = 10
        if data_args.max_eval_samples == -1:
            data_args.max_eval_samples = 10
        if training_args.max_steps <= 0 or training_args.max_steps > 20:
            training_args.max_steps = 20
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 10
        training_args.save_steps = 10
        training_args.logging_steps = 5

    # Document-level 模式调整
    if data_args.document_level and not training_args.quick:
        training_args.eval_steps = 25
        training_args.save_steps = 25
        training_args.logging_steps = 10
    # Page-level 模式调整（非 quick 模式）
    elif not data_args.document_level and not training_args.quick:
        training_args.logging_steps = 10
        training_args.eval_steps = 100

    set_seed(training_args.seed)

    # 打印配置
    logger.info("=" * 60)
    logger.info(f"Training Mode: {model_args.mode.upper()}")
    logger.info("=" * 60)
    logger.info(f"Environment:    {data_args.env}")
    logger.info(f"Dataset:        {data_args.dataset}")
    logger.info(f"Model Path:     {model_args.model_name_or_path}")
    logger.info(f"Output Dir:     {training_args.output_dir}")
    if model_args.mode == "stage1":
        logger.info(f"Loss Weights:   cls={model_args.lambda_cls} (Stage3/4 disabled)")
    elif model_args.mode == "stage34":
        logger.info(f"Loss Weights:   cls=0.0 (frozen), parent={model_args.lambda_parent}, rel={model_args.lambda_rel}")
    else:
        logger.info(f"Loss Weights:   cls={model_args.lambda_cls}, parent={model_args.lambda_parent}, rel={model_args.lambda_rel}")
    logger.info("=" * 60)

    if training_args.dry_run:
        logger.info("[Dry run mode - exiting]")
        return

    os.makedirs(training_args.output_dir, exist_ok=True)
    exp_manager.mark_stage_started(training_args.exp, stage_name, data_args.dataset)

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_path = model_args.model_name_or_path
    tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        tokenizer_path = config.model.local_path or config.model.name_or_path
        logger.warning(f"tokenizer.json not found, using base model: {tokenizer_path}")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(tokenizer_path)

    # 准备数据集
    logger.info("Preparing datasets...")
    train_dataset, eval_dataset, raw_train_dataset = prepare_datasets(tokenizer, data_args, training_args, model_args)

    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("No eval dataset, disabling evaluation")
        training_args.evaluation_strategy = "no"
        training_args.eval_steps = None

    # 自动调整 epoch
    if not training_args.quick and train_dataset is not None and training_args.max_steps <= 0:
        effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        steps_per_epoch = max(len(train_dataset) // effective_batch_size, 1)
        current_total_steps = steps_per_epoch * int(training_args.num_train_epochs)

        if current_total_steps < training_args.min_steps:
            required_epochs = (training_args.min_steps + steps_per_epoch - 1) // steps_per_epoch
            logger.info(f"Auto-adjusting epochs: {int(training_args.num_train_epochs)} -> {required_epochs}")
            training_args.num_train_epochs = float(required_epochs)

    # Data collator（所有模式都使用 HRDocJointDataCollator 以支持 line_ids）
    if data_args.document_level:
        data_collator = HRDocDocumentLevelCollator(tokenizer=tokenizer, padding=True, max_length=512)
        logger.info("Using DOCUMENT-LEVEL collator")
    else:
        data_collator = HRDocJointDataCollator(tokenizer=tokenizer, padding=True, max_length=512)
        logger.info("Using PAGE-LEVEL collator")

    # 构建模型
    logger.info("Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, joint_model_path = build_model(model_args, config, raw_train_dataset, device)

    # 创建评估 DataLoader
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
        )

    # 创建 callbacks（根据 mode 选择）
    callbacks = [JointLoggingCallback(mode=model_args.mode), AMPDiagnosticCallback()]
    if eval_dataloader is not None:
        if model_args.mode == "stage1":
            # Stage1 模式只评估分类指标
            callbacks.append(Stage1EvaluationCallback(eval_dataloader=eval_dataloader))
        else:
            # Joint/Stage34 模式使用 E2E 评估
            runs_dir = os.path.join(STAGE_ROOT, "runs") if training_args.save_predictions else None
            callbacks.append(E2EEvaluationCallback(
                eval_dataloader=eval_dataloader,
                data_collator=data_collator,
                compute_teds=not training_args.quick,
                save_predictions=training_args.save_predictions,
                output_dir=runs_dir,
            ))

    # 创建 Trainer
    trainer = JointTrainer(
        model=model,
        args=training_args,
        model_args=model_args,
        learning_rate_stage34=training_args.learning_rate_stage34,
        document_level=data_args.document_level,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_line_level_metrics,
        callbacks=callbacks,
    )

    # 训练前评估
    if training_args.eval_before_train and eval_dataset is not None:
        logger.info("Running pre-training evaluation...")
        trainer.evaluate()

    # 训练
    logger.info("Starting training...")
    resume_from = None
    if training_args.resume_from_checkpoint:
        resume_from = training_args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {resume_from}")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # 保存
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 最终评估
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        trainer.evaluate()

    # 更新实验状态
    exp_manager.mark_stage_completed(
        training_args.exp, stage_name, data_args.dataset,
        best_checkpoint="final",
        metrics={"train_loss": float(metrics.get("train_loss", 0.0))},
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
