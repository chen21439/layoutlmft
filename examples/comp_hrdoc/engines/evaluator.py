"""评估管线

Detect-Order-Construct 模型的评估引擎。
支持从 JSON 文件或模型输出进行评估。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..metrics import (
    TEDSMetric,
    ClassificationMetric,
    ReadingOrderMetric,
    IntraRegionOrderMetric,
    EvaluationReport,
    save_metrics,
    format_metrics,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    # 数据路径
    gt_folder: str = ""
    pred_folder: str = ""
    output_dir: str = "./eval_output"

    # 评估选项
    eval_classification: bool = True
    eval_reading_order: bool = True
    eval_structure: bool = True

    # 并行选项
    num_workers: int = 4


class DOCEvaluator:
    """Detect-Order-Construct 评估器

    支持评估:
    - Detect: 逻辑角色分类
    - Order: 阅读顺序预测
    - Construct: 层级文档结构
    """

    def __init__(self, config: EvaluatorConfig = None):
        self.config = config or EvaluatorConfig()
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.classification_metric = ClassificationMetric()
        self.reading_order_metric = ReadingOrderMetric()
        self.structure_metric = TEDSMetric()
        self.intra_order_metric = IntraRegionOrderMetric()

    def evaluate_from_folders(
        self,
        gt_folder: str,
        pred_folder: str,
        output_dir: str = None,
    ) -> EvaluationReport:
        """从文件夹评估

        Args:
            gt_folder: 真实标签文件夹 (JSON 文件)
            pred_folder: 预测结果文件夹 (JSON 文件)
            output_dir: 输出目录

        Returns:
            评估报告
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 检查文件一致性
        gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.json'))
        pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.json'))

        if gt_files != pred_files:
            missing_in_pred = gt_files - pred_files
            missing_in_gt = pred_files - gt_files
            if missing_in_pred:
                logger.warning(f"预测缺少文件: {missing_in_pred}")
            if missing_in_gt:
                logger.warning(f"GT缺少文件: {missing_in_gt}")
            gt_files = gt_files & pred_files

        logger.info(f"评估 {len(gt_files)} 个文件")

        # 评估每个文件
        self.reset()
        for json_file in tqdm(sorted(gt_files), desc="评估中"):
            gt_path = os.path.join(gt_folder, json_file)
            pred_path = os.path.join(pred_folder, json_file)

            try:
                self._evaluate_single_file(gt_path, pred_path, json_file)
            except Exception as e:
                logger.error(f"评估 {json_file} 失败: {e}")

        # 生成报告
        report = self._generate_report()

        # 保存结果
        self._save_results(output_dir, report)

        return report

    def evaluate_from_folders_parallel(
        self,
        gt_folder: str,
        pred_folder: str,
        output_dir: str = None,
        num_workers: int = None,
    ) -> EvaluationReport:
        """并行从文件夹评估

        Args:
            gt_folder: 真实标签文件夹
            pred_folder: 预测结果文件夹
            output_dir: 输出目录
            num_workers: 并行工作进程数

        Returns:
            评估报告
        """
        output_dir = output_dir or self.config.output_dir
        num_workers = num_workers or self.config.num_workers
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件列表
        gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.json'))
        pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.json'))
        common_files = sorted(gt_files & pred_files)

        logger.info(f"并行评估 {len(common_files)} 个文件 (workers={num_workers})")

        # 并行处理
        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for json_file in common_files:
                gt_path = os.path.join(gt_folder, json_file)
                pred_path = os.path.join(pred_folder, json_file)
                future = executor.submit(_evaluate_file_worker, gt_path, pred_path, json_file)
                futures[future] = json_file

            for future in tqdm(as_completed(futures), total=len(futures), desc="评估中"):
                json_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"评估 {json_file} 失败: {e}")

        # 聚合结果
        self.reset()
        for result in all_results:
            self._aggregate_result(result)

        # 生成报告
        report = self._generate_report()
        self._save_results(output_dir, report)

        return report

    def _evaluate_single_file(
        self,
        gt_path: str,
        pred_path: str,
        sample_id: str,
    ):
        """评估单个文件"""
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        if len(gt_data) != len(pred_data):
            logger.warning(f"{sample_id}: GT({len(gt_data)}) != Pred({len(pred_data)})")
            return

        # 提取数据
        gt_texts = [f"{item['class']}:{item.get('text', '')}" for item in gt_data]
        gt_parent_ids = [item['parent_id'] for item in gt_data]
        gt_relations = [item['relation'] for item in gt_data]
        gt_classes = [item['class'] for item in gt_data]

        pred_texts = [f"{item['class']}:{item.get('text', '')}" for item in pred_data]
        pred_parent_ids = [item['parent_id'] for item in pred_data]
        pred_relations = [item['relation'] for item in pred_data]
        pred_classes = [item['class'] for item in pred_data]

        # 分类评估
        if self.config.eval_classification:
            self.classification_metric.update_from_classes(
                pred_classes, gt_classes,
                pred_context=pred_data, gt_context=gt_data
            )

        # 阅读顺序评估
        if self.config.eval_reading_order:
            self.reading_order_metric.update(
                pred_texts, pred_parent_ids, pred_relations,
                gt_texts, gt_parent_ids, gt_relations,
                sample_id=sample_id
            )

        # 层级结构评估
        if self.config.eval_structure:
            self.structure_metric.update(
                pred_texts, pred_parent_ids, pred_relations,
                gt_texts, gt_parent_ids, gt_relations,
                sample_id=sample_id
            )

    def _aggregate_result(self, result: Dict):
        """聚合单个文件的评估结果"""
        if 'classification' in result:
            cls_result = result['classification']
            self.classification_metric.all_preds.extend(cls_result['preds'])
            self.classification_metric.all_labels.extend(cls_result['labels'])

        if 'reading_order' in result:
            ro_result = result['reading_order']
            self.reading_order_metric.teds_list.append(ro_result['teds'])
            self.reading_order_metric.distance_list.append(ro_result['distance'])
            self.reading_order_metric.gt_nodes_list.append(ro_result['gt_nodes'])
            self.reading_order_metric.pred_nodes_list.append(ro_result['pred_nodes'])
            self.reading_order_metric.teds_floating_list.append(ro_result.get('teds_floating', 1.0))
            self.reading_order_metric.distance_floating_list.append(ro_result.get('distance_floating', 0))
            self.reading_order_metric.gt_floating_nodes_list.append(ro_result.get('gt_floating_nodes', 0))
            self.reading_order_metric.pred_floating_nodes_list.append(ro_result.get('pred_floating_nodes', 0))

        if 'structure' in result:
            struct_result = result['structure']
            self.structure_metric.teds_list.append(struct_result['teds'])
            self.structure_metric.distance_list.append(struct_result['distance'])
            self.structure_metric.gt_nodes_list.append(struct_result['gt_nodes'])
            self.structure_metric.pred_nodes_list.append(struct_result['pred_nodes'])

    def _generate_report(self) -> EvaluationReport:
        """生成评估报告"""
        report = EvaluationReport()

        # 分类指标
        if self.config.eval_classification:
            cls_result = self.classification_metric.compute()
            report.detect_classification_macro_f1 = cls_result.macro_f1
            report.detect_classification_micro_f1 = cls_result.micro_f1

        # 阅读顺序指标
        if self.config.eval_reading_order:
            ro_result = self.reading_order_metric.compute()
            report.order_main_macro_teds = ro_result.macro_teds
            report.order_main_micro_teds = ro_result.micro_teds
            report.order_floating_macro_teds = ro_result.macro_teds_floating
            report.order_floating_micro_teds = ro_result.micro_teds_floating

        # 层级结构指标
        if self.config.eval_structure:
            struct_result = self.structure_metric.compute()
            report.construct_macro_teds = struct_result.macro_teds
            report.construct_micro_teds = struct_result.micro_teds

        report.num_samples = max(
            len(self.classification_metric.all_preds),
            len(self.reading_order_metric.teds_list),
            len(self.structure_metric.teds_list),
        )

        return report

    def _save_results(self, output_dir: str, report: EvaluationReport):
        """保存评估结果"""
        # 保存报告
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        save_metrics(report.to_dict(), report_path)
        logger.info(f"报告已保存到: {report_path}")

        # 打印报告
        print(report)

        # 保存详细结果
        if self.config.eval_classification:
            cls_result = self.classification_metric.compute()
            cls_path = os.path.join(output_dir, 'classification_detail.json')
            save_metrics({
                'macro_f1': cls_result.macro_f1,
                'micro_f1': cls_result.micro_f1,
                'accuracy': cls_result.accuracy,
                'per_class_f1': cls_result.per_class_f1,
                'per_class_precision': cls_result.per_class_precision,
                'per_class_recall': cls_result.per_class_recall,
            }, cls_path)

        if self.config.eval_reading_order:
            ro_result = self.reading_order_metric.compute()
            ro_path = os.path.join(output_dir, 'reading_order_detail.json')
            save_metrics({
                'macro_teds': ro_result.macro_teds,
                'micro_teds': ro_result.micro_teds,
                'macro_teds_floating': ro_result.macro_teds_floating,
                'micro_teds_floating': ro_result.micro_teds_floating,
                'per_sample': ro_result.per_sample,
            }, ro_path)

        if self.config.eval_structure:
            struct_result = self.structure_metric.compute()
            struct_path = os.path.join(output_dir, 'structure_detail.json')
            save_metrics({
                'macro_teds': struct_result.macro_teds,
                'micro_teds': struct_result.micro_teds,
                'per_sample': struct_result.per_sample,
            }, struct_path)


def _evaluate_file_worker(gt_path: str, pred_path: str, sample_id: str) -> Optional[Dict]:
    """并行评估工作函数"""
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        if len(gt_data) != len(pred_data):
            return None

        result = {'sample_id': sample_id}

        # 分类
        from ..metrics.classification import normalize_class, CLASS2ID
        gt_classes = [normalize_class(item['class'], gt_data, item) for item in gt_data]
        pred_classes = [normalize_class(item['class'], pred_data, item) for item in pred_data]
        result['classification'] = {
            'preds': [CLASS2ID.get(c, 6) for c in pred_classes],
            'labels': [CLASS2ID.get(c, 6) for c in gt_classes],
        }

        # 阅读顺序和结构
        from ..metrics.teds import (
            generate_doc_tree, transfer_tree_to_chain, split_chain_by_tag,
            sequence_edit_distance, min_edit_distance_between_groups, tree_edit_distance
        )

        gt_texts = [f"{item['class']}:{item.get('text', '')}" for item in gt_data]
        gt_parent_ids = [item['parent_id'] for item in gt_data]
        gt_relations = [item['relation'] for item in gt_data]

        pred_texts = [f"{item['class']}:{item.get('text', '')}" for item in pred_data]
        pred_parent_ids = [item['parent_id'] for item in pred_data]
        pred_relations = [item['relation'] for item in pred_data]

        gt_tree = generate_doc_tree(gt_texts, gt_parent_ids, gt_relations)
        pred_tree = generate_doc_tree(pred_texts, pred_parent_ids, pred_relations)

        # 阅读顺序
        gt_main, gt_float = transfer_tree_to_chain(gt_tree)
        pred_main, pred_float = transfer_tree_to_chain(pred_tree)

        dist_main, teds_main = sequence_edit_distance(pred_main, gt_main)
        gt_float_groups = split_chain_by_tag(gt_float[1:])
        pred_float_groups = split_chain_by_tag(pred_float[1:])
        dist_float, teds_float = min_edit_distance_between_groups(gt_float_groups, pred_float_groups)

        result['reading_order'] = {
            'teds': teds_main,
            'distance': dist_main,
            'gt_nodes': len(gt_main),
            'pred_nodes': len(pred_main),
            'teds_floating': teds_float,
            'distance_floating': dist_float,
            'gt_floating_nodes': sum(len(g) for g in gt_float_groups),
            'pred_floating_nodes': sum(len(g) for g in pred_float_groups),
        }

        # 结构
        dist_struct, teds_struct = tree_edit_distance(pred_tree, gt_tree)
        result['structure'] = {
            'teds': teds_struct,
            'distance': dist_struct,
            'gt_nodes': len(gt_tree),
            'pred_nodes': len(pred_tree),
        }

        return result

    except Exception as e:
        logger.error(f"评估 {sample_id} 失败: {e}")
        return None


def evaluate_doc(
    gt_folder: str,
    pred_folder: str,
    output_dir: str = "./eval_output",
    num_workers: int = 4,
    parallel: bool = True,
) -> EvaluationReport:
    """便捷评估函数

    Args:
        gt_folder: 真实标签文件夹
        pred_folder: 预测结果文件夹
        output_dir: 输出目录
        num_workers: 并行工作进程数
        parallel: 是否并行评估

    Returns:
        评估报告
    """
    config = EvaluatorConfig(
        gt_folder=gt_folder,
        pred_folder=pred_folder,
        output_dir=output_dir,
        num_workers=num_workers,
    )
    evaluator = DOCEvaluator(config)

    if parallel and num_workers > 1:
        return evaluator.evaluate_from_folders_parallel(
            gt_folder, pred_folder, output_dir, num_workers
        )
    else:
        return evaluator.evaluate_from_folders(
            gt_folder, pred_folder, output_dir
        )
