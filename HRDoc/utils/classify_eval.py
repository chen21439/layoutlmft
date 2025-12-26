#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: JeffreyMa
# -----
# Copyright (c) 2023 iFLYTEK & USTC
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import os
import json
import tqdm
import logging
import argparse
from sklearn.metrics import f1_score
from utils import trans_class

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class2id_dict = {
    "title": 0,
    "author": 1,
    "mail": 2,
    "affili": 3,
    "section": 4,
    "fstline": 5,
    "paraline": 6,
    "table": 7,
    "figure": 8,
    "caption": 9,
    "equation": 10,
    "footer": 11,
    "header": 12,
    "footnote": 13,
}

def class2id(jdata, unit):
    class_ = unit['class']
    if class_ not in class2id_dict.keys():
        class_ = trans_class(jdata, unit)
    assert class_ in class2id_dict.keys(), "{} not in {} classes!".format(
        class_, len(class2id_dict.keys()))
    return class2id_dict[class_]

def assert_filetree(args):
    """ Make sure the json files of `gt_folder` and `pred_folder` are the same """

    gt_folders = set(os.listdir(args.gt_folder))
    pred_folders = set(os.listdir(args.pred_folder))
    assert gt_folders == pred_folders, "{} and {} contains different PDF files!".format(
        args.gt_folder, args.pred_folder)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gt_folder", default="/Users/majiefeng/Desktop/讯飞实习/HRDoc工作相关/HRDoc_dataset/HRDH/test_1", type=str, required=True,
                        help="The folder storing ground-truth files.")
    parser.add_argument("--pred_folder", default="/Users/majiefeng/Desktop/讯飞实习/HRDoc工作相关/HRDoc_dataset/HRDH/test_2", type=str, required=True,
                        help="The folder storing predicted results.")
    
    args = parser.parse_args()
    logging.info("Args received, gt_folder: {}, pred_folder: {}".format(args.gt_folder, args.pred_folder))

    assert_filetree(args=args)
    logging.info("File tree matched, start parse through json files!")

    gt_class = []
    pred_class = []

    for pdf_file in tqdm.tqdm(os.listdir(args.gt_folder)):
        gt_file = os.path.join(args.gt_folder, pdf_file)
        pred_file = os.path.join(args.pred_folder, pdf_file)
        gt_json = json.load(open(gt_file))
        pred_json = json.load(open(pred_file))
        assert len(gt_json) == len(pred_json), "{} and {} contains different numbers of units".format(
            gt_file, pred_file)
        gt_class.extend([class2id(gt_json, x) for x in gt_json])
        pred_class.extend([class2id(pred_json, x) for x in pred_json])
    logging.info("Parse finished, got {} units in total. Start calculate f1!".format(len(gt_class)))

    detailed_f1 = f1_score(gt_class, pred_class, average=None)
    macro_f1 = f1_score(gt_class, pred_class, average='macro')
    micro_f1 = f1_score(gt_class, pred_class, average='micro')

    # 计算 Accuracy（对于单标签分类，micro_f1 = accuracy）
    accuracy = sum(1 for g, p in zip(gt_class, pred_class) if g == p) / len(gt_class)

    # Build id2class mapping
    id2class_dict = {v: k for k, v in class2id_dict.items()}

    # Get unique classes in the evaluation set
    unique_classes = sorted(set(gt_class) | set(pred_class))

    # Build per-class results with class names
    class_results = []
    for i, class_id in enumerate(unique_classes):
        if i < len(detailed_f1):
            class_name = id2class_dict.get(class_id, f"unknown_{class_id}")
            class_results.append((class_name, class_id, detailed_f1[i]))

    # Sort by F1 score descending
    class_results_sorted = sorted(class_results, key=lambda x: x[2], reverse=True)

    # Print per-class F1 scores
    logging.info("=" * 50)
    logging.info("Per-class F1 scores (sorted by F1 descending):")
    logging.info("-" * 50)
    logging.info(f"{'Rank':<6}{'Class':<12}{'ID':<6}{'F1 Score':<10}")
    logging.info("-" * 50)
    for rank, (class_name, class_id, f1) in enumerate(class_results_sorted, 1):
        logging.info(f"{rank:<6}{class_name:<12}{class_id:<6}{f1:.4f}")
    logging.info("-" * 50)
    logging.info(f"{'Accuracy:':<24}{accuracy:.4f}")
    logging.info(f"{'Macro F1:':<24}{macro_f1:.4f}")
    logging.info(f"{'Micro F1:':<24}{micro_f1:.4f}")
    logging.info("=" * 50)

    # Also print original format for backward compatibility
    logging.info("accuracy : {}, macro_f1 : {}, micro_f1 : {}".format(accuracy, macro_f1, micro_f1))

if __name__ == "__main__":
    main()
