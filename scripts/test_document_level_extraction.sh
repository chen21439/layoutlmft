#!/bin/bash
# 测试文档级别特征提取

source /root/miniforge3/etc/profile.d/conda.sh
conda activate layoutlmv2

export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc_test"
export LAYOUTLMFT_NUM_SAMPLES="50"
export LAYOUTLMFT_DOCS_PER_CHUNK="10"

cd /root/code/layoutlmft
python examples/extract_line_features_document_level.py
