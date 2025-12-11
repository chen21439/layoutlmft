# Evaluation scripts for HRDoc pipeline
#
# Usage:
#   Stage 1 Evaluation (semantic classification):
#     python run_classify_eval.py --gt_folder /path/to/gt --pred_folder /path/to/pred
#
#   End-to-End Evaluation (TEDS):
#     python run_teds_eval.py --gt_folder /path/to/gt --pred_folder /path/to/pred
#
#   Inference:
#     python run_inference.py --env test --dataset hrds --stage 1
#
# Note: Evaluation scripts wrap HRDoc's original implementation for consistency.
