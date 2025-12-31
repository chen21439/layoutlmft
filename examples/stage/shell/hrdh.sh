mkdir -p logs

LOG=logs/train_joint_$(date +%Y%m%d_%H%M%S).log

  nohup python examples/stage/train_joint.py \
      --env test \
      --min_steps 1000  \
      --gradient_checkpointing  \
      --per_device_train_batch_size 1 \
      --dataset hrdh \
      --covmatch doc_covmatch_dev12_seed42 \
      --eval_before_train \
      --stage1_no_grad false  \
      --model_name_or_path /data/LLM_group/layoutlmft/artifact/exp_hrds/joint_hrds/checkpoint-100  \
      --report_to tensorboard \
      --save_predictions  \
      --new_exp test_speed_hrdh  \
        > "$LOG" 2>&1 &

echo "Started. Log: $LOG"
tail -f "$LOG"

#	    --force_rebuild	\