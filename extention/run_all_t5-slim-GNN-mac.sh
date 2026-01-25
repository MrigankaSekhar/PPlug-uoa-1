#!/bin/bash

task_id=3
epoch=1
len=10

# Optional: check if graph embeddings exist
GRAPH_EMB=../graph_emb/task_${task_id}_graph.npy
if [ ! -f "$GRAPH_EMB" ]; then
  echo "⚠ Graph embeddings not found for task_id=${task_id}. Model will skip GNN."
fi

# Create log directory if not present
mkdir -p ./log

# Generate timestamped log filename: output_mac_<taskid>_<YYYYMMDD>_<HHMMSS>.log
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./log/output_mac_${task_id}_${timestamp}.log"

# Run without deepspeed, on CPU, writing full output to log file
python main_profile-slim-GNN.py \
    --model_path ../FlanT5-small/ \
    --emb_model_path ../bge-base-en-v1.5/ \
    --train_file ../LaMP_time_${task_id}_subset_id/train_aug_input.json \
    --dev_file ../LaMP_time_${task_id}_subset_id/dev_profile.json \
    --max_input_len 256 \
    --use_subset True \
    --max_his_len 512 \
    --max_new_len ${len} \
    --output_dir output_${task_id} \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_dir ./log/ \
    --report_to none \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --save_only_model True \
    --eval_steps 0.1 \
    --log_level warning \
    --save_total_limit 1 \
    --use_cpu True \
    2>&1 | tee "${LOG_FILE}"

echo "✅ Training log saved to ${LOG_FILE}"