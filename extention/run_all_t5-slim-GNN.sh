#!/bin/bash
task_id=3
epoch=1
len=10

# Check if graph embeddings exist — optional
GRAPH_EMB=../graph_emb/task_${task_id}_graph.npy
if [ ! -f "$GRAPH_EMB" ]; then
  echo "⚠ Graph embeddings not found for task_id=${task_id}. Model will skip GNN."
fi

# Execute updated main script in extention/
deepspeed --master_port=29501 main_profile-slim-GNN.py \
    --model_path ../FlanT5-small/ \
    --emb_model_path ../bge-base-en-v1.5/ \
    --train_file ../LaMP_time_${task_id}_id/train_aug_input.json \
    --dev_file ../LaMP_time_${task_id}_id/dev_profile.json \
    --use_profile True \
    --use_session False \
    --use_graph False \
    --use_inst_token False \
    --use_4bit True \
    --use_8bit False \
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
    --gradient_accumulation_steps 8 \
    --logging_dir ./log/ \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --save_only_model True \
    --eval_steps 0.1 \
    --log_level warning \
    --deepspeed dp.json \
    --report_to none \
    --save_total_limit 1 \
    --bf16 False \
> output.log 2>&1