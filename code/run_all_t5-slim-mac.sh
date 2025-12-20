
#!/bin/bash

# Run for task_id = 3 only
task_id=3

# Set parameters for task_id=3
epoch=1
len=10

# Execute deepspeed command
python main_profile-dev.py \
    --model_path ../FlanT5-small/ \
    --emb_model_path ../bge-base-en-v1.5/ \
    --train_file ../LaMP_time_${task_id}_id/train_aug_input.json \
    --dev_file ../LaMP_time_${task_id}_id/dev_profile.json \
    --max_input_len 256 \
    --max_his_len 512 \
    --max_new_len ${len} \
    --output_dir output_${task_id} \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_dir ./log/ \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --save_strategy epoch \
    --save_only_model True \
    --eval_steps 0.1 \
    --log_level warning \
    --report_to none \
    --save_total_limit 1 \
    --use_cpu True\
> output_mac.log 2>&1
