#!/bin/bash

# Training script for tiny DeepSeek V3 model that fits on a single H100

echo "==========================================="
echo "Training Tiny DeepSeek V3 Model"
echo "==========================================="

cd /home/ubuntu/qizixi/DeepSeek-V3/training

# Run with tiny config
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --output_dir ./outputs_tiny \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --bf16 \
    --gradient_checkpointing \
    --block_size 512 \
    --dataloader_num_workers 4

echo "Training complete!"
