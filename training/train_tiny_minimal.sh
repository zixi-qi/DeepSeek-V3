#!/bin/bash

# Training script for tiny DeepSeek V3 model with minimal memory usage

echo "==========================================="
echo "Training Tiny DeepSeek V3 Model (Minimal Memory)"
echo "==========================================="

cd /home/ubuntu/qizixi/DeepSeek-V3/training

# Run with minimal memory settings
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --output_dir ./outputs_tiny_minimal \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 100 \
    --bf16 \
    --gradient_checkpointing \
    --block_size 128 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory False

echo "Minimal memory training complete!"
