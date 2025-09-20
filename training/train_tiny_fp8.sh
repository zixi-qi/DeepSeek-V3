#!/bin/bash

# Training script for tiny DeepSeek V3 model with FP8 quantization for maximum memory efficiency

echo "==========================================="
echo "Training Tiny DeepSeek V3 Model with FP8"
echo "==========================================="

cd /home/ubuntu/qizixi/DeepSeek-V3/training

# Run with tiny config and FP8 enabled
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --output_dir ./outputs_tiny_fp8 \
    --use_fp8 \
    --fp8_weight_quant \
    --fp8_activation_quant \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --bf16 \
    --gradient_checkpointing \
    --block_size 512 \
    --dataloader_num_workers 4

echo "FP8 training complete!"
