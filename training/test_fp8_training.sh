#!/bin/bash

# Script to test FP8 training with the mini DeepSeek V3 model

echo "==========================================="
echo "Testing FP8 Training on DeepSeek V3 Mini"
echo "==========================================="

cd /home/ubuntu/qizixi/DeepSeek-V3/training

# Run with FP8 enabled
echo "Running training with FP8 quantization enabled..."
python train.py \
    --use_fp8 \
    --fp8_weight_quant \
    --fp8_activation_quant \
    --output_dir ./outputs_fp8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 5 \
    --eval_steps 50 \
    --save_steps 100 \
    --bf16 \
    --gradient_checkpointing

echo "FP8 training test complete!"
