#!/bin/bash

# Script to run DeepSeek V3 mini model training on tiny Shakespeare dataset

echo "==========================================="
echo "DeepSeek V3 Mini Training on Tiny Shakespeare"
echo "==========================================="

# Check if running in the correct directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the training directory"
    echo "cd /home/ubuntu/qizixi/DeepSeek-V3/training"
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "No GPU detected, training will use CPU (will be slow)"
    echo ""
fi

# Parse command line arguments
MODE=${1:-"simple"}

case $MODE in
    "simple")
        echo "Running simple training (single process)..."
        python train.py \
            --output_dir ./outputs \
            --num_train_epochs 5 \
            --logging_steps 10
        ;;
        
    "accelerate")
        echo "Running with accelerate (recommended for multi-GPU)..."
        accelerate launch train.py \
            --output_dir ./outputs_accelerate \
            --num_train_epochs 5 \
            --logging_steps 10
        ;;
        
    "test")
        echo "Running test setup..."
        python test_setup.py
        ;;
        
    "quick")
        echo "Running quick training test (1 epoch, small batch)..."
        python train.py \
            --output_dir ./outputs_test \
            --num_train_epochs 1 \
            --per_device_train_batch_size 2 \
            --logging_steps 5 \
            --save_steps 50 \
            --eval_steps 25
        ;;
        
    *)
        echo "Usage: $0 [simple|accelerate|test|quick]"
        echo "  simple    - Run standard single-GPU training"
        echo "  accelerate - Run with accelerate for multi-GPU"
        echo "  test      - Test the setup without training"  
        echo "  quick     - Quick training test (1 epoch)"
        exit 1
        ;;
esac

echo ""
echo "Training complete! Check the output directory for results."
