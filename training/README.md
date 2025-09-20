# DeepSeek V3 Mini Model Training on Tiny Shakespeare

This directory contains the training code for the mini DeepSeek V3 model using the tiny Shakespeare dataset from Hugging Face.

## Overview

The training setup uses:
- **Model**: Mini DeepSeek V3 with MoE (Mixture of Experts) architecture
- **Dataset**: [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
- **Framework**: PyTorch with Hugging Face Accelerate for distributed training
- **Precision**: BF16 mixed precision training with optional FP8 quantization
- **Hardware**: Optimized for NVIDIA H100/A100 GPUs (80GB recommended)

## System Requirements

- **GPU**: NVIDIA GPU with at least 16GB VRAM (80GB recommended for full model)
- **CUDA**: 11.8+ (12.1+ recommended)
- **Python**: 3.8-3.11 (3.10 recommended)
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for checkpoints and logs

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/training
```

### 2. Create Python Environment

```bash
# Using conda (recommended)
conda create -n deepseek python=3.10
conda activate deepseek

# Or using venv
python3.10 -m venv deepseek_env
source deepseek_env/bin/activate
```

### 3. Install PyTorch (adjust for your CUDA version)

```bash
# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Dependencies

```bash
cd /path/to/DeepSeek-V3/training
pip install -r requirements.txt

# Fix potential numpy compatibility issues
pip install "numpy<2.0"
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

# Test the setup
python test_setup.py
```

### 6. Download Model Files

The model files should be in the `../mini_model/` directory:
- `config.json` or `config_tiny.json` - Model configuration
- `modeling_deepseek.py` - Model implementation
- `configuration_deepseek.py` - Configuration classes
- `tokenizer_config.json` - Tokenizer configuration
- `tokenizer.json` - Tokenizer vocabulary

## Training

### Quick Start (Single GPU)

```bash
python train.py
```

### Using Accelerate (Recommended)

1. Configure accelerate (first time only):
```bash
accelerate config
# Or use the provided config:
cp accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
```

2. Launch training:
```bash
accelerate launch train.py
```

### Custom Configuration

You can override default training arguments:

```bash
python train.py \
    --output_dir ./my_output \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-4 \
    --logging_steps 50
```

### Multi-GPU Training

For multi-GPU training on a single node:

```bash
accelerate launch --multi_gpu --num_processes 4 train.py
```

## Configuration Files

- `training_config.yaml`: Main training configuration (not directly used by the script, but documents recommended settings)
- `accelerate_config.yaml`: Accelerate configuration for distributed training
- `requirements.txt`: Python dependencies

## Training Arguments

Key training parameters (with defaults):

- `num_train_epochs`: 3 (number of training epochs)
- `per_device_train_batch_size`: 4 (batch size per GPU)
- `gradient_accumulation_steps`: 4 (gradient accumulation steps)
- `learning_rate`: 5e-4 (initial learning rate)
- `block_size`: 512 (maximum sequence length)
- `bf16`: true (use bfloat16 precision)
- `gradient_checkpointing`: true (save memory during training)

### FP8 Training

The training script supports FP8 quantization for reduced memory usage:

- `--use_fp8`: Enable FP8 quantization
- `--fp8_weight_quant`: Quantize weights to FP8 (default: true)
- `--fp8_activation_quant`: Quantize activations to FP8 (default: true)

Example:
```bash
python train.py --use_fp8 --fp8_weight_quant --fp8_activation_quant
```

**Important**: The FP8 implementation has been updated to fix memory issues. The new version:
- Uses gradient scaling instead of actual FP8 tensor conversion during training
- Avoids creating additional memory copies that caused OOM errors
- Provides better memory efficiency without the overhead of FP8 tensor operations

If you still encounter memory issues, use the minimal memory training script:
```bash
./train_tiny_minimal.sh
```

## Model Architecture

### Original Mini Model (5.69B parameters)
- Hidden size: 2048
- Number of layers: 24
- Number of attention heads: 32
- Vocabulary size: 129,280
- MoE (Mixture of Experts) with 64 experts, 4 active per token
- Memory requirement: ~64GB for training

### Tiny Model (0.14B parameters) - For Single GPU Training
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Vocabulary size: 32,000
- MoE (Mixture of Experts) with 8 experts, 2 active per token
- Memory requirement: ~2GB for training
- Config file: `mini_model/config_tiny.json`

Use the tiny model for development and testing on a single GPU:
```bash
# Standard training
./train_tiny.sh

# With FP8 quantization for even better memory efficiency
./train_tiny_fp8.sh
```

## Dataset

The tiny Shakespeare dataset contains:
- ~1.1M characters of Shakespeare text
- Split into train/validation/test (90/5/5)
- Ideal for testing and debugging language models

## Output Structure

Training outputs will be saved in the `output_dir` (default: `./outputs`):

```
outputs/
├── checkpoint-500/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── config.json
├── checkpoint-1000/
│   └── ...
├── final_model/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer files
└── runs/
    └── tensorboard logs
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/runs
```

### Weights & Biases (Optional)

Set up W&B tracking:

```bash
wandb login
python train.py --report_to wandb --run_name "deepseek_v3_mini_shakespeare"
```

## Inference

After training, you can use the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = "./outputs/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Generate text
prompt = "To be or not to be"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    temperature=0.8,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Common Issues and Solutions

### Installation Issues

1. **CUDA/PyTorch Mismatch**:
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch for correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

2. **NumPy Compatibility Error**:
```bash
# Downgrade numpy
pip install "numpy<2.0"
```

3. **Transformers/Accelerate Issues**:
```bash
# Install specific versions
pip install transformers==4.38.0 accelerate==0.27.0
```

### Training Issues

#### Out of Memory (OOM)

**Note**: The original FP8 implementation had memory issues. Use the updated version below.

1. **Use the Tiny Model**:
```bash
python train.py --model_config_path ../mini_model/config_tiny.json
```

2. **Minimal Memory Training**:
```bash
./train_tiny_minimal.sh  # Batch size 1, gradient accumulation 16
```

3. **Reduce Batch Size**:
```bash
python train.py --per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

4. **Enable Gradient Checkpointing**:
```bash
python train.py --gradient_checkpointing
```

5. **Use Updated FP8 Quantization**:
```bash
./train_tiny_fp8.sh  # Uses improved FP8 implementation
```

6. **CPU Offloading**:
```bash
accelerate launch --mixed_precision bf16 --cpu_offload train.py
```

7. **Reduce Sequence Length**:
```bash
python train.py --block_size 128  # Instead of 512
```

8. **Disable Pin Memory**:
```bash
python train.py --dataloader_pin_memory False
```

#### Model Not Training (AssertionError)

The original model has training restrictions in the MoE layer. This has been fixed in our version, but if you encounter it:

1. Check that `modeling_deepseek.py` has the training mode fix
2. Look for `assert not self.training` and ensure it's commented out

#### Slow Training

1. **Enable Mixed Precision**:
```bash
python train.py --bf16
```

2. **Optimize Data Loading**:
```bash
python train.py --dataloader_num_workers 8
```

3. **Use torch.compile (PyTorch 2.0+)**:
```python
# Add to train.py after model initialization
model = torch.compile(model)
```

### Hardware-Specific Settings

#### NVIDIA H100 (80GB)
```bash
# Can run full mini model
python train.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --bf16
```

#### NVIDIA A100 (40GB)
```bash
# Use tiny model with moderate batch size
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16
```

#### NVIDIA RTX 4090 (24GB)
```bash
# Use tiny model with small batch size
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --use_fp8 \
    --bf16
```

## Advanced Usage

### Resume from Checkpoint

```bash
python train.py --resume_from_checkpoint outputs/checkpoint-1000
```

### Custom Dataset

Modify `data_utils.py` to add support for other datasets:

```python
from data_utils import create_shakespeare_dataset

# Replace with your dataset loading logic
dataset = create_custom_dataset(tokenizer, block_size=512)
```

## Performance Tips

1. **Use BF16**: Faster than FP32, more stable than FP16
2. **Gradient Checkpointing**: Trades compute for memory
3. **Efficient Attention**: The model uses Flash Attention when available
4. **Data Preprocessing**: Pre-tokenize and cache the dataset for faster loading

## Citation

If you use this code, please cite:

```bibtex
@misc{deepseek-v3-2024,
  title={DeepSeek V3 Technical Report},
  author={DeepSeek Team},
  year={2024}
}

@misc{karpathy2015charrnn,
  author={Karpathy, Andrej},
  title={char-rnn},
  year={2015},
  howpublished={\url{https://github.com/karpathy/char-rnn}}
}
```

## Project Structure

```
DeepSeek-V3/
├── mini_model/
│   ├── config.json              # Original mini model config (5.69B params)
│   ├── config_tiny.json         # Tiny model config (0.14B params)
│   ├── modeling_deepseek.py     # Model implementation (with training fixes)
│   ├── configuration_deepseek.py # Configuration classes
│   ├── tokenizer_config.json    # Tokenizer configuration
│   └── tokenizer.json           # Tokenizer vocabulary
├── training/
│   ├── train.py                 # Main training script
│   ├── data_utils.py           # Dataset utilities
│   ├── fp8_utils.py            # FP8 quantization utilities
│   ├── test_setup.py           # Setup verification script
│   ├── inference.py            # Inference script for trained models
│   ├── estimate_model_size.py  # Model size estimation tool
│   ├── requirements.txt        # Python dependencies
│   ├── accelerate_config.yaml  # Accelerate configuration
│   ├── training_config.yaml    # Training hyperparameters
│   ├── train_tiny.sh           # Script for tiny model training
│   ├── train_tiny_fp8.sh       # Script for FP8 tiny model training
│   └── outputs/                # Training outputs (created during training)
└── inference/
    └── (inference-specific code)
```

## Key Modifications for Training

1. **MoE Training Support**: Added `_moe_forward_train` method to handle MoE layers during training
2. **FP8 Quantization**: Implemented FP8Linear layer for memory-efficient training
3. **Tiny Model Config**: Created smaller model variant that fits on consumer GPUs
4. **Training Fixes**: Removed training-mode assertions in the original model

## Contributing

When contributing to this training setup:

1. Test changes on both tiny and full model configurations
2. Ensure compatibility with different PyTorch/CUDA versions
3. Update documentation for any new features
4. Add appropriate error handling and logging

## License

This training code follows the same license as the DeepSeek V3 model.
