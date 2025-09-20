# DeepSeek V3 Training Quick Start Guide

## 🚀 Quick Setup (5 minutes)

```bash
# 1. Clone and navigate
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd DeepSeek-V3/training

# 2. Run automatic setup
./setup.sh

# 3. Activate environment
source deepseek_env/bin/activate

# 4. Start training (choose based on your GPU)
./train_tiny.sh        # For GPUs with 16GB+ VRAM
```

## 📊 Model Configurations

| Model | Parameters | Min GPU VRAM | Recommended | Config File |
|-------|------------|--------------|-------------|-------------|
| Tiny | 0.14B | 16GB | 24GB | `config_tiny.json` |
| Mini | 5.69B | 64GB | 80GB | `config.json` |

## 🎯 Common Training Commands

### Tiny Model (Most GPUs)
```bash
# Basic training
python train.py --model_config_path ../mini_model/config_tiny.json

# With FP8 quantization (saves memory)
python train.py --model_config_path ../mini_model/config_tiny.json --use_fp8

# Small batch for limited memory
python train.py --model_config_path ../mini_model/config_tiny.json \
    --per_device_train_batch_size 2 --gradient_checkpointing
```

### Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View training logs
tensorboard --logdir outputs/runs

# Check training progress
tail -f outputs/deepseek_v3_mini/*/events.out.tfevents.*
```

## 🔧 Quick Fixes

### Out of Memory?
```bash
# Use tiny model + small batch + gradient checkpointing + FP8
python train.py \
    --model_config_path ../mini_model/config_tiny.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --use_fp8
```

### Training Not Starting?
```bash
# Check GPU
nvidia-smi

# Verify setup
python test_setup.py

# Check model can forward pass
python -c "
import torch
from train import *
model_args = ModelArguments(model_config_path='../mini_model/config_tiny.json')
# Test continues...
"
```

### Wrong CUDA Version?
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## 📈 Training Tips

1. **Start Small**: Use tiny model first to verify everything works
2. **Monitor Memory**: Keep `nvidia-smi` open in another terminal
3. **Save Checkpoints**: Model saves every 500 steps by default
4. **Adjust Batch Size**: Larger = faster but more memory
5. **Use FP8**: Reduces memory usage with minimal quality loss

## 🎮 Interactive Testing

```python
# In Python/Jupyter
from inference import load_model_and_tokenizer, generate_text

# Load your trained model
model, tokenizer = load_model_and_tokenizer("./outputs_tiny/final_model")

# Generate text
prompt = "To be or not to be"
output = generate_text(model, tokenizer, prompt, max_length=100)
print(output[0])
```

## 📊 Expected Training Times

| Model | GPU | Batch Size | Time per Epoch |
|-------|-----|------------|----------------|
| Tiny | RTX 4090 | 4 | ~5 minutes |
| Tiny | A100 40GB | 16 | ~2 minutes |
| Mini | H100 80GB | 8 | ~15 minutes |

## 🆘 Get Help

- Check full README.md for detailed documentation
- Run `python train.py --help` for all options
- Look at error messages - they usually tell you what's wrong
- GPU memory issues? Always try smaller batch size first
