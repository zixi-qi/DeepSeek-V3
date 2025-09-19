# DeepSeek Mini 1B Model Summary

## Overview
Successfully initialized and tested the DeepSeek Mini 1B model with the following key features:

### Architecture Details
- **Model Dimension**: 2,048
- **Vocabulary Size**: 129,280
- **Total Layers**: 24 (2 dense + 22 MoE)
- **Attention Heads**: 32
- **Max Sequence Length**: 16,384

### Quantization
- **FP8 Quantization**: Enabled for weights (float8_e4m3fn)
- **Scale Parameters**: FP32 for quantization scales
- **Activation**: BF16

### Mixture of Experts (MoE)
- **Total Experts**: 64 per layer
- **Activated Experts**: 4 per token (sparse activation)
- **Shared Experts**: 1
- **Expert Groups**: 8
- **Scoring Function**: Sigmoid with route scale 2.5

### Multi-Head Latent Attention (MLA)
- **Q LoRA Rank**: 768
- **KV LoRA Rank**: 256
- **QK Rope/Nope Head Dim**: 64/64
- **V Head Dim**: 64

### Parameter Count
- **All Experts Instantiated**: ~5.36B parameters
- **Effective Model Size**: ~1.20B parameters (with sparse activation)

## Running the Model

### Interactive Mode
```bash
cd /home/ubuntu/DeepSeek-V3/inference
python generate.py --ckpt-path /home/ubuntu/DeepSeek-V3 \
    --config configs/config_1B.json \
    --interactive \
    --max-new-tokens 100 \
    --temperature 0.7
```

### Batch Mode
```bash
cd /home/ubuntu/DeepSeek-V3/inference
python generate.py --ckpt-path /home/ubuntu/DeepSeek-V3 \
    --config configs/config_1B.json \
    --input-file test_prompts.txt \
    --max-new-tokens 100 \
    --temperature 0.7
```

## Notes
- Currently using randomly initialized weights for demonstration
- To use pretrained weights, download them and convert using `convert.py`
- The model uses sparse MoE activation, making it computationally efficient despite having 5.36B total parameters
- FP8 quantization reduces memory usage while maintaining model quality
