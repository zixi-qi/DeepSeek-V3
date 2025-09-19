# DeepSeek Mini 1B Model (vLLM Compatible)

This directory contains the DeepSeek Mini 1B model configured for use with vLLM and HuggingFace transformers.

## Model Information
- **Architecture**: DeepSeek V3 Mini
- **Parameters**: ~5.36B total (1.2B effective with sparse activation)
- **Quantization**: FP8 (float8_e4m3fn)
- **Model Dimension**: 2,048
- **Vocabulary Size**: 129,280
- **Max Sequence Length**: 16,384 tokens
- **HuggingFace Compatible**: ✅
- **vLLM Compatible**: ✅

## Files
- `model.safetensors`: Model weights in HuggingFace format (5.5 GB)
- `config.json`: HuggingFace-compatible configuration
- `model.safetensors.index.json`: Weight mapping index
- `tokenizer.json`: Tokenizer vocabulary
- `tokenizer_config.json`: Tokenizer configuration
- `modeling_deepseek.py`: Model implementation
- `configuration_deepseek.py`: Configuration classes

## Architecture Features
- **Multi-Head Latent Attention (MLA)** with LoRA compression
- **Mixture of Experts (MoE)**: 64 experts, 4 activated per token
- **FP8 Quantization**: 90.1% of parameters in FP8
- **YARN Positional Encoding**: Extended context support

## Usage with vLLM

### Installation
```bash
pip install vllm
```

### Basic Usage
```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="/home/ubuntu/DeepSeek-V3/model",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.95,
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# Generate text
prompts = ["Hello, how are you?", "What is machine learning?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
```

## Usage with HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/DeepSeek-V3/model",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/DeepSeek-V3/model")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Memory Requirements
- **Model Weights**: ~5.5 GB
- **vLLM Runtime**: ~8-10 GB (depending on batch size and sequence length)
- **Recommended GPU**: RTX 3090/4090 or A100 with at least 24GB VRAM

## Notes
- This model uses randomly initialized weights (not pre-trained)
- For production use, replace with properly trained weights
- The model uses sparse MoE activation for efficiency
- FP8 quantization may require specific GPU support (e.g., H100)
- For older GPUs, consider using bfloat16 or float16 instead

## Troubleshooting

If you encounter issues with vLLM:
1. Ensure you have sufficient GPU memory
2. Try reducing `max_model_len` parameter
3. Set `enforce_eager=True` if Flash Attention causes issues
4. Check that your GPU supports the specified dtype