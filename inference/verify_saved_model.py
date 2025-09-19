#!/usr/bin/env python3
"""Verify the saved DeepSeek Mini 1B model can be loaded and used."""

import os
import json
import torch
from safetensors.torch import load_file
from model import Transformer, ModelArgs
from generate import generate
from transformers import AutoTokenizer

def load_saved_model(model_path: str):
    """Load the saved model from disk."""
    print(f"Loading model from: {model_path}")
    
    # Load configuration
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
        args = ModelArgs(**config_dict)
    
    print("Loaded configuration:")
    print(f"  - Model dimension: {args.dim}")
    print(f"  - Vocabulary size: {args.vocab_size:,}")
    print(f"  - Layers: {args.n_layers}")
    print(f"  - Quantization: {args.dtype}")
    
    # Initialize model architecture
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    
    with torch.device("cuda"):
        model = Transformer(args)
    
    # Load weights
    weights_path = os.path.join(model_path, "model0-mp1.safetensors")
    print(f"\nLoading weights from: {weights_path}")
    
    state_dict = load_file(weights_path)
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=True)
    print("✅ Model weights loaded successfully!")
    
    return model, args

def test_inference(model, args, model_path):
    """Test inference with the loaded model."""
    print("\n" + "="*60)
    print("Testing inference with loaded model...")
    
    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        # Fallback to parent directory
        try:
            tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/DeepSeek-V3")
            print("Using tokenizer from parent directory")
        except:
            print("Warning: Could not load tokenizer")
            return
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "The capital of France is"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        print(f"Tokens: {len(tokens)}")
        
        # Generate
        with torch.no_grad():
            completion_tokens = generate(
                model, 
                [tokens], 
                max_new_tokens=30,
                eos_id=tokenizer.eos_token_id,
                temperature=0.7
            )
        
        # Decode
        completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        print(f"Completion: {completion}")

def check_model_stats(model):
    """Display model statistics."""
    print("\n" + "="*60)
    print("Model Statistics:")
    
    # Count parameters by type
    fp8_params = 0
    bf16_params = 0
    fp32_params = 0
    
    for name, param in model.named_parameters():
        if param.dtype == torch.float8_e4m3fn:
            fp8_params += param.numel()
        elif param.dtype == torch.bfloat16:
            bf16_params += param.numel()
        elif param.dtype == torch.float32:
            fp32_params += param.numel()
    
    total_params = fp8_params + bf16_params + fp32_params
    
    print(f"Total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    print(f"  - FP8 parameters: {fp8_params:,} ({fp8_params/total_params*100:.1f}%)")
    print(f"  - BF16 parameters: {bf16_params:,} ({bf16_params/total_params*100:.1f}%)")
    print(f"  - FP32 parameters: {fp32_params:,} ({fp32_params/total_params*100:.1f}%)")
    
    # Memory usage estimate
    memory_bytes = (fp8_params * 1) + (bf16_params * 2) + (fp32_params * 4)
    memory_gb = memory_bytes / (1024**3)
    print(f"\nEstimated memory usage: {memory_gb:.2f} GB")

def main():
    model_path = "/home/ubuntu/DeepSeek-V3/model"
    
    # Load model
    model, args = load_saved_model(model_path)
    
    # Check statistics
    check_model_stats(model)
    
    # Test inference
    test_inference(model, args, model_path)
    
    print("\n✅ Model verification completed successfully!")

if __name__ == "__main__":
    main()
