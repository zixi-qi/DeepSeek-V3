#!/usr/bin/env python3
"""Save the initialized DeepSeek Mini 1B model."""

import os
import json
import torch
from safetensors.torch import save_file
from model import Transformer, ModelArgs
import shutil

def save_deepseek_mini_model(save_path: str, config_path: str):
    """Initialize and save the DeepSeek Mini 1B model."""
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    print(f"Creating model directory at: {save_path}")
    
    # Load configuration
    with open(config_path) as f:
        config_dict = json.load(f)
        args = ModelArgs(**config_dict)
    
    print("Model configuration:")
    print(args)
    
    # Initialize model
    print("\nInitializing DeepSeek Mini 1B model...")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    
    with torch.device("cuda"):
        model = Transformer(args)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    
    # Prepare state dict for saving
    # We'll save it in a format compatible with the inference pipeline
    print("\nPreparing model for saving...")
    state_dict = {}
    
    for name, param in model.named_parameters():
        # Convert to CPU for saving
        param_cpu = param.cpu()
        
        # Handle FP8 weights specially - save both weight and scale
        if param.dtype == torch.float8_e4m3fn:
            state_dict[name] = param_cpu
            # The scale should be saved with the same base name
            if hasattr(param, 'scale'):
                scale_name = name.replace('.weight', '.scale')
                if scale_name in dict(model.named_parameters()):
                    scale_param = dict(model.named_parameters())[scale_name]
                    state_dict[scale_name] = scale_param.cpu()
        else:
            state_dict[name] = param_cpu
    
    # Save model weights
    model_file = os.path.join(save_path, "model0-mp1.safetensors")
    print(f"\nSaving model weights to: {model_file}")
    save_file(state_dict, model_file)
    
    # Save configuration
    config_file = os.path.join(save_path, "config.json")
    print(f"Saving configuration to: {config_file}")
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Copy tokenizer files if they exist
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file in tokenizer_files:
        src = os.path.join(os.path.dirname(os.path.dirname(config_path)), file)
        if os.path.exists(src):
            dst = os.path.join(save_path, file)
            print(f"Copying {file} to model directory")
            shutil.copy2(src, dst)
    
    # Create model index file for compatibility
    index_dict = {
        "metadata": {
            "total_size": sum(p.numel() * p.element_size() for p in model.parameters())
        },
        "weight_map": {name: "model0-mp1.safetensors" for name in state_dict.keys()}
    }
    
    index_file = os.path.join(save_path, "model.safetensors.index.json")
    print(f"Saving model index to: {index_file}")
    with open(index_file, 'w') as f:
        json.dump(index_dict, f, indent=2)
    
    print("\n✅ Model saved successfully!")
    print(f"Total files saved: {len(os.listdir(save_path))}")
    
    return model, args

def verify_saved_model(save_path: str):
    """Verify the saved model can be loaded."""
    print("\n" + "="*60)
    print("Verifying saved model...")
    
    # Check files exist
    required_files = ["model0-mp1.safetensors", "config.json"]
    for file in required_files:
        file_path = os.path.join(save_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {file}: {size:.2f} MB")
        else:
            print(f"✗ {file}: NOT FOUND")
    
    # Try loading config
    config_path = os.path.join(save_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
        print(f"\n✓ Configuration loaded successfully")
        print(f"  - Model dimension: {config['dim']}")
        print(f"  - Vocabulary size: {config['vocab_size']:,}")
        print(f"  - Quantization: {config['dtype'].upper()}")

def main():
    # Paths
    config_path = "configs/config_1B.json"
    save_path = "/home/ubuntu/DeepSeek-V3/model"
    
    # Save model
    model, args = save_deepseek_mini_model(save_path, config_path)
    
    # Verify
    verify_saved_model(save_path)

if __name__ == "__main__":
    main()
