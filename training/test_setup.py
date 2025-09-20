#!/usr/bin/env python3
"""
Quick test script to verify the training setup is working correctly.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

# Add the parent directory to Python path to allow package imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

from mini_model.configuration_deepseek import DeepseekV3Config
from mini_model.modeling_deepseek import DeepseekV3ForCausalLM
from data_utils import create_shakespeare_dataset, calculate_dataset_statistics


def test_model_initialization():
    """Test model initialization"""
    print("Testing model initialization...")
    
    # Load config
    config_path = "../mini_model/config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = DeepseekV3Config(**config_dict)
    
    # Initialize model
    model = DeepseekV3ForCausalLM(config)
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**3:.2f} GB (assuming float32)")
    
    return model


def test_tokenizer():
    """Test tokenizer loading"""
    print("\nTesting tokenizer...")
    
    tokenizer_path = "../mini_model"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Test encoding/decoding
    test_text = "To be or not to be, that is the question."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Test text: {test_text}")
    print(f"  Token count: {len(tokens)}")
    print(f"  Decoded: {decoded}")
    
    return tokenizer


def test_dataset(tokenizer):
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        # Create dataset
        dataset = create_shakespeare_dataset(
            tokenizer=tokenizer,
            split="train",
            block_size=128,  # Small block size for testing
        )
        
        # Calculate statistics
        stats = calculate_dataset_statistics(dataset, tokenizer)
        
        # Test getting a sample
        sample = dataset[0]
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of examples: {len(dataset)}")
        print(f"  Sample input shape: {sample['input_ids'].shape}")
        print(f"  Sample label shape: {sample['labels'].shape}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return None


def test_forward_pass(model, tokenizer):
    """Test a forward pass through the model"""
    print("\nTesting forward pass...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    text = "To be or not to be"
    inputs = tokenizer(text, return_tensors="pt")
    
    # Add labels for loss calculation
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Forward pass successful")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    
    return outputs


def test_memory_usage():
    """Test memory usage"""
    print("\nChecking memory usage...")
    
    if torch.cuda.is_available():
        print(f"  GPU available: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Currently allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("  No GPU available, will use CPU")


def main():
    """Run all tests"""
    print("=" * 50)
    print("DeepSeek V3 Mini Training Setup Test")
    print("=" * 50)
    
    # Test components
    model = test_model_initialization()
    tokenizer = test_tokenizer()
    dataset = test_dataset(tokenizer)
    
    if model and tokenizer:
        test_forward_pass(model, tokenizer)
    
    test_memory_usage()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
