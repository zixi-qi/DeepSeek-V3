#!/usr/bin/env python3
"""
DeepSeek Mini 1B Model Architecture Demo

This script demonstrates the key architectural features of the DeepSeek Mini 1B model:
- FP8 quantization
- Multi-Token Prediction (MTP) with MoE
- Multi-Head Latent Attention (MLA)
"""

import json
import sys
import os
import torch

from model import Transformer, ModelArgs

def print_model_info(model, args):
    """Print detailed information about the model architecture."""
    print("=" * 60)
    print("DeepSeek Mini 1B Model Architecture")
    print("=" * 60)
    
    # Basic configuration
    print("\n[Configuration]")
    print(f"Model dimension: {args.dim}")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Number of attention heads: {args.n_heads}")
    print(f"Sequence length: {args.max_seq_len:,}")
    print(f"Quantization: {args.dtype.upper()}")
    
    # MoE configuration
    print("\n[Mixture of Experts (MoE)]")
    print(f"Dense layers: {args.n_dense_layers}")
    print(f"MoE layers: {args.n_layers - args.n_dense_layers}")
    print(f"Total experts: {args.n_routed_experts}")
    print(f"Activated experts per token: {args.n_activated_experts}")
    print(f"Shared experts: {args.n_shared_experts}")
    print(f"Expert groups: {args.n_expert_groups}")
    print(f"Scoring function: {args.score_func}")
    print(f"Route scale: {args.route_scale}")
    
    # MLA configuration
    print("\n[Multi-Head Latent Attention (MLA)]")
    print(f"Q LoRA rank: {args.q_lora_rank}")
    print(f"KV LoRA rank: {args.kv_lora_rank}")
    print(f"QK nope head dim: {args.qk_nope_head_dim}")
    print(f"QK rope head dim: {args.qk_rope_head_dim}")
    print(f"V head dim: {args.v_head_dim}")
    
    # Parameter count estimation
    print("\n[Parameter Count Estimation]")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    
    # Layer breakdown
    print("\n[Layer Breakdown]")
    for i, layer in enumerate(model.layers):
        layer_type = "Dense MLP" if i < args.n_dense_layers else "MoE"
        print(f"Layer {i:2d}: {layer_type}")

def test_forward_pass(model, args):
    """Test a simple forward pass through the model."""
    print("\n[Testing Forward Pass]")
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    dummy_tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device="cuda")
    
    print(f"Input shape: {dummy_tokens.shape}")
    print(f"Input tokens (first 10): {dummy_tokens[0, :10].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_tokens)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Check probabilities
    probs = torch.softmax(logits[0], dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    print("\nTop 5 predicted tokens (first position):")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"  {i+1}. Token {idx.item():6d} - Probability: {prob.item():.4f}")

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "inference", "configs", "config_1B.json")
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    
    print(f"Loading DeepSeek Mini 1B model configuration from {config_path}")
    
    # Initialize model
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    
    with torch.device("cuda"):
        model = Transformer(args)
    
    # Print model information
    print_model_info(model, args)
    
    # Test forward pass
    test_forward_pass(model, args)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Note: This uses randomly initialized weights.")
    print("For actual inference, load pretrained weights.")
    print("=" * 60)

if __name__ == "__main__":
    main()
