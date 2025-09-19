#!/usr/bin/env python3
"""Test FP8 quantization and parameter counting for DeepSeek Mini."""

import json
import torch
from model import Transformer, ModelArgs, Linear

def count_parameters_detailed(model):
    """Count parameters by layer type and precision."""
    param_stats = {
        'fp8': {'count': 0, 'params': 0},
        'fp32': {'count': 0, 'params': 0},
        'bf16': {'count': 0, 'params': 0}
    }
    
    layer_stats = {}
    
    for name, param in model.named_parameters():
        params = param.numel()
        
        # Determine precision
        if param.dtype == torch.float8_e4m3fn:
            precision = 'fp8'
        elif param.dtype == torch.float32:
            precision = 'fp32'
        elif param.dtype == torch.bfloat16:
            precision = 'bf16'
        else:
            precision = str(param.dtype)
        
        # Update stats
        if precision in param_stats:
            param_stats[precision]['count'] += 1
            param_stats[precision]['params'] += params
        
        # Track by layer
        layer_name = name.split('.')[0]
        if layer_name not in layer_stats:
            layer_stats[layer_name] = 0
        layer_stats[layer_name] += params
        
        # Print detailed info for first few parameters
        if len(layer_stats) <= 5:
            print(f"{name}: {params:,} params, dtype={param.dtype}, shape={list(param.shape)}")
    
    return param_stats, layer_stats

def estimate_actual_params(args):
    """Estimate the actual parameter count for 1B model."""
    # Embedding
    embed_params = args.vocab_size * args.dim
    
    # Per attention layer
    if args.q_lora_rank > 0:
        q_params = args.dim * args.q_lora_rank + args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim)
    else:
        q_params = args.dim * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim)
    
    kv_params = args.dim * (args.kv_lora_rank + args.qk_rope_head_dim)
    kv_params += args.kv_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.v_head_dim)
    o_params = args.n_heads * args.v_head_dim * args.dim
    
    attn_params = q_params + kv_params + o_params
    
    # Dense MLP layers
    dense_mlp_params = 3 * args.dim * args.inter_dim
    
    # MoE layers
    # Note: In actual deployment, only n_activated_experts are used per token
    expert_params = 3 * args.dim * args.moe_inter_dim
    gate_params = args.n_routed_experts * args.dim
    shared_expert_params = 3 * args.dim * (args.n_shared_experts * args.moe_inter_dim)
    
    # Total
    total_dense = args.n_dense_layers * (attn_params + dense_mlp_params)
    total_moe = (args.n_layers - args.n_dense_layers) * (attn_params + gate_params + shared_expert_params)
    
    # For 1B model, we typically count only activated experts
    activated_expert_params = args.n_activated_experts * expert_params
    total_moe += (args.n_layers - args.n_dense_layers) * activated_expert_params
    
    total = embed_params + total_dense + total_moe + embed_params  # embed + lm_head
    
    print("\n[Theoretical Parameter Breakdown]")
    print(f"Embedding: {embed_params:,}")
    print(f"Attention per layer: {attn_params:,}")
    print(f"Dense MLP per layer: {dense_mlp_params:,}")
    print(f"Gate per MoE layer: {gate_params:,}")
    print(f"Shared experts per MoE layer: {shared_expert_params:,}")
    print(f"Activated experts per MoE layer: {activated_expert_params:,}")
    print(f"Total dense layers: {total_dense:,}")
    print(f"Total MoE layers: {total_moe:,}")
    print(f"Total (with only activated experts): {total:,} (~{total/1e9:.2f}B)")
    
    # If all experts are instantiated
    all_expert_params = args.n_routed_experts * expert_params
    total_all = embed_params + total_dense + (args.n_layers - args.n_dense_layers) * (attn_params + gate_params + shared_expert_params + all_expert_params) + embed_params
    print(f"Total (with all experts instantiated): {total_all:,} (~{total_all/1e9:.2f}B)")

def test_fp8_linear():
    """Test FP8 linear layer."""
    print("\n[Testing FP8 Linear Layer]")
    
    # Set FP8 mode
    Linear.dtype = torch.float8_e4m3fn
    
    # Create a small linear layer
    linear = Linear(256, 512).cuda()
    
    print(f"Weight dtype: {linear.weight.dtype}")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Weight has scale: {hasattr(linear.weight, 'scale')}")
    if hasattr(linear.weight, 'scale'):
        print(f"Scale dtype: {linear.scale.dtype}")
        print(f"Scale shape: {linear.scale.shape}")
    
    # Test forward pass
    x = torch.randn(1, 256, dtype=torch.bfloat16, device='cuda')
    try:
        y = linear(x)
        print(f"Forward pass successful! Output shape: {y.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

def main():
    # Load configuration
    config_path = "configs/config_1B.json"
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    
    print("=" * 60)
    print("DeepSeek Mini 1B Parameter Analysis")
    print("=" * 60)
    
    # Test FP8
    test_fp8_linear()
    
    # Initialize model
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    
    with torch.device("cuda"):
        model = Transformer(args)
    
    # Count parameters
    print("\n[Actual Model Parameters]")
    param_stats, layer_stats = count_parameters_detailed(model)
    
    print("\n[Parameter Statistics by Precision]")
    total_params = 0
    for precision, stats in param_stats.items():
        if stats['params'] > 0:
            print(f"{precision}: {stats['count']} tensors, {stats['params']:,} parameters")
            total_params += stats['params']
    
    print(f"\nTotal parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    
    # Theoretical estimate
    estimate_actual_params(args)

if __name__ == "__main__":
    main()
