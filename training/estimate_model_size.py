#!/usr/bin/env python3
"""
Estimate model size and memory requirements for DeepSeek V3 configurations.
"""

import json
import sys

def estimate_model_params(config):
    """Estimate total parameters for a DeepSeek V3 model configuration."""
    
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    moe_intermediate_size = config['moe_intermediate_size']
    num_layers = config['num_hidden_layers']
    num_heads = config['num_attention_heads']
    num_kv_heads = config['num_key_value_heads']
    q_lora_rank = config['q_lora_rank']
    kv_lora_rank = config['kv_lora_rank']
    qk_nope_dim = config['qk_nope_head_dim']
    qk_rope_dim = config['qk_rope_head_dim']
    v_head_dim = config['v_head_dim']
    vocab_size = config['vocab_size']
    n_routed_experts = config['n_routed_experts']
    n_shared_experts = config['n_shared_experts']
    moe_layer_freq = config['moe_layer_freq']
    
    # Embedding parameters
    embed_params = vocab_size * hidden_size  # input embeddings
    lm_head_params = vocab_size * hidden_size  # output embeddings
    
    # Per-layer parameters
    layer_params = 0
    
    # Attention parameters (with LoRA)
    # Q projection: hidden_size -> q_lora_rank -> num_heads * (qk_nope_dim + qk_rope_dim)
    q_params = hidden_size * q_lora_rank + q_lora_rank * num_heads * (qk_nope_dim + qk_rope_dim)
    
    # KV projection: hidden_size -> kv_lora_rank + qk_rope_dim, then kv_lora_rank -> heads * (qk_nope_dim + v_dim)
    kv_params = hidden_size * (kv_lora_rank + qk_rope_dim) + kv_lora_rank * num_heads * (qk_nope_dim + v_head_dim)
    
    # O projection
    o_params = num_heads * v_head_dim * hidden_size
    
    attention_params = q_params + kv_params + o_params
    
    # MLP parameters (for dense layers)
    mlp_params = 3 * hidden_size * intermediate_size  # gate, up, down projections
    
    # MoE parameters (for MoE layers)
    moe_gate_params = hidden_size * n_routed_experts
    moe_expert_params = n_routed_experts * 3 * hidden_size * moe_intermediate_size
    moe_shared_params = n_shared_experts * 3 * hidden_size * moe_intermediate_size if n_shared_experts > 0 else 0
    moe_params = moe_gate_params + moe_expert_params + moe_shared_params
    
    # Layer norms
    norm_params = 3 * hidden_size  # input norm, post-attention norm, possibly one more
    
    # Calculate total based on layer types
    num_moe_layers = num_layers // moe_layer_freq if moe_layer_freq > 0 else 0
    num_dense_layers = num_layers - num_moe_layers
    
    total_layer_params = (
        num_layers * (attention_params + norm_params) +
        num_dense_layers * mlp_params +
        num_moe_layers * moe_params
    )
    
    total_params = embed_params + lm_head_params + total_layer_params
    
    return {
        'embedding': embed_params,
        'lm_head': lm_head_params,
        'attention': attention_params * num_layers,
        'mlp': mlp_params * num_dense_layers,
        'moe': moe_params * num_moe_layers,
        'norm': norm_params * num_layers,
        'total': total_params,
        'num_moe_layers': num_moe_layers,
        'num_dense_layers': num_dense_layers
    }

def estimate_memory_usage(params, batch_size=1, seq_len=512, precision='bf16'):
    """Estimate memory usage during training."""
    
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'fp8': 1
    }
    
    param_bytes = bytes_per_param.get(precision, 2)
    
    # Model parameters
    model_memory = params['total'] * param_bytes
    
    # Gradients (same size as parameters)
    gradient_memory = params['total'] * param_bytes
    
    # Optimizer states (Adam has 2 momentum terms)
    optimizer_memory = params['total'] * 4 * 2  # Always FP32 for optimizer states
    
    # Activations (rough estimate - depends on implementation)
    # This is a very rough estimate and actual usage can vary significantly
    activation_memory = batch_size * seq_len * 8192 * param_bytes * 10  # Rough multiplier
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model_memory_gb': model_memory / (1024**3),
        'gradient_memory_gb': gradient_memory / (1024**3),
        'optimizer_memory_gb': optimizer_memory / (1024**3),
        'activation_memory_gb': activation_memory / (1024**3),
        'total_memory_gb': total_memory / (1024**3)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_model_size.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\nModel Configuration: {config_path}")
    print("=" * 60)
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Num layers: {config['num_hidden_layers']}")
    print(f"Num experts: {config['n_routed_experts']}")
    print(f"Active experts: {config['num_experts_per_tok']}")
    print(f"Vocab size: {config['vocab_size']}")
    
    params = estimate_model_params(config)
    
    print(f"\nParameter Count:")
    print("=" * 60)
    print(f"Embedding:      {params['embedding']:,}")
    print(f"LM Head:        {params['lm_head']:,}")
    print(f"Attention:      {params['attention']:,}")
    print(f"MLP (Dense):    {params['mlp']:,}")
    print(f"MoE:            {params['moe']:,}")
    print(f"LayerNorm:      {params['norm']:,}")
    print(f"Total:          {params['total']:,}")
    print(f"\nTotal Parameters: {params['total']/1e9:.2f}B")
    
    # Estimate memory for different scenarios
    print(f"\nMemory Estimates (Training):")
    print("=" * 60)
    
    for batch_size in [1, 2, 4]:
        for seq_len in [512, 1024]:
            mem = estimate_memory_usage(params, batch_size, seq_len, 'bf16')
            print(f"\nBatch={batch_size}, Seq={seq_len}:")
            print(f"  Model:      {mem['model_memory_gb']:.2f} GB")
            print(f"  Gradients:  {mem['gradient_memory_gb']:.2f} GB")
            print(f"  Optimizer:  {mem['optimizer_memory_gb']:.2f} GB")
            print(f"  Activations: ~{mem['activation_memory_gb']:.2f} GB")
            print(f"  Total:      ~{mem['total_memory_gb']:.2f} GB")

if __name__ == "__main__":
    main()
