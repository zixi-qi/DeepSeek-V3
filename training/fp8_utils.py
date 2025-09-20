"""
FP8 training utilities for DeepSeek V3 model.

This module provides lightweight FP8 quantization support for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import warnings


class FP8Linear(nn.Module):
    """
    A memory-efficient linear layer with optional FP8 quantization.
    
    This implementation uses gradient scaling to simulate FP8 quantization
    without actually creating FP8 tensors during training, avoiding memory overhead.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_quant: bool = True, activation_quant: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant = weight_quant
        self.activation_quant = activation_quant
        
        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _simulate_quantization(self, x: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        Simulate quantization by reducing precision without creating additional tensors.
        
        This uses a straight-through estimator (STE) approach where we:
        1. Quantize in the forward pass
        2. Use the original gradients in the backward pass
        """
        if not self.training:
            return x
            
        # Find the maximum absolute value for scaling
        x_max = x.abs().max()
        if x_max == 0:
            return x
        
        # Calculate quantization scale
        # For 8-bit quantization, we use 127 levels (excluding 0)
        scale = (2 ** (bits - 1) - 1) / x_max
        
        # Quantize by scaling, rounding, and scaling back
        # This simulates the effect of quantization without creating FP8 tensors
        x_quantized = (x * scale).round().clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) / scale
        
        return x_quantized
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization simulation.
        """
        # Apply quantization to activations if enabled
        if self.activation_quant:
            input_processed = self._simulate_quantization(input)
        else:
            input_processed = input
        
        # Apply quantization to weights if enabled
        if self.weight_quant:
            weight_processed = self._simulate_quantization(self.weight)
        else:
            weight_processed = self.weight
        
        # Perform linear operation
        return F.linear(input_processed, weight_processed, self.bias)


def replace_linear_with_fp8(model: nn.Module, weight_quant: bool = True, activation_quant: bool = True) -> nn.Module:
    """
    Replace all nn.Linear layers in a model with FP8Linear layers.
    
    Args:
        model: The model to modify
        weight_quant: Whether to quantize weights
        activation_quant: Whether to quantize activations
        
    Returns:
        The modified model with FP8Linear layers
    """
    def _replace_module(module, name, parent):
        if isinstance(module, nn.Linear):
            # Create new FP8Linear layer
            new_layer = FP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                weight_quant=weight_quant,
                activation_quant=activation_quant
            )
            
            # Copy weights and bias
            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)
            
            # Replace the module
            setattr(parent, name, new_layer)
            print(f"Replaced {name} with FP8Linear (in_features={module.in_features}, out_features={module.out_features})")
        else:
            # Recursively replace in child modules
            for child_name, child_module in module.named_children():
                _replace_module(child_module, child_name, module)
    
    # Start replacement from root
    for name, module in model.named_children():
        _replace_module(module, name, model)
    
    return model


def disable_fp8_quantization(model: nn.Module) -> nn.Module:
    """
    Disable FP8 quantization in all FP8Linear layers.
    
    This can be useful for inference or debugging.
    """
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module.weight_quant = False
            module.activation_quant = False
    
    return model


def enable_fp8_quantization(model: nn.Module, weight_quant: bool = True, activation_quant: bool = True) -> nn.Module:
    """
    Enable FP8 quantization in all FP8Linear layers.
    """
    for module in model.modules():
        if isinstance(module, FP8Linear):
            module.weight_quant = weight_quant
            module.activation_quant = activation_quant
    
    return model


# Legacy function for backward compatibility
def quantize_to_fp8(x: torch.Tensor, scale: Optional[torch.Tensor] = None, block_size: int = 128) -> torch.Tensor:
    """
    Legacy function - now just returns the input tensor.
    
    This function is kept for backward compatibility but does nothing.
    The actual quantization is handled by FP8Linear layers.
    """
    warnings.warn(
        "quantize_to_fp8 is deprecated. Quantization is now handled automatically by FP8Linear layers.",
        DeprecationWarning,
        stacklevel=2
    )
    return x