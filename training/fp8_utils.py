"""
FP8 training utilities for DeepSeek V3 model.

This module provides FP8 quantization support for training using PyTorch's native FP8 types.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F


class FP8Linear(nn.Module):
    """
    A linear layer that supports FP8 quantization for both weights and activations.
    
    This is a drop-in replacement for nn.Linear that quantizes weights to FP8
    and optionally quantizes activations during forward pass.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_quant: bool = True,
        activation_quant: bool = True,
        block_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant = weight_quant
        self.activation_quant = activation_quant
        self.block_size = block_size
        
        # Initialize weight in FP32/BF16
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Initialize weights using the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_to_fp8(self, tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to FP8 format with per-block scaling.
        
        Args:
            tensor: Input tensor to quantize
            block_size: Size of blocks for quantization
            
        Returns:
            Tuple of (quantized_tensor, scale_factors)
        """
        # Ensure the tensor is in float32 for scale computation
        x = tensor.float()
        
        # Reshape for block-wise quantization
        orig_shape = x.shape
        x = x.reshape(-1, block_size)
        
        # Compute per-block scale
        amax = x.abs().max(dim=1, keepdim=True)[0]
        amax = torch.clamp(amax, min=1e-4)
        
        # FP8 E4M3 range is approximately [-448, 448]
        scale = amax / 448.0
        
        # Quantize
        x_scaled = x / scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        
        # Reshape back
        x_fp8 = x_fp8.reshape(orig_shape)
        scale = scale.squeeze(1)
        
        return x_fp8, scale
    
    def dequantize_from_fp8(self, x_fp8: torch.Tensor, scale: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """
        Dequantize from FP8 format.
        
        Args:
            x_fp8: FP8 quantized tensor
            scale: Scale factors
            orig_shape: Original shape to restore
            
        Returns:
            Dequantized tensor
        """
        # Reshape for dequantization
        x = x_fp8.reshape(-1, self.block_size)
        scale = scale.unsqueeze(1)
        
        # Dequantize
        x = x.float() * scale
        
        # Reshape back
        x = x.reshape(orig_shape)
        
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional FP8 quantization.
        
        During training, we simulate FP8 quantization by quantizing and immediately
        dequantizing, which allows gradients to flow while modeling quantization effects.
        """
        # Store original dtype
        orig_dtype = input.dtype
        
        # Quantize activations if enabled
        if self.activation_quant and self.training:
            # Simulate FP8 quantization during training
            # Flatten input for block-wise quantization
            batch_size = input.shape[0]
            input_flat = input.reshape(batch_size, -1)
            
            # Ensure input dimension is divisible by block_size
            if input_flat.shape[1] % self.block_size != 0:
                pad_size = self.block_size - (input_flat.shape[1] % self.block_size)
                input_flat = F.pad(input_flat, (0, pad_size))
            
            # Quantize and dequantize to simulate FP8
            input_fp8, input_scale = self.quantize_to_fp8(input_flat, self.block_size)
            input_dequant = self.dequantize_from_fp8(input_fp8, input_scale, input_flat.shape)
            
            # Remove padding and reshape
            if input_flat.shape[1] != input.reshape(batch_size, -1).shape[1]:
                input_dequant = input_dequant[:, :input.reshape(batch_size, -1).shape[1]]
            input_quant = input_dequant.reshape(input.shape)
        else:
            input_quant = input
        
        # Quantize weights if enabled
        if self.weight_quant and self.training:
            # Simulate FP8 weight quantization
            weight_flat = self.weight.reshape(-1, self.block_size)
            
            # Ensure weight dimension is divisible by block_size
            if weight_flat.shape[0] * self.block_size != self.weight.numel():
                # Pad weight if necessary
                total_elements = self.weight.numel()
                padded_size = ((total_elements + self.block_size - 1) // self.block_size) * self.block_size
                weight_padded = F.pad(self.weight.flatten(), (0, padded_size - total_elements))
                weight_flat = weight_padded.reshape(-1, self.block_size)
            
            weight_fp8, weight_scale = self.quantize_to_fp8(weight_flat, self.block_size)
            weight_dequant = self.dequantize_from_fp8(weight_fp8, weight_scale, weight_flat.shape)
            
            # Remove padding and reshape
            weight_quant = weight_dequant.flatten()[:self.weight.numel()].reshape(self.weight.shape)
        else:
            weight_quant = self.weight
        
        # Perform linear operation
        output = F.linear(input_quant.to(orig_dtype), weight_quant.to(orig_dtype), self.bias)
        
        return output


def replace_linear_with_fp8(model: nn.Module, weight_quant: bool = True, activation_quant: bool = True) -> nn.Module:
    """
    Replace all nn.Linear layers in a model with FP8Linear layers.
    
    Args:
        model: The model to modify
        weight_quant: Whether to quantize weights
        activation_quant: Whether to quantize activations
        
    Returns:
        The modified model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create FP8Linear replacement
            fp8_linear = FP8Linear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                weight_quant=weight_quant,
                activation_quant=activation_quant,
            )
            
            # Copy weights
            with torch.no_grad():
                fp8_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    fp8_linear.bias.copy_(module.bias)
            
            # Replace module
            setattr(model, name, fp8_linear)
        else:
            # Recursively replace in child modules
            replace_linear_with_fp8(module, weight_quant, activation_quant)
    
    return model


# Import math for initialization
import math
