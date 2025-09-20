"""
Data utilities for loading and preprocessing the tiny Shakespeare dataset.
"""

import os
import sys

# Add the parent directory to Python path if needed
parent_dir = os.path.join(os.path.dirname(__file__), '..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ShakespeareDataset(Dataset):
    """Custom dataset for tiny Shakespeare"""
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        block_size: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load the dataset
        logger.info(f"Loading tiny Shakespeare dataset, split: {split}")
        dataset = load_dataset("karpathy/tiny_shakespeare", split=split, cache_dir=cache_dir)
        
        # Get all text
        text = "\n".join(dataset["text"])
        
        # Tokenize all text at once
        logger.info("Tokenizing dataset...")
        self.tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Calculate number of blocks
        self.num_blocks = len(self.tokens) // block_size
        
        logger.info(f"Dataset has {len(self.tokens)} tokens, {self.num_blocks} blocks of size {block_size}")
    
    def __len__(self):
        return self.num_blocks
    
    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        tokens = self.tokens[start_idx:end_idx]
        
        # Create input_ids and labels (shifted by 1 for next token prediction)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_shakespeare_dataset(
    tokenizer,
    split: str = "train", 
    block_size: int = 512,
    cache_dir: Optional[str] = None,
) -> ShakespeareDataset:
    """Create a Shakespeare dataset instance"""
    return ShakespeareDataset(
        tokenizer=tokenizer,
        split=split,
        block_size=block_size,
        cache_dir=cache_dir,
    )


def get_shakespeare_data_collator(pad_token_id: int):
    """Get a data collator that handles padding"""
    def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack all input_ids and labels
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        labels = torch.stack([ex["labels"] for ex in examples])
        
        # Create attention mask (all ones since we're not padding in this case)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
    return collate_fn


def calculate_dataset_statistics(dataset: Dataset, tokenizer):
    """Calculate and print dataset statistics"""
    total_tokens = len(dataset) * dataset.block_size
    vocab_size = len(tokenizer)
    
    stats = {
        "num_examples": len(dataset),
        "block_size": dataset.block_size,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "approx_size_mb": total_tokens * 2 / (1024 * 1024),  # Assuming 2 bytes per token
    }
    
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats
