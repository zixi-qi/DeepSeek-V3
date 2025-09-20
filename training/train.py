#!/usr/bin/env python3
"""
Training script for mini DeepSeek V3 model on tiny Shakespeare dataset using Accelerate.
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed
from datasets import load_dataset
from tqdm.auto import tqdm

# Add the parent directory to Python path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

from mini_model.configuration_deepseek import DeepseekV3Config
from mini_model.modeling_deepseek import DeepseekV3ForCausalLM
from fp8_utils import replace_linear_with_fp8

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to model/config/tokenizer"""
    model_config_path: str = field(
        default="../mini_model/config.json",
        metadata={"help": "Path to model config JSON file"}
    )
    tokenizer_path: str = field(
        default="../mini_model",
        metadata={"help": "Path to tokenizer files"}
    )
    init_from_scratch: bool = field(
        default=True,
        metadata={"help": "Initialize model weights from scratch"}
    )
    use_fp8: bool = field(
        default=False,
        metadata={"help": "Use FP8 quantization for training"}
    )
    fp8_weight_quant: bool = field(
        default=True,
        metadata={"help": "Quantize weights to FP8"}
    )
    fp8_activation_quant: bool = field(
        default=True,
        metadata={"help": "Quantize activations to FP8"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to data"""
    dataset_name: str = field(
        default="karpathy/tiny_shakespeare",
        metadata={"help": "The name of the dataset to use"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training sets"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments"""
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of training steps for warmup"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit total number of checkpoints"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16 training"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bf16 training (requires Ampere GPUs)"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing to save memory"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
    )
    

def tokenize_function(examples, tokenizer, block_size):
    """Tokenize text and create input sequences"""
    # Concatenate all texts
    text = examples["text"]
    
    # Tokenize with padding=False to get actual lengths
    tokenized = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    
    # Concatenate all tokenized sequences
    concatenated_ids = []
    for ids in tokenized["input_ids"]:
        concatenated_ids.extend(ids)
    
    # Create chunks of block_size
    total_length = len(concatenated_ids)
    total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size
    result = {
        "input_ids": [],
        "labels": [],
    }
    
    for i in range(0, total_length, block_size):
        chunk = concatenated_ids[i : i + block_size]
        result["input_ids"].append(chunk)
        result["labels"].append(chunk.copy())  # For language modeling, labels = input_ids
    
    return result


def create_dataloaders(
    dataset_name: str,
    tokenizer,
    block_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 4,
):
    """Create train and eval dataloaders"""
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Tokenize dataset
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, block_size),
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset["train"].column_names,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.tensor([d["input_ids"] for d in x], dtype=torch.long),
            "labels": torch.tensor([d["labels"] for d in x], dtype=torch.long),
        },
        num_workers=num_workers,
    )
    
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            "input_ids": torch.tensor([d["input_ids"] for d in x], dtype=torch.long),
            "labels": torch.tensor([d["labels"] for d in x], dtype=torch.long),
        },
        num_workers=num_workers,
    )
    
    return train_dataloader, eval_dataloader


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "no"),
        log_with=["tensorboard"],
        project_dir=training_args.output_dir,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    set_seed(training_args.seed)
    accelerate_set_seed(training_args.seed)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config and initialize model
    logger.info("Loading model config...")
    with open(model_args.model_config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = DeepseekV3Config(**config_dict)
    
    logger.info("Initializing model...")
    if model_args.init_from_scratch:
        model = DeepseekV3ForCausalLM(config)
        logger.info("Model initialized from scratch")
    else:
        # Load from checkpoint if needed
        model = DeepseekV3ForCausalLM.from_pretrained(
            model_args.model_config_path,
            config=config,
            trust_remote_code=True,
        )
    
    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Enable FP8 quantization if specified
    if model_args.use_fp8:
        logger.info("Enabling FP8 quantization...")
        logger.info(f"  Weight quantization: {model_args.fp8_weight_quant}")
        logger.info(f"  Activation quantization: {model_args.fp8_activation_quant}")
        model = replace_linear_with_fp8(
            model,
            weight_quant=model_args.fp8_weight_quant,
            activation_quant=model_args.fp8_activation_quant
        )
        logger.info("FP8 quantization enabled")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_dataloaders(
        data_args.dataset_name,
        tokenizer,
        data_args.block_size,
        training_args.per_device_train_batch_size,
        training_args.per_device_eval_batch_size,
        training_args.dataloader_num_workers,
    )
    
    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps,
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Initialize trackers
    # Convert training args to a simple dict with only basic types for tensorboard
    tracker_config = {
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "warmup_ratio": training_args.warmup_ratio,
        "weight_decay": training_args.weight_decay,
        "seed": training_args.seed,
        "bf16": training_args.bf16,
        "fp16": training_args.fp16,
    }
    accelerator.init_trackers("deepseek_v3_mini", config=tracker_config)
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader) * training_args.per_device_train_batch_size}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Initialize progress bar
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                # Log metrics
                if completed_steps % training_args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / training_args.gradient_accumulation_steps / training_args.logging_steps
                    logger.info(f"Step: {completed_steps}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                    accelerator.log(
                        {
                            "train_loss": avg_loss,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=completed_steps,
                    )
                    total_loss = 0
                
                # Evaluate
                if completed_steps % training_args.eval_steps == 0:
                    model.eval()
                    losses = []
                    
                    for eval_batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                            loss = outputs.loss
                            losses.append(accelerator.gather_for_metrics(loss))
                    
                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses).item()
                    perplexity = math.exp(eval_loss)
                    
                    logger.info(f"Step {completed_steps}: Eval loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                    accelerator.log(
                        {
                            "eval_loss": eval_loss,
                            "perplexity": perplexity,
                        },
                        step=completed_steps,
                    )
                    model.train()
                
                # Save checkpoint
                if completed_steps % training_args.save_steps == 0:
                    output_dir = f"checkpoint-{completed_steps}"
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    
                    # Also save the model and tokenizer
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Saved checkpoint to {output_dir}")
            
            if completed_steps >= max_train_steps:
                break
    
    # Save final model
    output_dir = os.path.join(training_args.output_dir, "final_model")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training completed! Model saved to {output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
