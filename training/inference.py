#!/usr/bin/env python3
"""
Inference script for the trained mini DeepSeek V3 model.
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the parent directory to Python path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    num_return_sequences=1,
    do_sample=True,
):
    """Generate text from a prompt"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode outputs
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def interactive_mode(model, tokenizer):
    """Interactive generation mode"""
    print("\n" + "="*50)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        prompt = input("Enter prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...\n")
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=200,
            temperature=0.8,
            num_return_sequences=1,
        )
        
        print("Generated text:")
        print("-" * 40)
        print(generated[0])
        print("-" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained DeepSeek V3 model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum length of generated text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        # Single generation
        if args.prompt is None:
            # Default Shakespeare-style prompts
            prompts = [
                "To be or not to be",
                "O Romeo, Romeo, wherefore art thou",
                "All the world's a stage",
                "Friends, Romans, countrymen",
            ]
            prompt = prompts[0]
            print(f"No prompt provided, using default: '{prompt}'")
        else:
            prompt = args.prompt
        
        print(f"\nPrompt: {prompt}")
        print("\nGenerating text...\n")
        
        generated_texts = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_samples,
        )
        
        for i, text in enumerate(generated_texts):
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")
            print(text)
            if args.num_samples > 1:
                print()


if __name__ == "__main__":
    main()
