#!/usr/bin/env python3
"""
Data preparation for mixed training on arithmetic and TinyStories
This script helps prepare and verify the datasets needed for mixed training
"""

import os
import argparse
import subprocess
import torch
import random
import json
from tqdm import tqdm
from data import get_data, get_tinystories_data, load_tinystories, SHARED_VOCAB

def check_tinystories_data(data_path):
    """Check if TinyStories data exists and is valid"""
    if not os.path.exists(data_path):
        print(f"TinyStories data path {data_path} does not exist")
        return False
        
    # Check if it's a directory with processed stories
    if os.path.isdir(data_path):
        # Look for text files
        txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        if len(txt_files) == 0:
            print(f"No text files found in {data_path}")
            return False
            
        print(f"Found {len(txt_files)} text files in {data_path}")
        
        # Check a few random files
        sample_files = random.sample(txt_files, min(3, len(txt_files)))
        print("Sampling files to check content:")
        
        valid_files = 0
        for filename in sample_files:
            try:
                with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"  - {filename}: {len(content)} characters")
                if len(content) > 50:  # Minimum valid length
                    valid_files += 1
            except Exception as e:
                print(f"  - Error reading {filename}: {e}")
        
        return valid_files > 0
    
    # Check if it's a single file
    elif os.path.isfile(data_path):
        try:
            # Try to read the file
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Just read a sample
            print(f"File {data_path} seems valid with content starting with: {content[:100]}...")
            return True
        except Exception as e:
            print(f"Error reading file {data_path}: {e}")
            return False
    
    return False

def create_sample_tinystories(output_dir, num_stories=50):
    """Create sample TinyStories data for testing"""
    print(f"Creating {num_stories} sample TinyStories in {output_dir}")
    
    # Use the existing create_sample_stories function from create_sample_stories.py
    try:
        from create_sample_stories import create_sample_tinystories
        create_sample_tinystories(output_dir, num_stories)
        return True
    except ImportError:
        print("Could not import create_sample_stories module")
        
        # Fallback to downloading the script
        try:
            # Run the script directly
            subprocess.run(["python", "grokking/create_sample_stories.py", 
                           "--output_dir", output_dir, 
                           "--num_stories", str(num_stories)])
            return True
        except Exception as e:
            print(f"Error creating sample stories: {e}")
            return False

def prepare_tinystories_data(args):
    """Prepare TinyStories data"""
    # Check if data path is specified and exists
    if args.data_path and os.path.exists(args.data_path):
        print(f"Using existing TinyStories data from {args.data_path}")
        if check_tinystories_data(args.data_path):
            return args.data_path
        else:
            print("Data appears invalid")
    
    # If path doesn't exist or is invalid, create sample data
    print("No valid TinyStories data found")
    
    # Create sample data directory
    sample_dir = "./tinystories_data"
    if create_sample_tinystories(sample_dir):
        print(f"Created sample TinyStories data in {sample_dir}")
        return os.path.join(sample_dir, "processed")
    
    print("Failed to create TinyStories data")
    return None

def check_data_loading(args, data_path):
    """Test data loading to ensure it works properly"""
    print("\nTesting data loading...")
    
    # Test arithmetic data loading
    try:
        print("Testing arithmetic data loading...")
        train_loader, val_loader = get_data(
            args.operation,
            args.prime,
            args.arithmetic_training_fraction,
            args.batch_size
        )
        
        # Check a batch
        batch = next(iter(train_loader))
        inputs, targets = batch
        print(f"Arithmetic batch - inputs: {inputs.shape}, targets: {targets.shape}")
        print("Arithmetic data loading successful")
    except Exception as e:
        print(f"Error loading arithmetic data: {e}")
        return False
    
    # Test TinyStories data loading
    try:
        print("\nTesting TinyStories data loading...")
        train_loader, val_loader, vocab = get_tinystories_data(
            data_path,
            args.seq_len,
            args.batch_size,
            args.tinystories_training_fraction
        )
        
        # Check a batch
        batch = next(iter(train_loader))
        inputs, targets = batch
        print(f"TinyStories batch - inputs: {inputs.shape}, targets: {targets.shape}")
        
        # Test decoding a few tokens
        print("\nSample text from TinyStories:")
        reversed_vocab = {v: k for k, v in SHARED_VOCAB.items()}
        sample_text = ""
        for token in inputs[0, :20]:  # First 20 tokens of first sequence
            if token.item() in reversed_vocab:
                char = reversed_vocab[token.item()]
                if char not in ['<bos>', '<eos>', '<pad>']:
                    sample_text += char
        
        print(f"  {sample_text}...")
        print("TinyStories data loading successful")
    except Exception as e:
        print(f"Error loading TinyStories data: {e}")
        return False
    
    return True

def test_token_sharing():
    """Test that token sharing between arithmetic and text works as expected"""
    print("\nTesting token sharing...")
    
    # Ensure arithmetic operations use the same tokens as text
    values = list(range(10))  # Digit tokens for arithmetic
    digit_tokens = [SHARED_VOCAB[str(v)] for v in values]
    
    print("Digit tokens in shared vocabulary:")
    for v, t in zip(values, digit_tokens):
        print(f"  {v} -> {t}")
    
    # Look at alphabet tokens
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet_tokens = [SHARED_VOCAB[c] for c in alphabet]
    
    print("\nSample alphabet tokens in shared vocabulary:")
    for c, t in zip(alphabet[:6], alphabet_tokens[:6]):
        print(f"  {c} -> {t}")
    
    # Special tokens
    print("\nSpecial tokens:")
    special_tokens = ['<bos>', '<eos>', '<pad>']
    for token in special_tokens:
        print(f"  {token} -> {SHARED_VOCAB[token]}")
    
    return True

def show_data_statistics(data_path, args):
    """Show statistics about the datasets"""
    print("\nData Statistics:")
    
    # Arithmetic statistics
    prime = args.prime
    arithmetic_examples = prime * prime
    train_size = int(args.arithmetic_training_fraction * arithmetic_examples)
    val_size = arithmetic_examples - train_size
    
    print(f"Arithmetic (modulo {prime}):")
    print(f"  Total examples: {arithmetic_examples}")
    print(f"  Training examples: {train_size} ({args.arithmetic_training_fraction*100:.1f}%)")
    print(f"  Validation examples: {val_size}")
    
    # TinyStories statistics
    try:
        stories = load_tinystories(data_path)
        total_chars = sum(len(s) for s in stories)
        avg_len = total_chars / max(len(stories), 1)
        
        print(f"\nTinyStories:")
        print(f"  Total stories: {len(stories)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Average story length: {avg_len:.1f} characters")
        
        # Sequence statistics
        seq_len = args.seq_len
        stride = max(1, seq_len // 4)  # Same stride used in data.py
        sequences_per_story = max(1, (avg_len - seq_len) / stride + 1)
        total_seqs = int(len(stories) * sequences_per_story)
        
        print(f"  Sequence length: {seq_len}")
        print(f"  Estimated sequences per story: {sequences_per_story:.1f}")
        print(f"  Estimated total sequences: {total_seqs:,}")
        
        train_seqs = int(args.tinystories_training_fraction * total_seqs)
        val_seqs = total_seqs - train_seqs
        
        print(f"  Training sequences: ~{train_seqs:,} ({args.tinystories_training_fraction*100:.1f}%)")
        print(f"  Validation sequences: ~{val_seqs:,}")
        
    except Exception as e:
        print(f"Error computing TinyStories statistics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Data preparation for mixed training")
    
    # Data paths
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to TinyStories dataset (optional)")
    
    # Configuration parameters
    parser.add_argument("--operation", type=str, default="x*y",
                       help="Arithmetic operation to use")
    parser.add_argument("--prime", type=int, default=97,
                       help="Prime modulus for arithmetic")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Sequence length for TinyStories")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size for training")
    parser.add_argument("--arithmetic_training_fraction", type=float, default=0.2,
                       help="Fraction of arithmetic data to use for training")
    parser.add_argument("--tinystories_training_fraction", type=float, default=0.9,
                       help="Fraction of TinyStories data to use for training")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DATA PREPARATION FOR MIXED TRAINING")
    print("=" * 80)
    
    # 1. Prepare TinyStories data
    data_path = prepare_tinystories_data(args)
    if not data_path:
        print("Failed to prepare TinyStories data")
        return
    
    # 2. Test token sharing
    if not test_token_sharing():
        print("Token sharing test failed")
        return
    
    # 3. Check data loading
    if not check_data_loading(args, data_path):
        print("Data loading check failed")
        return
    
    # 4. Show data statistics
    show_data_statistics(data_path, args)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"TinyStories data path: {data_path}")
    print("\nYou can now run mixed training with the following command:")
    print(f"python mixed_training.py --data_path {data_path} --operation {args.operation} --prime {args.prime}")

if __name__ == "__main__":
    main()
