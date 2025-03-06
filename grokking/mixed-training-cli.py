#!/usr/bin/env python3
"""
Mixed Training CLI for grokking experiments with both arithmetic and TinyStories
"""
import argparse
import os
import torch
import wandb
import numpy as np
import random
import json
from datetime import datetime
from thop import profile
from tqdm import tqdm

from data import get_data, get_tinystories_data, SHARED_VOCAB, VOCAB_SIZE
from model import Transformer
from mixed_batch_trainer import MixedBatchTrainer

def setup_args():
    parser = argparse.ArgumentParser(description="Mixed training on arithmetic and TinyStories")
    
    # Training mode and stages
    parser.add_argument("--mode", type=str, choices=["arithmetic_only", "mixed_only", "two_stage"],
                      default="two_stage", help="Training mode")
    
    # Arithmetic parameters
    parser.add_argument("--operation", type=str, default="x*y")
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--arithmetic_steps", type=int, default=17000)
    parser.add_argument("--arithmetic_training_fraction", type=float, default=0.2)
    
    # TinyStories parameters
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to TinyStories dataset")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--mixed_steps", type=int, default=33000)
    parser.add_argument("--tinystories_training_fraction", type=float, default=0.9)
    
    # Model parameters
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Mixed training parameters
    parser.add_argument("--arithmetic_ratio", type=float, default=0.3,
                       help="Ratio of arithmetic samples in mixed training")
    
    # Checkpoint and logging
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="grokking-mixed",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    return parser.parse_args()

def train_arithmetic(model, args):
    """Train the model on arithmetic data only"""
    print("\n" + "=" * 80)
    print("ARITHMETIC TRAINING")
    print("=" * 80)
    
    device = torch.device(args.device)
    
    # Load arithmetic data
    train_loader, val_loader = get_data(
        args.operation,
        args.prime,
        args.arithmetic_training_fraction,
        args.batch_size
    )
    
    # Estimate FLOPs for arithmetic training
    sample_input = torch.randint(0, VOCAB_SIZE, (args.batch_size, 6)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        arith_flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = 6  # Arithmetic sequence length
        d = args.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * args.num_layers
        arith_flops_per_step = 2 * total_flops
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    # Set up trainer
    trainer = MixedBatchTrainer(model, optimizer, scheduler, device, arith_flops_per_step)
    
    # Training loop
    steps_remaining = args.arithmetic_steps
    global_step = 0
    cumulative_flops = 0
    epoch = 0
    
    print(f"Training on arithmetic for {args.arithmetic_steps} steps")
    
    # Calculate steps per epoch
    steps_per_epoch = min(len(train_loader), 1000)
    
    while steps_remaining > 0:
        epoch += 1
        steps_for_epoch = min(steps_remaining, steps_per_epoch)
        
        print(f"\nEpoch {epoch} - Steps remaining: {steps_remaining}")
        
        # Create iterators
        train_iter = iter(train_loader)
        
        # Dummy iterator for tinystories (won't be used, but needed for trainer API)
        # This is just to keep the same API - we'll use arithmetic_ratio=1.0 to only train on arithmetic
        dummy_iter = iter([])
        
        # Train for epoch
        cumulative_flops, global_step = trainer.train_mixed_epoch(
            train_iter, dummy_iter, steps_for_epoch, 1.0,  # Use only arithmetic
            global_step, epoch, cumulative_flops
        )
        
        # Update steps remaining
        steps_remaining -= steps_for_epoch
        
        # Evaluate
        trainer.mixed_evaluate(val_loader, None, epoch, cumulative_flops)
        
        # Save checkpoint
        if global_step % args.save_steps == 0 or steps_remaining == 0:
            checkpoint_path = os.path.join(args.save_dir, f"arithmetic_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cumulative_flops': cumulative_flops
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print(f"Arithmetic training complete after {global_step} steps")
    
    return model, global_step, cumulative_flops

def train_mixed(model, args, start_step=0, start_flops=0):
    """Train the model on mixed arithmetic and TinyStories data"""
    print("\n" + "=" * 80)
    print("MIXED TRAINING")
    print("=" * 80)
    
    device = torch.device(args.device)
    
    # Load both datasets
    arith_train_loader, arith_val_loader = get_data(
        args.operation,
        args.prime,
        args.arithmetic_training_fraction,
        args.batch_size
    )
    
    tiny_train_loader, tiny_val_loader, vocab = get_tinystories_data(
        args.data_path,
        args.seq_len,
        args.batch_size,
        args.tinystories_training_fraction
    )
    
    # Estimate FLOPs for mixed training (use sequence length from TinyStories)
    sample_input = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        mixed_flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = args.seq_len
        d = args.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * args.num_layers
        mixed_flops_per_step = 2 * total_flops
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    # Set up trainer
    trainer = MixedBatchTrainer(model, optimizer, scheduler, device, mixed_flops_per_step)
    
    # Training loop
    steps_remaining = args.mixed_steps
    global_step = start_step
    cumulative_flops = start_flops
    epoch = 0
    
    print(f"Mixed training for {args.mixed_steps} steps")
    print(f"Arithmetic ratio: {args.arithmetic_ratio:.2f}")
    
    # Steps per epoch - aim for approximately 1000 steps per epoch
    steps_per_epoch = 1000
    
    # Training loop
    while steps_remaining > 0:
        epoch += 1
        steps_for_epoch = min(steps_remaining, steps_per_epoch)
        
        print(f"\nEpoch {epoch} - Steps remaining: {steps_remaining}")
        
        # Create iterators
        arith_train_iter = iter(arith_train_loader)
        tiny_train_iter = iter(tiny_train_loader)
        
        # Train for epoch
        cumulative_flops, global_step = trainer.train_mixed_epoch(
            arith_train_iter, tiny_train_iter, steps_for_epoch, args.arithmetic_ratio,
            global_step, epoch, cumulative_flops
        )
        
        # Update steps remaining
        steps_remaining -= steps_for_epoch
        
        # Evaluate
        trainer.mixed_evaluate(arith_val_loader, tiny_val_loader, epoch, cumulative_flops)
        
        # Save checkpoint
        if global_step % args.save_steps == 0 or steps_remaining == 0:
            checkpoint_path = os.path.join(args.save_dir, f"mixed_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cumulative_flops': cumulative_flops
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print(f"Mixed training complete after {global_step} steps")
    
    return model, global_step, cumulative_flops, vocab

def main():
    args = setup_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize wandb
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.mode}_{args.operation}_{timestamp}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    device = torch.device(args.device)
    
    # Create model with largest sequence length
    model = Transformer(
        num_layers=args.num_layers,
        dim_model=args.dim_model,
        num_heads=args.num_heads,
        num_tokens=VOCAB_SIZE,
        seq_len=args.seq_len,  # Use TinyStories sequence length
        dropout=0.1
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Resume from checkpoint if provided
    start_step = 0
    start_flops = 0
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_step = checkpoint.get('step', 0)
            start_flops = checkpoint.get('cumulative_flops', 0)
            print(f"Resumed from checkpoint at step {start_step}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Train based on selected mode
    if args.mode == "arithmetic_only":
        model, final_step, final_flops = train_arithmetic(model, args)
    elif args.mode == "mixed_only":
        model, final_step, final_flops, vocab = train_mixed(model, args, start_step, start_flops)
    else:  # two_stage
        if start_step == 0:
            # First stage: Arithmetic only
            model, arith_step, arith_flops = train_arithmetic(model, args)
            
            # Second stage: Mixed training
            model, final_step, final_flops, vocab = train_mixed(model, args, arith_step, arith_flops)
        else:
            # If resuming from checkpoint, go straight to mixed training
            print("Resuming from checkpoint, starting with mixed training")
            model, final_step, final_flops, vocab = train_mixed(model, args, start_step, start_flops)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'step': final_step,
        'model_state_dict': model.state_dict(),
        'cumulative_flops': final_flops,
        'args': vars(args)
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Demo inference
    from training import inference_demo, generate_text
    
    print("\nArithmetic Inference Demo:")
    inference_demo(model, args.prime, device, args.operation)
    
    # Generate text if we trained on TinyStories
    if args.mode != "arithmetic_only":
        print("\nTinyStories Text Generation Demo:")
        seeds = ["Once upon a time", "There was a", "The little", "In a small town"]
        for seed in seeds:
            print(f"Seed: {seed}")
            text = generate_text(model, SHARED_VOCAB, device, seed_text=seed, max_len=200)
            print(f"Generated: {text}")
            print("-" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    main()
