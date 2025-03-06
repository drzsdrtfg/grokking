#!/usr/bin/env python3
import argparse
import torch
import os
import wandb
import json
from data import get_data, get_tinystories_data, ALL_OPERATIONS, SHARED_VOCAB, VOCAB_SIZE
from model import Transformer
from training import train_epoch, evaluate, inference_demo, generate_text
from torch.cuda.amp import GradScaler
import random
import numpy as np
import itertools
from tqdm import tqdm
from thop import profile

def setup_args():
    parser = argparse.ArgumentParser(description="Mixed training on arithmetic and TinyStories")
    
    # Arithmetic parameters
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x*y")
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
    
    return parser.parse_args()

def create_mixed_batch(arithmetic_batch, tinystories_batch, device, max_seq_len):
    """Create a mixed batch with both arithmetic and TinyStories data"""
    # Unpack the batches
    arith_inputs, arith_targets = arithmetic_batch
    text_inputs, text_targets = tinystories_batch
    
    # Ensure inputs are on the correct device
    arith_inputs = arith_inputs.to(device)
    arith_targets = arith_targets.to(device)
    text_inputs = text_inputs.to(device)
    text_targets = text_targets.to(device)
    
    # Create combined batch dict with task identifiers
    return {
        'arithmetic': {
            'inputs': arith_inputs,
            'targets': arith_targets,
            'seq_len': arith_inputs.size(1)
        },
        'tinystories': {
            'inputs': text_inputs,
            'targets': text_targets,
            'seq_len': text_inputs.size(1)
        }
    }

def process_mixed_batch(model, mixed_batch, criterion):
    """Process a mixed batch, handling different sequence lengths for each task"""
    outputs = {}
    losses = {}
    accuracies = {}
    
    for task, data in mixed_batch.items():
        inputs = data['inputs']
        targets = data['targets']
        
        # Forward pass
        model_outputs = model(inputs)
        
        if task == 'arithmetic':
            # For arithmetic, predict final answer only
            task_output = model_outputs[-1,:,:]
            task_loss = criterion(task_output, targets)
            with torch.no_grad():
                task_acc = (torch.argmax(task_output, dim=1) == targets).float().mean()
        else:  # tinystories
            # For text, align predictions with targets
            batch_size, seq_len = targets.shape
            
            # Reshape outputs to [batch_size, seq_len, vocab_size]
            task_output = model_outputs.permute(1, 0, 2)
            
            # Align outputs and targets
            task_output = task_output[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            task_targets = targets[:, 1:]      # [batch_size, seq_len-1]
            
            # Flatten
            outputs_flat = task_output.reshape(-1, task_output.size(-1))
            targets_flat = task_targets.reshape(-1)
            
            task_loss = criterion(outputs_flat, targets_flat)
            
            with torch.no_grad():
                preds = torch.argmax(outputs_flat, dim=1)
                task_acc = (preds == targets_flat).float().mean()
        
        outputs[task] = task_output
        losses[task] = task_loss
        accuracies[task] = task_acc
    
    # Combine losses (weighted sum)
    combined_loss = sum(losses.values()) / len(losses)
    
    return outputs, losses, accuracies, combined_loss

def train_mixed_step(model, arithmetic_loader, tinystories_loader, optimizer, scheduler, 
                    scaler, device, arithmetic_ratio, criterion, epoch, global_step, 
                    cumulative_flops, flops_per_step):
    """Execute one mixed training step"""
    model.train()
    
    # Get batches from both dataloaders
    try:
        arith_batch = next(arithmetic_loader)
    except StopIteration:
        # Restart arithmetic dataloader
        arithmetic_loader = iter(arithmetic_loader)
        arith_batch = next(arithmetic_loader)
        
    try:
        text_batch = next(tinystories_loader)
    except StopIteration:
        # Restart tinystories dataloader
        tinystories_loader = iter(tinystories_loader)
        text_batch = next(tinystories_loader)
    
    # Create mixed batch
    mixed_batch = create_mixed_batch(arith_batch, text_batch, device, model.seq_len)
    
    optimizer.zero_grad(set_to_none=True)
    
    with torch.cuda.amp.autocast():
        outputs, losses, accuracies, combined_loss = process_mixed_batch(model, mixed_batch, criterion)
    
    # Backward and optimize
    scaler.scale(combined_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    # Update metrics
    cumulative_flops += flops_per_step
    global_step += 1
    
    # Log metrics
    metrics = {
        "training/combined_loss": combined_loss.item(),
        "training/arithmetic_loss": losses['arithmetic'].item(),
        "training/tinystories_loss": losses['tinystories'].item(),
        "training/arithmetic_accuracy": accuracies['arithmetic'].item(),
        "training/tinystories_accuracy": accuracies['tinystories'].item(),
        "learning_rate": scheduler.get_last_lr()[0],
        "epoch": epoch,
        "step": global_step,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    return cumulative_flops, global_step, accuracies, losses

def mixed_evaluate(model, arithmetic_val_loader, tinystories_val_loader, device, epoch, 
                  cumulative_flops, max_eval_batches=50):
    """Evaluate on both arithmetic and TinyStories validation sets"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate on arithmetic
    arithmetic_acc = evaluate(model, arithmetic_val_loader, device, epoch, 
                             cumulative_flops, mode="arithmetic", 
                             max_eval_batches=max_eval_batches)
    
    # Evaluate on TinyStories
    tinystories_acc = evaluate(model, tinystories_val_loader, device, epoch, 
                              cumulative_flops, mode="tinystories", 
                              max_eval_batches=max_eval_batches)
    
    # Log combined metrics
    metrics = {
        "validation/combined_accuracy": (arithmetic_acc + tinystories_acc) / 2,
        "validation/arithmetic_accuracy": arithmetic_acc,
        "validation/tinystories_accuracy": tinystories_acc,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    return (arithmetic_acc + tinystories_acc) / 2

def main():
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="grokking-mixed-training", config=args)
    config = wandb.config
    device = torch.device(config.device)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Define metrics
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("cumulative_flops")
    
    # Create model with largest sequence length
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=VOCAB_SIZE,
        seq_len=config.seq_len,  # Use TinyStories sequence length
        dropout=0.1
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    
    # PHASE 1: Arithmetic-only training
    print("=" * 80)
    print("PHASE 1: Arithmetic-only training")
    print("=" * 80)
    
    # Load arithmetic data
    arith_train_loader, arith_val_loader = get_data(
        config.operation,
        config.prime,
        config.arithmetic_training_fraction,
        config.batch_size
    )
    
    # Estimate FLOPs for arithmetic training
    sample_input = torch.randint(0, VOCAB_SIZE, (config.batch_size, 6)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        arith_flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = 6  # Arithmetic sequence length
        d = config.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * config.num_layers
        arith_flops_per_step = 2 * total_flops
    
    # Training setup
    total_arith_steps = config.arithmetic_steps
    steps_remaining = total_arith_steps
    global_step = 0
    cumulative_flops = 0
    epoch = 0
    
    print(f"Training on arithmetic for {total_arith_steps} steps")
    print(f"Batch size: {config.batch_size}")
    
    # Arithmetic training loop
    while steps_remaining > 0:
        print(f"\nArithmetic Epoch {epoch+1} - Steps remaining: {steps_remaining}")
        
        # Train for one epoch or until steps_remaining reaches 0
        cumulative_flops, global_step, steps_completed = train_epoch(
            model, arith_train_loader, optimizer, scheduler, 
            scaler, device, steps_remaining, global_step, epoch, 
            arith_flops_per_step, cumulative_flops, "arithmetic"
        )
        
        # Update steps remaining
        steps_remaining -= steps_completed
        
        if steps_completed == 0:
            print("No steps completed in this epoch. Check your data loading.")
            break
        
        # Evaluate
        evaluate(model, arith_val_loader, device, epoch, cumulative_flops, "arithmetic")
        
        # Save checkpoint
        if global_step % config.save_steps == 0 or steps_remaining == 0:
            checkpoint_path = os.path.join(args.save_dir, f"arith_checkpoint_step_{global_step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
        epoch += 1
    
    print(f"\nArithmetic training complete after {global_step} steps ({epoch} epochs)")
    
    # PHASE 2: Mixed training with both arithmetic and TinyStories
    print("\n" + "=" * 80)
    print("PHASE 2: Mixed training with arithmetic and TinyStories")
    print("=" * 80)
    
    # Load TinyStories data
    tinystories_train_loader, tinystories_val_loader, vocab = get_tinystories_data(
        config.data_path,
        config.seq_len,
        config.batch_size,
        config.tinystories_training_fraction
    )
    
    # Estimate FLOPs for mixed training (use sequence length from TinyStories which is larger)
    sample_input = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        mixed_flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = config.seq_len
        d = config.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * config.num_layers
        mixed_flops_per_step = 2 * total_flops
    
    # Create iterators for both datasets
    arith_train_iter = iter(arith_train_loader)
    tinystories_train_iter = iter(tinystories_train_loader)
    
    # Reset step counter for mixed training
    total_mixed_steps = config.mixed_steps
    steps_remaining = total_mixed_steps
    mixed_epoch = 0
    
    print(f"Mixed training for {total_mixed_steps} steps")
    print(f"Arithmetic ratio: {config.arithmetic_ratio:.2f}")
    
    # Mixed training loop
    pbar = tqdm(total=total_mixed_steps, desc="Mixed Training")
    
    while steps_remaining > 0:
        mixed_epoch += 1
        
        for step in range(min(steps_remaining, 1000)):  # Process in chunks of 1000 steps
            # Perform one mixed training step
            cumulative_flops, global_step, accuracies, losses = train_mixed_step(
                model, arith_train_iter, tinystories_train_iter,
                optimizer, scheduler, scaler, device,
                config.arithmetic_ratio, criterion,
                mixed_epoch, global_step, cumulative_flops,
                mixed_flops_per_step
            )
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "a_acc": f"{accuracies['arithmetic'].item():.3f}",
                "t_acc": f"{accuracies['tinystories'].item():.3f}",
                "a_loss": f"{losses['arithmetic'].item():.3f}",
                "t_loss": f"{losses['tinystories'].item():.3f}"
            })
            
            # Evaluate periodically
            if global_step % config.eval_steps == 0:
                print(f"\nEvaluating at step {global_step}...")
                mixed_evaluate(model, arith_val_loader, tinystories_val_loader,
                              device, mixed_epoch, cumulative_flops)
            
            # Save checkpoint periodically
            if global_step % config.save_steps == 0:
                checkpoint_path = os.path.join(args.save_dir, f"mixed_checkpoint_step_{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Update steps remaining
        steps_remaining -= min(steps_remaining, 1000)
    
    pbar.close()
    print(f"\nMixed training complete after {global_step} steps ({mixed_epoch} epochs)")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    mixed_evaluate(model, arith_val_loader, tinystories_val_loader, device, mixed_epoch, cumulative_flops)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_mixed_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Demo both capabilities
    print("\n" + "=" * 80)
    print("Arithmetic Inference Demo:")
    print("=" * 80)
    inference_demo(model, config.prime, device, config.operation)
    
    print("\n" + "=" * 80)
    print("TinyStories Text Generation Demo:")
    print("=" * 80)
    seeds = ["Once upon a time", "There was a", "The little", "In a small town"]
    for seed in seeds:
        print(f"Seed: {seed}")
        text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
        print(f"Generated: {text}")
        print("-" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    main()
