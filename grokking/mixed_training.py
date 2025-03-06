#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
import wandb
from tqdm import tqdm
import itertools

# Import functions from existing modules
from data import (
    get_data, get_tinystories_data, ALL_OPERATIONS, 
    SHARED_VOCAB, SHARED_VOCAB_REVERSE, VOCAB_SIZE, 
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
)
from model import Transformer
from training import generate_text, inference_demo


class MixedDataLoader:
    """
    A custom data loader that alternates between arithmetic and TinyStories data.
    This allows training on both tasks simultaneously.
    """
    def __init__(self, arithmetic_loader, tinystories_loader, mix_ratio=0.5):
        """
        Args:
            arithmetic_loader: DataLoader for arithmetic data
            tinystories_loader: DataLoader for TinyStories data
            mix_ratio: Ratio of arithmetic samples to total samples
                       (0.5 means equal mix, 0.7 means 70% arithmetic, 30% TinyStories)
        """
        self.arithmetic_loader = arithmetic_loader
        self.tinystories_loader = tinystories_loader
        self.mix_ratio = mix_ratio
        
        # Create iterators for both loaders
        self.arithmetic_iter = iter(arithmetic_loader)
        self.tinystories_iter = iter(tinystories_loader)
        
        # Determine approximate length of mixed loader
        self.length = max(len(arithmetic_loader), len(tinystories_loader))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Return the next batch, mixing between arithmetic and TinyStories based on mix_ratio.
        Each returned batch includes a 'type' field to indicate the data source.
        """
        # Choose which data source to use for this batch
        if random.random() < self.mix_ratio:
            # Use arithmetic data
            try:
                batch = next(self.arithmetic_iter)
                return {'type': 'arithmetic', 'data': batch}
            except StopIteration:
                # Reset arithmetic iterator and try again
                self.arithmetic_iter = iter(self.arithmetic_loader)
                batch = next(self.arithmetic_iter)
                return {'type': 'arithmetic', 'data': batch}
        else:
            # Use TinyStories data
            try:
                batch = next(self.tinystories_iter)
                return {'type': 'tinystories', 'data': batch}
            except StopIteration:
                # Reset TinyStories iterator and try again
                self.tinystories_iter = iter(self.tinystories_loader)
                batch = next(self.tinystories_iter)
                return {'type': 'tinystories', 'data': batch}
    
    def __len__(self):
        return self.length


def mixed_train_epoch(model, mixed_loader, optimizer, scheduler, scaler, device, 
                    steps_remaining, global_step, epoch, flops_per_step, cumulative_flops):
    """Train for one epoch with mixed data or until steps_remaining reaches 0"""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Only process as many batches as needed to reach the step limit
    max_batches = min(len(mixed_loader), steps_remaining)
    
    if max_batches <= 0:
        return cumulative_flops, global_step, 0  # Return early if no steps remaining
    
    # Use a progress bar for better monitoring
    pbar = tqdm(range(max_batches), total=max_batches, 
                desc=f"Epoch {epoch+1}", leave=True)
    
    # Track metrics separately for each data type
    arithmetic_loss = 0
    arithmetic_acc = 0
    arithmetic_count = 0
    
    tinystories_loss = 0
    tinystories_acc = 0
    tinystories_count = 0
    
    steps_completed = 0
    
    for _ in pbar:
        if steps_completed >= steps_remaining:
            break
        
        # Get next mixed batch
        batch = next(mixed_loader)
        batch_type = batch['type']
        inputs, targets = batch['data']
        
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(inputs)
            
            if batch_type == 'arithmetic':
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                with torch.no_grad():
                    acc = (torch.argmax(output, dim=1) == targets).float().mean()
                    
                # Track arithmetic metrics
                arithmetic_loss += loss.item()
                arithmetic_acc += acc.item()
                arithmetic_count += 1
                
            else:  # tinystories
                # For text, we need to align the predictions with targets
                batch_size, seq_len = targets.shape
                
                # Reshape outputs to [batch_size, seq_len, vocab_size]
                outputs = outputs.permute(1, 0, 2)
                
                # We only need positions 0 to seq_len-1 from outputs to predict positions 1 to seq_len in targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [(batch_size * (seq_len-1)), vocab_size]
                targets_flat = targets.reshape(-1)                    # [(batch_size * (seq_len-1))]
                
                loss = criterion(outputs_flat, targets_flat)
                
                with torch.no_grad():
                    preds = torch.argmax(outputs_flat, dim=1)
                    acc = (preds == targets_flat).float().mean()
                
                # Track TinyStories metrics
                tinystories_loss += loss.item()
                tinystories_acc += acc.item()
                tinystories_count += 1
        
        # Backpropagation and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        cumulative_flops += flops_per_step
        steps_completed += 1
        global_step += 1
        
        # Calculate average metrics
        avg_arithmetic_loss = arithmetic_loss / max(1, arithmetic_count)
        avg_arithmetic_acc = arithmetic_acc / max(1, arithmetic_count)
        
        avg_tinystories_loss = tinystories_loss / max(1, tinystories_count)
        avg_tinystories_acc = tinystories_acc / max(1, tinystories_count)
        
        # Log metrics
        metrics = {
            "training/arithmetic_accuracy": avg_arithmetic_acc,
            "training/arithmetic_loss": avg_arithmetic_loss,
            "training/tinystories_accuracy": avg_tinystories_acc,
            "training/tinystories_loss": avg_tinystories_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": global_step,
            "cumulative_flops": cumulative_flops,
            "arithmetic_count": arithmetic_count,
            "tinystories_count": tinystories_count
        }
        wandb.log(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            "a_loss": f"{avg_arithmetic_loss:.4f}", 
            "a_acc": f"{avg_arithmetic_acc:.4f}",
            "t_loss": f"{avg_tinystories_loss:.4f}", 
            "t_acc": f"{avg_tinystories_acc:.4f}",
            "step": global_step,
            "remaining": steps_remaining - steps_completed
        })
    
    return cumulative_flops, global_step, steps_completed


def mixed_evaluate(model, arithmetic_loader, tinystories_loader, device, epoch, cumulative_flops, max_eval_batches=50):
    """Evaluate the model on both arithmetic and TinyStories validation sets"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate on arithmetic data
    total_loss = 0
    correct = 0
    total = 0
    
    # Limit validation to a reasonable number of batches to save time
    arithmetic_batches = list(itertools.islice(arithmetic_loader, max_eval_batches))
    
    with torch.no_grad():
        for batch in tqdm(arithmetic_batches, desc="Validating Arithmetic", leave=False):
            batch = tuple(t.to(device) for t in batch)
            inputs, targets = batch
            
            outputs = model(inputs)
            
            # For arithmetic, predict final answer only
            output = outputs[-1,:,:]
            loss = criterion(output, targets)
            pred = torch.argmax(output, dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            
            total_loss += loss.item() * inputs.size(0)
    
    arithmetic_avg_loss = total_loss / len(arithmetic_batches) if arithmetic_batches else 0
    arithmetic_accuracy = correct / total if total > 0 else 0
    
    print(f"Arithmetic Validation - Loss: {arithmetic_avg_loss:.4f}, Accuracy: {arithmetic_accuracy:.4f}")
    
    # Evaluate on TinyStories data
    total_loss = 0
    correct = 0
    total = 0
    
    # Limit validation to a reasonable number of batches to save time
    tinystories_batches = list(itertools.islice(tinystories_loader, max_eval_batches))
    
    with torch.no_grad():
        for batch in tqdm(tinystories_batches, desc="Validating TinyStories", leave=False):
            batch = tuple(t.to(device) for t in batch)
            inputs, targets = batch
            
            outputs = model(inputs)
            
            # Use same approach as training
            batch_size, seq_len = targets.shape
            
            # Reshape outputs 
            outputs = outputs.permute(1, 0, 2)
            
            # Align outputs and targets
            outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            targets = targets[:, 1:]      # [batch_size, seq_len-1]
            
            # Flatten 
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            
            # Calculate accuracy
            pred = torch.argmax(outputs_flat, dim=1)
            correct += (pred == targets_flat).sum().item()
            total += targets_flat.size(0)
            
            total_loss += loss.item() * inputs.size(0)
    
    tinystories_avg_loss = total_loss / len(tinystories_batches) if tinystories_batches else 0
    tinystories_accuracy = correct / total if total > 0 else 0
    
    print(f"TinyStories Validation - Loss: {tinystories_avg_loss:.4f}, Accuracy: {tinystories_accuracy:.4f}")
    
    # Log combined metrics
    metrics = {
        "validation/arithmetic_accuracy": arithmetic_accuracy,
        "validation/arithmetic_loss": arithmetic_avg_loss,
        "validation/tinystories_accuracy": tinystories_accuracy,
        "validation/tinystories_loss": tinystories_avg_loss,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    # Return combined accuracy as a single metric (weighted average)
    combined_accuracy = (arithmetic_accuracy + tinystories_accuracy) / 2
    return combined_accuracy


def main():
    parser = argparse.ArgumentParser(description="Mixed Arithmetic and TinyStories Training")
    
    # Arithmetic parameters
    parser.add_argument("--arithmetic_operation", type=str, default="x*y", 
                      choices=ALL_OPERATIONS.keys(), help="Arithmetic operation to learn")
    parser.add_argument("--prime", type=int, default=97,
                       help="Prime modulus for arithmetic (must be <= 128 for shared vocab)")
    parser.add_argument("--arithmetic_fraction", type=float, default=0.2,
                       help="Fraction of total data used for arithmetic training")
    
    # TinyStories parameters
    parser.add_argument("--data_path", type=str, default="./tinystories_data",
                       help="Path to TinyStories dataset directory")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Sequence length for text generation")
    
    # Mixed training parameters
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                       help="Ratio of arithmetic samples (0.5 means equal mix)")
    
    # Model parameters
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=25000)
    parser.add_argument("--arithmetic_steps", type=int, default=17000,
                       help="Number of steps to train on arithmetic before mixing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prime > 128:
        parser.error("Prime modulus must be <= 128 for shared vocabulary implementation")
    
    # Initialize wandb
    wandb.init(project="grokking-mixed-training", config=args)
    config = wandb.config
    
    # Set device
    device = torch.device(config.device)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Define wandb metrics
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("cumulative_flops")
    wandb.define_metric("training/arithmetic_accuracy", step_metric='step')
    wandb.define_metric("training/arithmetic_loss", step_metric='step')
    wandb.define_metric("training/tinystories_accuracy", step_metric='step')
    wandb.define_metric("training/tinystories_loss", step_metric='step')
    wandb.define_metric("validation/arithmetic_accuracy", step_metric='step')
    wandb.define_metric("validation/arithmetic_loss", step_metric='step')
    wandb.define_metric("validation/tinystories_accuracy", step_metric='step')
    wandb.define_metric("validation/tinystories_loss", step_metric='step')
    
    print("Loading data...")
    
    # Load arithmetic data
    arithmetic_train_loader, arithmetic_val_loader = get_data(
        config.arithmetic_operation,
        config.prime,
        config.arithmetic_fraction,
        config.batch_size
    )
    
    # Load TinyStories data
    tinystories_train_loader, tinystories_val_loader, vocab = get_tinystories_data(
        config.data_path,
        config.seq_len,
        config.batch_size
    )
    
    # Create model
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=VOCAB_SIZE,
        seq_len=config.seq_len,  # Use the longer sequence length
        dropout=0.1
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scaler = GradScaler()
    
    # Estimate FLOPs
    sample_input = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len)).to(device)
    flops_per_step = 1e12  # Default approximation
    
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(sample_input,))
        flops_per_step = 2 * flops
        print(f"Estimated FLOPs per training step: {flops_per_step}")
    except ImportError:
        print("Could not import thop for FLOPs estimation, using approximation")
        n = config.seq_len
        d = config.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        flops_per_step = 2 * flops_per_layer * config.num_layers
    
    # Training setup
    cumulative_flops = 0
    global_step = 0
    max_epochs = 1000000  # Safety limit
    
    # First phase: Train on arithmetic only
    if config.arithmetic_steps > 0:
        print(f"\n{'=' * 80}")
        print(f"PHASE 1: Training on arithmetic for {config.arithmetic_steps} steps")
        print(f"{'=' * 80}")
        
        steps_remaining = config.arithmetic_steps
        epoch = 0
        
        while steps_remaining > 0 and epoch < max_epochs:
            print(f"\nArithmetic Epoch {epoch+1} - Steps remaining: {steps_remaining}")
            
            # Use the training function from the standard training module
            from training import train_epoch, evaluate
            
            # Train for one epoch on arithmetic data
            cumulative_flops, global_step, steps_completed = train_epoch(
                model, arithmetic_train_loader, optimizer, scheduler, 
                scaler, device, steps_remaining, global_step, epoch, 
                flops_per_step, cumulative_flops, mode="arithmetic"
            )
            
            # Update steps remaining
            steps_remaining -= steps_completed
            
            if steps_completed == 0:
                print("No steps completed in this epoch. Check your data loading.")
                break
            
            # Evaluate on arithmetic data
            evaluate(model, arithmetic_val_loader, device, epoch, cumulative_flops, mode="arithmetic")
            
            # Save checkpoint
            if global_step % config.save_every == 0 or steps_remaining <= 0:
                arithmetic_model_path = f"model_arithmetic_phase1_step_{global_step}.pt"
                torch.save(model.state_dict(), arithmetic_model_path)
                print(f"Checkpoint saved to {arithmetic_model_path}")
            
            epoch += 1
    
    # Phase 2: Mixed training
    remaining_steps = config.num_steps - global_step
    
    if remaining_steps > 0:
        print(f"\n{'=' * 80}")
        print(f"PHASE 2: Mixed training for {remaining_steps} steps with mix ratio {config.mix_ratio}")
        print(f"{'=' * 80}")
        
        # Create mixed data loader
        mixed_loader = MixedDataLoader(
            arithmetic_train_loader, 
            tinystories_train_loader, 
            mix_ratio=config.mix_ratio
        )
        
        epoch = 0
        steps_remaining = remaining_steps
        
        while steps_remaining > 0 and epoch < max_epochs:
            print(f"\nMixed Training Epoch {epoch+1} - Steps remaining: {steps_remaining}")
            
            # Train for one epoch on mixed data
            cumulative_flops, global_step, steps_completed = mixed_train_epoch(
                model, mixed_loader, optimizer, scheduler, 
                scaler, device, steps_remaining, global_step, epoch, 
                flops_per_step, cumulative_flops
            )
            
            # Update steps remaining
            steps_remaining -= steps_completed
            
            if steps_completed == 0:
                print("No steps completed in this epoch. Check your data loading.")
                break
            
            # Evaluate on both datasets
            mixed_evaluate(
                model, 
                arithmetic_val_loader, 
                tinystories_val_loader, 
                device, 
                epoch, 
                cumulative_flops
            )
            
            # Save checkpoint
            if global_step % config.save_every == 0 or steps_remaining <= 0:
                model_path = f"model_mixed_step_{global_step}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"Checkpoint saved to {model_path}")
            
            epoch += 1
    
    # Final model save
    final_model_path = "model_mixed_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save vocabulary
    vocab_path = "model_mixed_vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(SHARED_VOCAB, f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Demonstrate arithmetic inference
    print("\nArithmetic Inference Demo:")
    inference_demo(model, config.prime, device, config.arithmetic_operation)
    
    # Demonstrate text generation
    print("\nText Generation Demo:")
    seeds = ["Once upon a time", "There was a", "The little", "In a small"]
    for seed in seeds:
        print(f"\nSeed: {seed}")
        text = generate_text(model, SHARED_VOCAB, device, seed_text=seed, max_len=200)
        print(f"Generated: {text}")
    
    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
