from math import ceil
import torch
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
import time
from data import get_data, get_tinystories_data, ALL_OPERATIONS, DIVISION_MODULO_OPERATIONS
from model import Transformer
from thop import profile

def predict(model, inputs, device):
    """Make predictions using the trained model"""
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)[-1,:,:]
        predictions = torch.argmax(outputs, dim=1)
    return predictions

def inference_demo(model, prime, device, operation="x*y", num_examples=5):
    """Demonstrate model inference with BOS and EOS tokens"""
    model.eval()
    
    x = torch.randint(0, prime, (num_examples,))
    y = torch.randint(1 if operation in DIVISION_MODULO_OPERATIONS else 0, prime, (num_examples,))
    
    eq_token = prime
    op_token = prime + 1
    bos_token = prime + 2
    eos_token = prime + 3
    
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    bos = torch.ones_like(x) * bos_token
    eos = torch.ones_like(x) * eos_token
    
    inputs = torch.stack([bos, x, op, y, eq, eos], dim=1)
    
    x_np, y_np = x.numpy(), y.numpy()
    actual_results = []
    for i in range(num_examples):
        _, _, result = ALL_OPERATIONS[operation](x_np[i], y_np[i], prime)
        actual_results.append(result)
    
    predictions = predict(model, inputs, device)
    
    print(f"\nInference Results for operation: {operation}")
    print("-" * 50)
    for i in range(num_examples):
        print(f"Input: {x_np[i]} {operation} {y_np[i]} = {predictions[i].item()}")
        print(f"Actual: {actual_results[i]}")
        print(f"Correct: {predictions[i].item() == actual_results[i]}")
        print("-" * 50)

def generate_text(model, vocab, device, seed_text="", max_len=100, temperature=0.8):
    """Generate text using the trained model"""
    model.eval()
    
    # Create inverse vocabulary (id -> char)
    id_to_char = {v: k for k, v in vocab.items()}
    
    # Convert seed text to token IDs
    tokens = [vocab.get(char, vocab['<pad>']) for char in seed_text]
    
    # Add BOS token
    tokens = [vocab['<bos>']] + tokens
    
    # Create input tensor
    input_sequence = torch.tensor([tokens], device=device)
    
    # Generate text
    generated = seed_text
    
    for _ in range(max_len):
        # Keep only the last tokens that fit in the model's context
        if input_sequence.size(1) > model.position_embeddings.weight.size(0):
            input_sequence = input_sequence[:, -model.position_embeddings.weight.size(0):]
        
        # Forward pass
        with torch.no_grad():
            output = model(input_sequence)
            # Get logits for the next token
            next_token_logits = output[-1, -1, :]
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if EOS token is generated
            if next_token == vocab['<eos>'] or next_token == vocab['<pad>']:
                break
                
            # Add to the sequence
            input_sequence = torch.cat(
                [input_sequence, torch.tensor([[next_token]], device=device)], dim=1
            )
            
            # Add the character to the generated text
            if next_token in id_to_char:
                char = id_to_char[next_token]
                if char not in ['<bos>', '<eos>', '<pad>']:
                    generated += char
    
    return generated

def log_text_samples(model, vocab, device, seed_texts, wandb_run, step, prefix="sample"):
    """Generate and log text samples to WandB"""
    samples = {}
    
    # Generate samples from each seed text
    for i, seed in enumerate(seed_texts):
        generated_text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
        samples[f"{prefix}_{i+1}"] = f"Seed: '{seed}'\n\n{generated_text}"
    
    # Log to wandb
    wandb_run.log({"text_samples": samples}, step=step)
    
    return samples

def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_wandb_logging(args, model):
    """Set up improved WandB logging"""
    # Create a more descriptive run name
    if args.mode == "arithmetic":
        run_name = f"arith_{args.operation}_{args.num_layers}L_{args.dim_model}d"
    else:
        run_name = f"tiny_{args.seq_len}seq_{args.num_layers}L_{args.dim_model}d"
    
    # Initialize wandb with improved config
    wandb_run = wandb.init(
        project="grokking-study",
        name=run_name,
        config=args,
        reinit=True
    )
    
    # Add model architecture details to config
    wandb_run.config.update({
        "model_parameters": count_parameters(model),
        "model_type": "Transformer",
        "model_structure": f"{args.num_layers}L_{args.dim_model}d_{args.num_heads}h"
    })
    
    # Define metrics with consistent step tracking
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    
    # Training metrics
    wandb.define_metric("training/loss", step_metric="step")
    wandb.define_metric("training/accuracy", step_metric="step")
    wandb.define_metric("training/perplexity", step_metric="step")
    wandb.define_metric("training/tokens_per_sec", step_metric="step")
    
    # Validation metrics
    wandb.define_metric("validation/loss", step_metric="step")
    wandb.define_metric("validation/accuracy", step_metric="step")
    wandb.define_metric("validation/perplexity", step_metric="step")
    
    # Learning rate
    wandb.define_metric("learning_rate", step_metric="step")
    
    # Timing metrics
    wandb.define_metric("time/epoch_duration_sec", step_metric="epoch")
    wandb.define_metric("time/epoch_samples", step_metric="epoch")
    
    # Resource usage
    wandb.define_metric("resources/gpu_utilization", step_metric="step")
    
    return wandb_run

def get_gpu_memory_usage():
    """Get GPU memory usage if available"""
    try:
        gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        return gpu_memory_used
    except:
        return 0

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Parse device setting
    device = torch.device(args.device)
    
    # Load data based on mode
    if args.mode == "arithmetic":
        train_loader, val_loader = get_data(
            args.operation,
            args.prime,
            args.training_fraction,
            args.batch_size
        )
        num_tokens = args.prime + 4  # Include special tokens
        seq_len = 6
        vocab = None
    else:  # tinystories
        train_loader, val_loader, vocab = get_tinystories_data(
            args.data_path,
            args.seq_len,
            args.batch_size,
            args.training_fraction
        )
        num_tokens = len(vocab)
        seq_len = args.seq_len
    
    # Create model
    model = Transformer(
        num_layers=args.num_layers,
        dim_model=args.dim_model,
        num_heads=args.num_heads,
        num_tokens=num_tokens,
        seq_len=seq_len,
        dropout=0.1
    ).to(device)
    
    # Initialize wandb
    wandb_run = setup_wandb_logging(args, model)
    
    # Use gradient checkpointing for larger models if requested
    if hasattr(args, 'use_gradient_checkpointing') and args.use_gradient_checkpointing:
        model.use_gradient_checkpointing()
    
    # Handle inference-only mode
    if args.inference_only and args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        
        if args.mode == "arithmetic":
            inference_demo(model, args.prime, device, args.operation)
        else:  # tinystories
            # Load vocabulary
            vocab_path = args.model_path.replace(".pt", "_vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    vocab = json.load(f)
            
            # Generate text samples
            print("\nGenerated Text Samples:")
            print("-" * 50)
            seeds = ["Once upon a time", "There was a", "The little", "In a small town"]
            for seed in seeds:
                print(f"Seed: {seed}")
                text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
                print(f"Generated: {text}")
                print("-" * 50)
            
            # Log samples to wandb
            log_text_samples(model, vocab, device, seeds, wandb_run, 0)
        
        wandb.finish()
        return

    # Estimate FLOPs
    sample_input = torch.randint(0, num_tokens, (args.batch_size, seq_len)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        flops_per_step = 2 * flops
        wandb.run.summary["model/flops_per_forward"] = flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = seq_len
        d = args.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * args.num_layers
        flops_per_step = 2 * total_flops
    
    # Log model architecture as config
    model_config = {
        "num_layers": args.num_layers,
        "dim_model": args.dim_model,
        "num_heads": args.num_heads,
        "num_tokens": num_tokens,
        "seq_len": seq_len,
        "trainable_params": count_parameters(model)
    }
    wandb.config.update({"model_details": model_config})
    
    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scaler = GradScaler()
    
    num_epochs = ceil(args.num_steps / len(train_loader))
    
    # Create seed texts for periodic sample generation (for tinystories)
    seed_texts = ["Once upon a time", "There was a", "The little", "In a small town"]
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_start_time = time.time()
        
        # Train for one epoch
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, 
            scaler, device, args.num_steps, epoch, 
            global_step, args.mode, wandb_run, vocab
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, device, epoch, 
            global_step, args.mode, wandb_run
        )
        
        # Calculate epoch statistics
        epoch_duration = time.time() - epoch_start_time
        samples_processed = len(train_loader) * args.batch_size
        
        epoch_metrics = {
            "epoch": epoch,
            "time/epoch_duration_sec": epoch_duration,
            "time/epoch_samples": samples_processed,
            "time/samples_per_sec": samples_processed / epoch_duration
        }
        wandb_run.log(epoch_metrics, step=global_step)
        
        # Generate and log text samples (for tinystories)
        if args.mode == "tinystories" and epoch % 5 == 0:
            samples = log_text_samples(
                model, vocab, device, seed_texts, 
                wandb_run, global_step
            )
            print("\nSample generation at epoch", epoch)
            for key, sample in list(samples.items())[:2]:  # Print first two samples
                print(f"{sample}\n---")
        
        # Save model if validation improves
        if val_metrics["validation/loss"] < best_val_loss:
            best_val_loss = val_metrics["validation/loss"]
            checkpoint_path = f"model_{args.mode}_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["best_epoch"] = epoch
        
        # Periodic checkpoint
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = f"model_{args.mode}_epoch{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            if args.mode == "tinystories":
                # Save vocabulary
                vocab_path = checkpoint_path.replace(".pt", "_vocab.json")
                with open(vocab_path, 'w') as f:
                    json.dump(vocab, f)
    
    # Save final model
    if args.mode == "arithmetic":
        model_path = f"model_{args.operation}_{args.prime}.pt"
        torch.save(model.state_dict(), model_path)
        
        print("\nRunning inference demo...")
        inference_demo(model, args.prime, device, args.operation)
    else:  # tinystories
        model_path = f"model_tinystories_{args.seq_len}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary
        vocab_path = f"model_tinystories_{args.seq_len}_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        # Generate text samples
        print("\nGenerated Text Samples:")
        print("-" * 50)
        for seed in seed_texts:
            print(f"Seed: {seed}")
            text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
            print(f"Generated: {text}")
            print("-" * 50)
    
    wandb_run.finish()
    print(f"\nModel saved to {model_path}")

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                num_steps, epoch, global_step, mode, wandb_run, vocab=None):
    """Train for one epoch with improved logging"""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_start_time = time.time()
    batch_start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        step_start_time = time.time()
        
        # Move data to device
        batch = tuple(t.to(device) for t in batch)
        inputs, targets = batch
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with autocast for mixed precision
        with autocast():
            outputs = model(inputs)
            
            if mode == "arithmetic":
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                with torch.no_grad():
                    preds = torch.argmax(output, dim=1)
                    correct = (preds == targets).sum().item()
                    total = targets.size(0)
            else:  # tinystories
                # For text, align predictions with targets
                batch_size, seq_len = targets.shape
                
                # Reshape outputs to [batch_size, seq_len, vocab_size]
                outputs = outputs.permute(1, 0, 2)
                
                # Align outputs and targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [(batch_size * (seq_len-1)), vocab_size]
                targets_flat = targets.reshape(-1)                    # [(batch_size * (seq_len-1))]
                
                loss = criterion(outputs_flat, targets_flat)
                
                with torch.no_grad():
                    preds = torch.argmax(outputs_flat, dim=1)
                    correct = (preds == targets_flat).sum().item()
                    total = targets_flat.size(0)
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Accumulate statistics
        epoch_loss += loss.item() * inputs.size(0)
        epoch_correct += correct
        epoch_total += total
        
        # Calculate performance metrics
        accuracy = correct / total
        perplexity = calculate_perplexity(loss.item())
        
        # Calculate tokens processed per second
        batch_duration = time.time() - batch_start_time
        tokens_per_sec = (inputs.size(0) * inputs.size(1)) / batch_duration
        
        # GPU utilization and memory stats
        gpu_mem_used = get_gpu_memory_usage()
        
        # Log metrics
        metrics = {
            "training/loss": loss.item(),
            "training/accuracy": accuracy,
            "training/perplexity": perplexity,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": global_step,
            "training/tokens_per_sec": tokens_per_sec,
            "resources/gpu_memory_gb": gpu_mem_used,
        }
        
        # Log at regular intervals to avoid flooding WandB
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            wandb_run.log(metrics, step=global_step)
        
        # Periodic text generation for tinystories (every 1000 steps)
        if mode == "tinystories" and global_step > 0 and global_step % 1000 == 0 and vocab is not None:
            # Generate a single sample to track improvement
            seed_text = "Once upon a time"
            text = generate_text(model, vocab, device, seed_text=seed_text, max_len=100)
            wandb_run.log({
                "samples/text": wandb.Html(f"<h3>Step {global_step}</h3><p><b>Seed:</b> {seed_text}</p><p>{text}</p>")
            }, step=global_step)
        
        global_step += 1
        batch_start_time = time.time()
        
        # Exit if we've reached the maximum number of steps
        if global_step >= num_steps:
            break
    
    # Log epoch summary
    epoch_accuracy = epoch_correct / epoch_total
    epoch_avg_loss = epoch_loss / len(train_loader)
    epoch_duration = time.time() - epoch_start_time
    
    epoch_summary = {
        "training/epoch_loss": epoch_avg_loss,
        "training/epoch_accuracy": epoch_accuracy,
        "training/epoch_perplexity": calculate_perplexity(epoch_avg_loss),
        "time/train_epoch_duration": epoch_duration
    }
    wandb_run.log(epoch_summary, step=global_step)
    
    return global_step

def evaluate(model, val_loader, device, epoch, global_step, mode, wandb_run):
    """Evaluate model with improved logging"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, targets = batch
            
            outputs = model(inputs)
            
            if mode == "arithmetic":
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                pred = torch.argmax(output, dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
            else:  # tinystories
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
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    perplexity = calculate_perplexity(avg_loss)
    eval_duration = time.time() - eval_start_time
    
    # Log validation metrics
    metrics = {
        "validation/loss": avg_loss,
        "validation/accuracy": accuracy,
        "validation/perplexity": perplexity,
        "time/validation_duration": eval_duration,
    }
    
    wandb_run.log(metrics, step=global_step)
    
    return metrics
