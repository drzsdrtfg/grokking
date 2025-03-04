from math import ceil
import torch
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
from data import get_data, get_tinystories_data, ALL_OPERATIONS, DIVISION_MODULO_OPERATIONS
from data import SHARED_VOCAB, SHARED_VOCAB_REVERSE, VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
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
    """Demonstrate model inference with shared vocabulary tokens"""
    model.eval()
    
    # Map operation to appropriate tokens
    op_str = operation[1]  # Extract operation symbol (*, +, -, /)
    op_token = SHARED_VOCAB.get(op_str, SHARED_VOCAB['*'])  # Default to * if not found
    eq_token = SHARED_VOCAB['=']
    
    # Generate random examples
    x = torch.randint(0, prime, (num_examples,))
    y = torch.randint(1 if operation in DIVISION_MODULO_OPERATIONS else 0, prime, (num_examples,))
    
    # Create tensor with BOS and EOS tokens
    bos = torch.ones_like(x) * BOS_TOKEN
    eos = torch.ones_like(x) * EOS_TOKEN
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    
    inputs = torch.stack([bos, x, op, y, eq, eos], dim=1)
    
    # Calculate expected results
    x_np, y_np = x.numpy(), y.numpy()
    actual_results = []
    for i in range(num_examples):
        _, _, result = ALL_OPERATIONS[operation](x_np[i], y_np[i], prime)
        actual_results.append(result)
    
    # Get model predictions
    predictions = predict(model, inputs, device)
    
    # Print results
    print(f"\nInference Results for operation: {operation}")
    print("-" * 50)
    for i in range(num_examples):
        print(f"Input: {x_np[i]} {operation} {y_np[i]} = {predictions[i].item()}")
        print(f"Actual: {actual_results[i]}")
        print(f"Correct: {predictions[i].item() == actual_results[i]}")
        print("-" * 50)

def generate_text(model, vocab, device, seed_text="", max_len=100, temperature=0.8):
    """Generate text using the trained model with shared vocabulary"""
    model.eval()
    
    # Convert seed text to token IDs using shared vocabulary
    tokens = [vocab.get(char, vocab[' ']) for char in seed_text]
    
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
            
            # Stop if EOS token or PAD token is generated
            if next_token == EOS_TOKEN or next_token == PAD_TOKEN:
                break
                
            # Add to the sequence
            input_sequence = torch.cat(
                [input_sequence, torch.tensor([[next_token]], device=device)], dim=1
            )
            
            # Add the character to the generated text
            if next_token in SHARED_VOCAB_REVERSE:
                char = SHARED_VOCAB_REVERSE[next_token]
                if char not in ['<bos>', '<eos>', '<pad>']:
                    generated += char
    
    return generated

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                num_steps, epoch, flops_per_step, cumulative_flops, mode="arithmetic"):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        inputs, targets = batch
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(inputs)
            
            if mode == "arithmetic":
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                with torch.no_grad():
                    acc = (torch.argmax(output, dim=1) == targets).float().mean()
            else:  # tinystories
                # For text, we need to align the predictions with targets
                # First, reshape targets
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
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        cumulative_flops += flops_per_step
        metrics = {
            "training/accuracy": acc,
            "training/loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": epoch * len(train_loader) + batch_idx,
            "cumulative_flops": cumulative_flops
        }
        wandb.log(metrics)
        
        if epoch * len(train_loader) + batch_idx >= num_steps:
            return cumulative_flops
    
    return cumulative_flops

def evaluate(model, val_loader, device, epoch, cumulative_flops, mode="arithmetic"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
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
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    metrics = {
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    return accuracy

def main(args):
    wandb.init(project="grokking-study", config=args)
    config = wandb.config
    device = torch.device(config.device)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("cumulative_flops")
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='cumulative_flops')
    wandb.define_metric("validation/accuracy", step_metric='cumulative_flops')
    wandb.define_metric("validation/loss", step_metric='cumulative_flops')
    
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(config.data_path) if config.data_path else "./", "vocabulary.json")
    
    # Load data based on mode
    if config.mode == "arithmetic":
        train_loader, val_loader = get_data(
            config.operation,
            config.prime,
            config.training_fraction,
            config.batch_size
        )
        # For arithmetic mode, use a short sequence length 
        effective_seq_len = 6  # Original arithmetic sequence length
        vocab = SHARED_VOCAB  # Use shared vocabulary
    else:  # tinystories
        train_loader, val_loader, vocab = get_tinystories_data(
            config.data_path,
            config.seq_len,
            config.batch_size,
            config.training_fraction
        )
        effective_seq_len = config.seq_len  # Use the specified sequence length for text
    
    # Always use the same vocabulary size for the model
    num_tokens = VOCAB_SIZE  # Fixed vocabulary size for both modes
    
    # Create model with consistent vocabulary size
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=num_tokens,
        seq_len=effective_seq_len,
        dropout=0.1
    ).to(device)

    # Handle inference-only mode
    if args.inference_only and args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        
        if config.mode == "arithmetic":
            inference_demo(model, config.prime, device, config.operation)
        else:  # tinystories
            # Generate text samples
            print("\nGenerated Text Samples:")
            print("-" * 50)
            seeds = ["Once upon a time", "There was a", "The little", "In a small town"]
            for seed in seeds:
                print(f"Seed: {seed}")
                text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
                print(f"Generated: {text}")
                print("-" * 50)
        
        return

    # Estimate FLOPs
    sample_input = torch.randint(0, num_tokens, (config.batch_size, effective_seq_len)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = effective_seq_len
        d = config.dim_model
        flops_per_layer = 2 * n**2 * d + 2 * n * d**2
        total_flops = flops_per_layer * config.num_layers
        flops_per_step = 2 * total_flops
    
    cumulative_flops = 0
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scaler = GradScaler()
    
    num_epochs = ceil(config.num_steps / len(train_loader))
    
    for epoch in tqdm(range(num_epochs)):
        cumulative_flops = train_epoch(
            model, train_loader, optimizer, scheduler, 
            scaler, device, config.num_steps, epoch, 
            flops_per_step, cumulative_flops, config.mode
        )
        evaluate(model, val_loader, device, epoch, cumulative_flops, config.mode)
    
    # Save model and perform inference demo
    if config.mode == "arithmetic":
        model_path = f"model_{config.operation}_{config.prime}_shared.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save vocabulary for reference
        vocab_path = f"model_{config.operation}_{config.prime}_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")
        
        print("\nRunning inference demo...")
        inference_demo(model, config.prime, device, config.operation)
    else:  # tinystories
        model_path = f"model_tinystories_{config.seq_len}_shared.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary
        vocab_path = f"model_tinystories_{config.seq_len}_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vocabulary saved to {vocab_path}")
        
        # Generate text samples
        print("\nGenerated Text Samples:")
        print("-" * 50)
        seeds = ["Once upon a time", "There was a", "The little", "In a small town"]
        for seed in seeds:
            print(f"Seed: {seed}")
            text = generate_text(model, vocab, device, seed_text=seed, max_len=200)
            print(f"Generated: {text}")
            print("-" * 50)
