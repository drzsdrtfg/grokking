# training.py
from math import ceil
import torch
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from data import get_data
from model import Transformer  # Added missing import
from thop import profile

def main(args: dict):
    wandb.init(project="grokking-study", config=args)
    config = wandb.config
    device = torch.device(config.device)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define metrics
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("cumulative_flops")
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='cumulative_flops')
    wandb.define_metric("validation/accuracy", step_metric='cumulative_flops')
    wandb.define_metric("validation/loss", step_metric='cumulative_flops')
    
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
    )
    
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=4,
        dropout=0.1
    ).to(device)
    
    # Profile the model to get FLOPs per step
    sample_input = torch.randint(0, config.prime + 2, (config.batch_size, 4)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        flops_per_step = 2 * flops  # Account for both forward and backward passes
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        # Fallback to manual estimation
        n = 4
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
        cumulative_flops = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config.num_steps, epoch, flops_per_step, cumulative_flops)
        evaluate(model, val_loader, device, epoch, cumulative_flops)


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, num_steps, epoch, flops_per_step, cumulative_flops):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        with torch.no_grad():
            acc = (torch.argmax(output, dim=1) == labels).float().mean()
        
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
            return cumulative_flops  # Ensure cumulative_flops is returned here
    
    return cumulative_flops  # Ensure cumulative_flops is returned even if the loop doesn't break


def evaluate(model, val_loader, device, epoch, cumulative_flops):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            
            outputs = model(inputs)[-1,:,:]
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += inputs.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Log validation metrics with cumulative_flops
    metrics = {
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    return accuracy
