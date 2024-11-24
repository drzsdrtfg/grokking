# training.py
from math import ceil
import torch
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from data import get_data
from model import Transformer  # Added missing import

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
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')
    
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
        train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config.num_steps, epoch)
        evaluate(model, val_loader, device, epoch)

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, num_steps, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with autocast():
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
        
        # Scale loss and backprop
        scaler.scale(loss).backward()
        
        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Calculate accuracy
        with torch.no_grad():
            acc = (torch.argmax(output, dim=1) == labels).float().mean()
        
        # Log metrics
        metrics = {
            "training/accuracy": acc,
            "training/loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": epoch * len(train_loader) + batch_idx
        }
        wandb.log(metrics)
        
        if epoch * len(train_loader) + batch_idx >= num_steps:
            return

def evaluate(model, val_loader, device, epoch):
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
    
    metrics = {
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch
    }
    wandb.log(metrics)
    
    return accuracy