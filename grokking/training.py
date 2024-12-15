from math import ceil
import torch
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from data import get_data, ALL_OPERATIONS, DIVISION_MODULO_OPERATIONS
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

def main(args: dict):
    wandb.init(project="grokking-study", config=args)
    config = wandb.config
    device = torch.device(config.device)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
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
        seq_len=6,
        dropout=0.1
    ).to(device)

    if args.inference_only and args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        inference_demo(model, config.prime, device, config.operation)
        return

    sample_input = torch.randint(0, config.prime + 4, (config.batch_size, 6)).to(device)
    try:
        flops, _ = profile(model, inputs=(sample_input,))
        flops_per_step = 2 * flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        n = 6
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
        cumulative_flops = train_epoch(model, train_loader, optimizer, scheduler, 
                                     scaler, device, config.num_steps, epoch, 
                                     flops_per_step, cumulative_flops)
        evaluate(model, val_loader, device, epoch, cumulative_flops)
    
    print("\nRunning inference demo...")
    inference_demo(model, config.prime, device, config.operation)
    
    model_path = f"model_{config.operation}_{config.prime}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                num_steps, epoch, flops_per_step, cumulative_flops):
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
            return cumulative_flops
    
    return cumulative_flops

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
    
    metrics = {
        "validation/accuracy": accuracy,
        "validation/loss": avg_loss,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    return accuracy
