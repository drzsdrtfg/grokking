def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, 
                num_steps, epoch, flops_per_step, cumulative_flops, mode="arithmetic"):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    log_interval = 50  # Log every 50 steps as requested
    
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        inputs, targets = batch
        batch_size = inputs.size(0)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(inputs)
            
            if mode == "arithmetic":
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                with torch.no_grad():
                    preds = torch.argmax(output, dim=1)
                    acc = (preds == targets).float().mean().item()
            else:  # tinystories
                # For text, we need to align the predictions with targets
                # Reshape outputs to [batch_size, seq_len, vocab_size]
                outputs = outputs.permute(1, 0, 2)
                
                # We only need positions 0 to seq_len-1 from outputs to predict positions 1 to seq_len in targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [(batch_size * (seq_len-1)), vocab_size]
                targets_flat = targets.reshape(-1)                    # [(batch_size * (seq_len-1))]
                
                # Don't compute loss for padding tokens if vocab exists
                if 'vocab' in globals() and '<pad>' in vocab:
                    pad_idx = vocab['<pad>']
                    mask = targets_flat != pad_idx
                    outputs_flat = outputs_flat[mask]
                    targets_flat = targets_flat[mask]
                
                if len(targets_flat) > 0:  # Only compute loss if we have non-padding tokens
                    loss = criterion(outputs_flat, targets_flat)
                    
                    with torch.no_grad():
                        preds = torch.argmax(outputs_flat, dim=1)
                        acc = (preds == targets_flat).float().mean().item()
                else:
                    loss = torch.tensor(0.0, device=device)
                    acc = 0.0
        
        # Accumulate metrics
        batch_loss = loss.item()
        total_loss += batch_loss * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size
        
        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        cumulative_flops += flops_per_step * batch_size
        global_step = epoch * len(train_loader) + batch_idx
        
        # Log more frequently to get smooth curves
        if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
            metrics = {
                "training/loss": batch_loss,
                "training/accuracy": acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": global_step,
                "cumulative_flops": cumulative_flops,
                "batch": batch_idx
            }
            wandb.log(metrics)
        
        # Check if we've reached the max steps
        if global_step >= num_steps:
            # Log final metrics for this partial epoch
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_acc / total_samples
                
                metrics = {
                    "training/epoch_loss": avg_loss,
                    "training/epoch_accuracy": avg_acc,
                    "epoch": epoch,
                    "cumulative_flops": cumulative_flops
                }
                wandb.log(metrics)
                
            return cumulative_flops
    
    # Log average metrics for the full epoch
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        metrics = {
            "training/epoch_loss": avg_loss,
            "training/epoch_accuracy": avg_acc,
            "epoch": epoch,
            "cumulative_flops": cumulative_flops
        }
        wandb.log(metrics)
    
    return cumulative_flops

def evaluate(model, val_loader, device, epoch, cumulative_flops, mode="arithmetic"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Evaluate in batches to record more detailed progress
    log_interval = 50  # Keep consistent with training logging interval
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = tuple(t.to(device) for t in batch)
            inputs, targets = batch
            batch_size = inputs.size(0)
            
            outputs = model(inputs)
            
            if mode == "arithmetic":
                # For arithmetic, predict final answer only
                output = outputs[-1,:,:]
                loss = criterion(output, targets)
                preds = torch.argmax(output, dim=1)
                
                # Update batch stats
                batch_correct = (preds == targets).sum().item()
                batch_total = targets.size(0)
            else:  # tinystories
                # Use same approach as training
                outputs = outputs.permute(1, 0, 2)
                
                # Align outputs and targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten 
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                
                # Mask out padding tokens if vocab exists
                if 'vocab' in globals() and '<pad>' in vocab:
                    pad_idx = vocab['<pad>']
                    mask = targets_flat != pad_idx
                    outputs_flat = outputs_flat[mask]
                    targets_flat = targets_flat[mask]
                
                if len(targets_flat) > 0:
                    loss = criterion(outputs_flat, targets_flat)
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs_flat, dim=1)
                    batch_correct = (preds == targets_flat).sum().item()
                    batch_total = targets_flat.size(0)
                else:
                    loss = torch.tensor(0.0, device=device)
                    batch_correct = 0
                    batch_total = 0
            
            # Update totals
            total_loss += loss.item() * batch_size
            correct += batch_correct
            total += batch_total
            
            # Log intermediate validation metrics to see progress
            if batch_idx % log_interval == 0 or batch_idx == len(val_loader) - 1:
                if batch_total > 0:
                    batch_acc = batch_correct / batch_total
                else:
                    batch_acc = 0
                
                batch_metrics = {
                    "validation/batch_loss": loss.item(),
                    "validation/batch_accuracy": batch_acc,
                    "epoch": epoch,
                    "cumulative_flops": cumulative_flops,
                    "val_batch": batch_idx
                }
                wandb.log(batch_metrics)
    
    # Calculate and log overall validation metrics
    if total > 0:
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = correct / total
    else:
        avg_loss = 0
        accuracy = 0
    
    metrics = {
        "validation/loss": avg_loss,
        "validation/accuracy": accuracy,
        "epoch": epoch,
        "cumulative_flops": cumulative_flops
    }
    wandb.log(metrics)
    
    print(f"Validation - Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return accuracy
