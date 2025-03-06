#!/usr/bin/env python3
import torch
import os
import random
import itertools
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np

class MixedBatchTrainer:
    """A class to handle mixed batch training with arithmetic and TinyStories data"""
    
    def __init__(self, model, optimizer, scheduler, device, flops_per_step):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.flops_per_step = flops_per_step
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
    def create_mixed_batch(self, arithmetic_batch, tinystories_batch):
        """Create a mixed batch with both arithmetic and TinyStories data"""
        # Unpack the batches
        arith_inputs, arith_targets = arithmetic_batch
        text_inputs, text_targets = tinystories_batch
        
        # Ensure inputs are on the correct device
        arith_inputs = arith_inputs.to(self.device)
        arith_targets = arith_targets.to(self.device)
        text_inputs = text_inputs.to(self.device)
        text_targets = text_targets.to(self.device)
        
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
    
    def process_mixed_batch(self, mixed_batch):
        """Process a mixed batch, handling different sequence lengths for each task"""
        outputs = {}
        losses = {}
        accuracies = {}
        
        for task, data in mixed_batch.items():
            inputs = data['inputs']
            targets = data['targets']
            
            # Forward pass
            model_outputs = self.model(inputs)
            
            if task == 'arithmetic':
                # For arithmetic, predict final answer only
                task_output = model_outputs[-1,:,:]
                task_loss = self.criterion(task_output, targets)
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
                
                task_loss = self.criterion(outputs_flat, targets_flat)
                
                with torch.no_grad():
                    preds = torch.argmax(outputs_flat, dim=1)
                    task_acc = (preds == targets_flat).float().mean()
            
            outputs[task] = task_output
            losses[task] = task_loss
            accuracies[task] = task_acc
        
        # Combine losses (weighted sum)
        combined_loss = sum(losses.values()) / len(losses)
        
        return outputs, losses, accuracies, combined_loss
    
    def train_step(self, arithmetic_loader, tinystories_loader, arithmetic_ratio, 
                   global_step, epoch, cumulative_flops):
        """Execute one mixed training step"""
        self.model.train()
        
        # Select which type of batch to use based on ratio
        use_arithmetic = random.random() < arithmetic_ratio
        
        if use_arithmetic:
            # Get arithmetic batch
            try:
                batch = next(arithmetic_loader)
                task_type = 'arithmetic'
            except StopIteration:
                # Restart arithmetic dataloader
                arithmetic_loader = iter(arithmetic_loader)
                batch = next(arithmetic_loader)
                task_type = 'arithmetic'
        else:
            # Get tinystories batch
            try:
                batch = next(tinystories_loader)
                task_type = 'tinystories'
            except StopIteration:
                # Restart tinystories dataloader
                tinystories_loader = iter(tinystories_loader)
                batch = next(tinystories_loader)
                task_type = 'tinystories'
        
        # Process single task batch
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            if task_type == 'arithmetic':
                # For arithmetic, predict final answer only
                outputs = self.model(inputs)
                output = outputs[-1,:,:]
                loss = self.criterion(output, targets)
                with torch.no_grad():
                    acc = (torch.argmax(output, dim=1) == targets).float().mean()
            else:  # tinystories
                # For text, we need to align the predictions with targets
                outputs = self.model(inputs)
                
                # Reshape outputs to [batch_size, seq_len, vocab_size]
                outputs = outputs.permute(1, 0, 2)
                
                # Align outputs and targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                
                with torch.no_grad():
                    preds = torch.argmax(outputs_flat, dim=1)
                    acc = (preds == targets_flat).float().mean()
        
        # Backward and optimize
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        # Update metrics
        cumulative_flops += self.flops_per_step
        global_step += 1
        
        # Log metrics
        metrics = {
            f"training/{task_type}_loss": loss.item(),
            f"training/{task_type}_accuracy": acc.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "epoch": epoch,
            "step": global_step,
            "cumulative_flops": cumulative_flops,
            "task_type": task_type
        }
        wandb.log(metrics)
        
        return cumulative_flops, global_step, acc.item(), loss.item(), task_type
    
    def train_mixed_epoch(self, arithmetic_loader, tinystories_loader, 
                          steps_per_epoch, arithmetic_ratio, 
                          global_step, epoch, cumulative_flops):
        """Train for one epoch using mixed batches"""
        self.model.train()
        
        # Create iterators if not already iterators
        if not isinstance(arithmetic_loader, itertools.Iterator):
            arithmetic_loader = iter(arithmetic_loader)
        
        if not isinstance(tinystories_loader, itertools.Iterator):
            tinystories_loader = iter(tinystories_loader)
        
        # Training loop
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        arith_acc_sum = 0
        arith_loss_sum = 0
        arith_count = 0
        
        tiny_acc_sum = 0
        tiny_loss_sum = 0
        tiny_count = 0
        
        for _ in pbar:
            # Train one step
            cumulative_flops, global_step, acc, loss, task_type = self.train_step(
                arithmetic_loader, tinystories_loader, arithmetic_ratio,
                global_step, epoch, cumulative_flops
            )
            
            # Update running averages
            if task_type == 'arithmetic':
                arith_acc_sum += acc
                arith_loss_sum += loss
                arith_count += 1
            else:
                tiny_acc_sum += acc
                tiny_loss_sum += loss
                tiny_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                "a_acc": f"{arith_acc_sum/max(1,arith_count):.3f}" if arith_count > 0 else "N/A",
                "t_acc": f"{tiny_acc_sum/max(1,tiny_count):.3f}" if tiny_count > 0 else "N/A",
                "step": global_step
            })
        
        # Calculate epoch metrics
        epoch_metrics = {}
        if arith_count > 0:
            epoch_metrics["epoch/arithmetic_accuracy"] = arith_acc_sum / arith_count
            epoch_metrics["epoch/arithmetic_loss"] = arith_loss_sum / arith_count
        
        if tiny_count > 0:
            epoch_metrics["epoch/tinystories_accuracy"] = tiny_acc_sum / tiny_count
            epoch_metrics["epoch/tinystories_loss"] = tiny_loss_sum / tiny_count
        
        epoch_metrics["epoch"] = epoch
        wandb.log(epoch_metrics)
        
        return cumulative_flops, global_step

    def mixed_evaluate(self, arithmetic_val_loader, tinystories_val_loader, 
                       epoch, cumulative_flops, max_eval_batches=50):
        """Evaluate on both arithmetic and TinyStories validation sets"""
        self.model.eval()
        
        # Arithmetic evaluation
        arith_correct = 0
        arith_total = 0
        arith_loss_sum = 0
        
        # Limit validation to a reasonable number of batches
        arith_val_batches = list(itertools.islice(arithmetic_val_loader, max_eval_batches))
        
        with torch.no_grad():
            for batch in tqdm(arith_val_batches, desc="Arithmetic Validation"):
                inputs, targets = [t.to(self.device) for t in batch]
                
                outputs = self.model(inputs)
                output = outputs[-1,:,:]
                loss = self.criterion(output, targets)
                pred = torch.argmax(output, dim=1)
                
                arith_correct += (pred == targets).sum().item()
                arith_total += targets.size(0)
                arith_loss_sum += loss.item() * inputs.size(0)
        
        # TinyStories evaluation
        tiny_correct = 0
        tiny_total = 0
        tiny_loss_sum = 0
        
        # Limit validation to a reasonable number of batches
        tiny_val_batches = list(itertools.islice(tinystories_val_loader, max_eval_batches))
        
        with torch.no_grad():
            for batch in tqdm(tiny_val_batches, desc="TinyStories Validation"):
                inputs, targets = [t.to(self.device) for t in batch]
                
                # For text validation
                outputs = self.model(inputs)
                
                # Reshape outputs to [batch_size, seq_len, vocab_size]
                outputs = outputs.permute(1, 0, 2)
                
                # Align outputs and targets
                outputs = outputs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                targets = targets[:, 1:]      # [batch_size, seq_len-1]
                
                # Flatten for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                
                # Calculate accuracy
                pred = torch.argmax(outputs_flat, dim=1)
                tiny_correct += (pred == targets_flat).sum().item()
                tiny_total += targets_flat.size(0)
                tiny_loss_sum += loss.item() * inputs.size(0)
        
        # Calculate metrics
        arith_accuracy = arith_correct / arith_total if arith_total > 0 else 0
        arith_avg_loss = arith_loss_sum / len(arith_val_batches) if arith_val_batches else 0
        
        tiny_accuracy = tiny_correct / tiny_total if tiny_total > 0 else 0
        tiny_avg_loss = tiny_loss_sum / len(tiny_val_batches) if tiny_val_batches else 0
        
        # Combined metrics (average of both tasks)
        combined_accuracy = (arith_accuracy + tiny_accuracy) / 2
        combined_loss = (arith_avg_loss + tiny_avg_loss) / 2
        
        print(f"\nValidation Results - Epoch {epoch}")
        print(f"Arithmetic - Accuracy: {arith_accuracy:.4f}, Loss: {arith_avg_loss:.4f}")
        print(f"TinyStories - Accuracy: {tiny_accuracy:.4f}, Loss: {tiny_avg_loss:.4f}")
        print(f"Combined - Accuracy: {combined_accuracy:.4f}, Loss: {combined_loss:.4f}")
        
        # Log metrics
        metrics = {
            "validation/combined_accuracy": combined_accuracy,
            "validation/combined_loss": combined_loss,
            "validation/arithmetic_accuracy": arith_accuracy,
            "validation/arithmetic_loss": arith_avg_loss,
            "validation/tinystories_accuracy": tiny_accuracy,
            "validation/tinystories_loss": tiny_avg_loss,
            "epoch": epoch,
            "cumulative_flops": cumulative_flops
        }
        wandb.log(metrics)
        
        return combined_accuracy, combined_loss
