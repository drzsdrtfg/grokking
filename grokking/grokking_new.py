#!/usr/bin/env python3
import argparse
import subprocess
import os
import json
from datetime import datetime
import time
import random

def main():
    parser = argparse.ArgumentParser(description="Run multiple grokking experiments")
    parser.add_argument("--experiment_type", type=str, choices=["weight_decay", "learning_rate", "model_size", "training_ratio", "all", "baseline"],
                      default="all", help="Type of experiment to run")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per experiment")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--random_order", action="store_true", help="Run experiments in random order")
    
    args = parser.parse_args()
    
    # Timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base arguments for all experiments
    base_args = [
        "python", "cli.py",
        "--operation", "x*y",
        "--prime", "97",
        "--num_steps", "50000",
        "--device", args.device
    ]
    
    # Baseline parameters (default values)
    baseline_params = {
        "weight_decay": 1.0,
        "learning_rate": 1e-3,
        "dim_model": 256,
        "num_layers": 4,
        "training_fraction": 0.2,
        "batch_size": 512
    }
    
    experiments = []
    
    # Add baseline experiment if requested
    if args.experiment_type in ["baseline", "all"]:
        experiments.append({
            "type": "baseline",
            "params": baseline_params.copy(),
        })
    
    # Weight decay experiment
    if args.experiment_type in ["weight_decay", "all"]:
        values = [0.5, 2.0]
        for val in values:
            params = baseline_params.copy()
            params["weight_decay"] = val
            experiments.append({
                "type": "weight_decay",
                "params": params,
            })
    
    # Learning rate experiment
    if args.experiment_type in ["learning_rate", "all"]:
        values = [5e-4, 5e-3]
        for val in values:
            params = baseline_params.copy()
            params["learning_rate"] = val
            experiments.append({
                "type": "learning_rate",
                "params": params,
            })
    
    # Model size experiment
    if args.experiment_type in ["model_size", "all"]:
        sizes = [
            (128, 2),   # Small   # Medium
            (512, 6),   # Large
        ]
        for dim_model, num_layers in sizes:
            params = baseline_params.copy()
            params["dim_model"] = dim_model
            params["num_layers"] = num_layers
            experiments.append({
                "type": "model_size",
                "params": params,
            })
    
    # Training ratio experiment
    if args.experiment_type in ["training_ratio", "all"]:
        values = [0.1, 0.5]
        for val in values:
            params = baseline_params.copy()
            params["training_fraction"] = val
            experiments.append({
                "type": "training_ratio",
                "params": params,
            })
    
    # Randomize experiment order if requested
    if args.random_order:
        random.shuffle(experiments)
    
    # Create experiment log file
    os.makedirs("experiment_logs", exist_ok=True)
    log_file = f"experiment_logs/log_{timestamp}.txt"
    
    # Setup wandb - use a single project for all experiments
    os.environ["WANDB_PROJECT"] = "grokking_new"
    
    # Display experiment plan
    total_exps = len(experiments) * args.runs
    print(f"\n{'=' * 80}")
    print(f"GROKKING EXPERIMENTS PLAN")
    print(f"{'=' * 80}")
    print(f"Total experiments: {total_exps} ({len(experiments)} configs × {args.runs} runs)")
    print(f"Log file: {log_file}")
    print(f"Experiments:")
    for i, exp in enumerate(experiments):
        param_str = ", ".join([f"{k}={v}" for k, v in exp["params"].items()])
        print(f"  {i+1}. {exp['type']}: {param_str}")
    print(f"{'=' * 80}\n")
    
    # Run experiments
    completed = 0
    successful = 0
    
    with open(log_file, "w") as log:
        log.write(f"GROKKING EXPERIMENTS LOG - {timestamp}\n")
        log.write(f"{'=' * 80}\n\n")
        
        for exp_idx, experiment in enumerate(experiments):
            for run_id in range(1, args.runs + 1):
                completed += 1
                
                # Create experiment name with more detailed information
                exp_type = experiment["type"]
                params = experiment["params"]
                
                if exp_type == "baseline":
                    wandb_name = f"x*y_baseline_run_{run_id}"
                else:
                    # Format specific parameter values for better readability
                    if exp_type == "weight_decay":
                        formatted_value = f"wd_{params['weight_decay']}"
                    elif exp_type == "learning_rate":
                        formatted_value = f"lr_{params['learning_rate']}"
                    elif exp_type == "model_size":
                        formatted_value = f"dim{params['dim_model']}_L{params['num_layers']}"
                    elif exp_type == "training_ratio":
                        formatted_value = f"train{float(params['training_fraction'])*100:.0f}pct"
                    else:
                        formatted_value = f"{exp_type}"
                    
                    wandb_name = f"x*y_{formatted_value}_run_{run_id}"
                
                # Set wandb run name
                os.environ["WANDB_NAME"] = wandb_name
                
                # Print progress
                print(f"\nRunning experiment {completed}/{total_exps}")
                print(f"Configuration: {exp_type} (Run {run_id}/{args.runs})")
                print(f"Name: {wandb_name}")
                
                # Log experiment details
                log.write(f"Experiment {completed}/{total_exps}\n")
                log.write(f"Type: {exp_type}\n")
                log.write(f"Run: {run_id}/{args.runs}\n")
                log.write(f"Parameters: {json.dumps(params, indent=2)}\n")
                
                # Prepare command
                cmd = base_args.copy()
                for param, value in params.items():
                    cmd.append(f"--{param}")
                    cmd.append(str(value))
                
                # Print command
                cmd_str = " ".join(cmd)
                print(f"Command: {cmd_str}")
                
                # Run experiment
                start_time = time.time()
                process = subprocess.Popen(cmd)
                return_code = process.wait()
                end_time = time.time()
                
                # Log result
                if return_code == 0:
                    successful += 1
                    log.write(f"Status: SUCCESS\n")
                    print(f"✅ Experiment completed successfully")
                else:
                    log.write(f"Status: FAILED\n")
                    print(f"⚠️ Experiment failed with return code {return_code}")
                
                duration = (end_time - start_time) / 60
                log.write(f"Duration: {duration:.2f} minutes\n")
                log.write(f"Progress: {completed}/{total_exps} completed ({successful} successful)\n")
                log.write(f"{'=' * 50}\n\n")
                log.flush()
                
                # Add a short delay between runs to avoid wandb issues
                time.sleep(5)
    
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENTS COMPLETED")
    print(f"{'=' * 80}")
    print(f"Total: {completed}/{total_exps}")
    print(f"Successful: {successful}/{completed}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
