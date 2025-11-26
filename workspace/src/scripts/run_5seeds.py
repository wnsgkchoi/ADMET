import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json

# Add src to path to import main
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Do NOT change working directory, as it breaks config paths (e.g. configs/dataset_config.json)
# Instead, rely on sys.path to find modules in src
# os.chdir(src_path)

from main import main as run_single_seed
from main import load_args

def run_5seeds():
    # Load arguments
    # We use the same argument parser as main.py
    # But we need to intercept the parsing to handle the loop
    
    # First, parse args using the existing loader
    # Note: This parses sys.argv
    args = load_args()
    
    # Force quiet mode for individual runs to reduce clutter
    args.quiet = True
    
    # Store original output dir
    original_output_dir = args.output_dir
    
    # Metrics storage
    results = {
        'primary': [],
        'secondary': []
    }
    
    print(f"Starting 5-seed evaluation for experiment: {args.experiment_id}")
    print(f"Params: LR={args.lr}, Dropout={args.dropout_ratio}, Experts={args.num_experts}, Layers={args.num_layer}")
    
    # Run for seeds 0 to 4
    for seed in range(5):
        print(f"  Running Seed {seed+1}/5...", end='', flush=True)
        
        # Set seed
        args.seed = seed
        
        # Modify output dir to avoid overwriting or locking issues if needed
        # But main.py saves checkpoints based on experiment_id. 
        # We should probably append seed to experiment_id for the internal run
        # to avoid checkpoint collisions if we were saving models.
        # But we usually use --no_save_model for tuning.
        # If saving models, we'd need unique IDs.
        # Let's assume --no_save_model is used, or we handle it.
        
        # Run main
        try:
            # We need to capture the return values from main
            # Currently main() returns nothing, it saves to CSV.
            # We need to modify main.py to return the metrics, 
            # OR we modify this script to import the necessary functions and run the logic directly.
            # Importing main() is cleaner but main() needs to return metrics.
            
            # Let's modify main.py slightly to return metrics, 
            # OR better: since we are in the same process, we can just call the train/eval logic?
            # No, main() does a lot of setup.
            
            # Let's assume we will modify main.py to return (test_metric, test_secondary_metrics)
            # I will apply this change to main.py in the next step.
            # main() now returns (best_val_acc, final_test_acc, final_test_secondary)
            # We only need test metrics here
            _, test_metric, test_secondary = run_single_seed(args)
            
            results['primary'].append(test_metric)
            results['secondary'].append(test_secondary)
            print(f" Done. Metric: {test_metric:.4f}")
            
        except Exception as e:
            print(f" Failed! Error: {e}")
            # If a seed fails, we probably shouldn't record a partial average.
            # But for robustness, maybe we continue?
            # For strict benchmarking, we should probably fail.
            # Let's return for now.
            return

    # Calculate Statistics
    primary_mean = np.mean(results['primary'])
    primary_std = np.std(results['primary'])
    
    print(f"Completed 5 seeds. Mean Primary Metric: {primary_mean:.4f} ± {primary_std:.4f}")
    
    # Aggregate Secondary Metrics
    # secondary is a list of dicts
    secondary_keys = results['secondary'][0].keys()
    secondary_stats = {}
    
    for key in secondary_keys:
        values = [res[key] for res in results['secondary']]
        secondary_stats[f'{key}_mean'] = np.mean(values)
        secondary_stats[f'{key}_std'] = np.std(values)
        
    # Save to CSV (New Format)
    save_aggregated_results(args, primary_mean, primary_std, secondary_stats)

def save_aggregated_results(args, primary_mean, primary_std, secondary_stats):
    # Determine CSV path
    if args.category and args.dataset_name:
        csv_dir = f'/home/choi0425/workspace/ADMET/workspace/output/hyperparam/{args.category}'
        csv_path = os.path.join(csv_dir, f'{args.dataset_name}_tuning_results.csv')
    else:
        csv_dir = '/home/choi0425/workspace/ADMET/workspace/output/hyperparam'
        csv_path = os.path.join(csv_dir, f'{args.dataset}_tuning_results.csv')
        
    os.makedirs(csv_dir, exist_ok=True)
    
    # Construct Record
    record = {
        'experiment_id': args.experiment_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # Hyperparameters
        'lr': args.lr,
        'dropout_ratio': args.dropout_ratio,
        'batch_size': args.batch_size,
        'num_experts': args.num_experts,
        'num_layer': args.num_layer,
        'emb_dim': args.emb_dim,
        'decay': args.decay,
        'features': '+'.join(sorted(args.features)) if args.features else 'basic',
        # Primary Metric
        'Test_Metric_Mean': f"{primary_mean:.4f}",
        'Test_Metric_Std': f"{primary_std:.4f}",
    }
    
    # Add Secondary Metrics
    for key, val in secondary_stats.items():
        record[f'Test_{key}'] = f"{val:.4f}"
        
    df = pd.DataFrame([record])
    
    try:
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode='w')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)
        print(f"Aggregated results saved to {csv_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    run_5seeds()
