import optuna
import argparse
import os
import sys
import json
import numpy as np
import torch
import copy

# Add src to path to import main
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from main import main as run_single_seed
from main import load_args

def objective(trial):
    # 1. Load base arguments
    # We need to parse args again or construct them manually
    # Since we are running inside a script, we can't rely on sys.argv for the trial params
    # We should have passed the base args via command line to THIS script
    
    # However, load_args() parses sys.argv. 
    # We need to be careful not to confuse the parser with optuna args if we mix them.
    # Strategy: Parse known args for setup, and use trial to override hyperparameters.
    
    # Clone the base args
    args = copy.deepcopy(base_args_global)
    
    # 2. Suggest Hyperparameters
    # Search Space Definition
    args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    args.dropout_ratio = trial.suggest_float("dropout_ratio", 0.0, 0.5, step=0.1)
    args.decay = trial.suggest_float("decay", 1e-6, 0.0001, log=True)
    args.num_layer = trial.suggest_int("num_layer", 3, 7)
    args.num_experts = trial.suggest_int("num_experts", 3, 10)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    args.emb_dim = trial.suggest_categorical("emb_dim", [300]) # Fixed for now or search
    
    # Loss balance params
    args.alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    args.beta = trial.suggest_float("beta", 0.001, 0.1, log=True)

    args.min_temp = trial.suggest_float("min_temp", 0.01, 1, log=True)
    
    # 3. Run 5-Seed Average with Pruning
    # To support pruning, we need to report intermediate results from INSIDE the training loop.
    # But we are running 5 seeds. 
    # Strategy: 
    # - Run Seed 0 first. Pass the 'trial' object to main().
    # - If Seed 0 is pruned, the whole trial is pruned.
    # - If Seed 0 survives, run Seeds 1-4 (without pruning or with loose pruning).
    # - Return the average of 5 seeds.
    
    val_metrics = []
    
    # Seed 0: Enable Pruning
    print(f"  [Trial {trial.number}] Running Seed 0 with Pruning...")
    args.seed = 0
    args.quiet = True # Suppress output
    args.experiment_id = f"optuna_trial_{trial.number}_seed_0"
    
    try:
        # main() returns (best_val, test_metric, test_secondary)
        val_acc, _, _ = run_single_seed(args, trial=trial)
        val_metrics.append(val_acc)
    except optuna.exceptions.TrialPruned:
        print(f"  [Trial {trial.number}] Pruned at Seed 0.")
        raise # Re-raise to let Optuna handle it
    except Exception as e:
        print(f"  [Trial {trial.number}] Failed at Seed 0: {e}")
        # If it fails (e.g. OOM), we can return a bad value or fail
        return 0.0 # Assuming maximization
        
    # Seeds 1-4: No Pruning (or we could prune if we want to be very aggressive)
    # For stability, if Seed 0 is good, we run the rest.
    for seed in range(1, 5):
        args.seed = seed
        args.experiment_id = f"optuna_trial_{trial.number}_seed_{seed}"
        try:
            val_acc, _, _ = run_single_seed(args, trial=None) # No trial passed, so no pruning
            val_metrics.append(val_acc)
        except Exception as e:
            print(f"  [Trial {trial.number}] Failed at Seed {seed}: {e}")
            return 0.0

    # Return Mean Validation Metric
    mean_val = np.mean(val_metrics)
    print(f"  [Trial {trial.number}] Completed. Mean Val: {mean_val:.4f}")
    return mean_val

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == "__main__":
    # Parse arguments for the optimization script
    # We need to handle: --dataset_name, --features, --n_trials, --output_dir, etc.
    # And also pass through necessary args for main.py (like data paths)
    
    # We'll use a separate parser for the wrapper, and then manually set up the args object for main
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--features', type=str, nargs='+', default=['basic'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='workspace/output/optimization')
    parser.add_argument('--config_path', type=str, default='configs/dataset_config.json')
    parser.add_argument('--device_no', type=int, default=0)
    
    # Add other necessary args that main() expects but we might want to fix or pass through
    parser.add_argument('--gin_pretrained_file', type=str, default='workspace/src/pre-trained/supervised_contextpred.pth')
    
    # Parse known args, ignore others (or we could use parse_known_args)
    opt_args, unknown = parser.parse_known_args()
    
    # Setup Global Base Args for main()
    # We create a dummy args object or use load_args() and override
    # But load_args() parses sys.argv. We need to clean sys.argv or manually construct.
    
    # Let's manually construct a namespace or use load_args() with a clean sys.argv
    # Save original sys.argv
    original_argv = sys.argv
    
    # Mock sys.argv for load_args()
    # We need to pass minimal required args to load_args
    sys.argv = [sys.argv[0], 
                '--dataset_name', opt_args.dataset_name, 
                '--config_path', opt_args.config_path,
                '--device_no', str(opt_args.device_no),
                '--gin_pretrained_file', opt_args.gin_pretrained_file,
                '--no_save_model', # Don't save models during optimization
                '--quiet' # Default quiet
               ]
    
    # Add features
    sys.argv.append('--features')
    sys.argv.extend(opt_args.features)
    
    # Load base args
    base_args_global = load_args()
    
    # Restore sys.argv (optional, but good practice)
    sys.argv = original_argv
    
    # Ensure output directory exists
    feature_str = "_".join(sorted(opt_args.features))
    study_name = f"{opt_args.dataset_name}_{feature_str}"
    save_dir = os.path.join(opt_args.output_dir, study_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Optuna Study: {study_name}")
    print(f"  Dataset: {opt_args.dataset_name}")
    print(f"  Features: {opt_args.features}")
    print(f"  Trials: {opt_args.n_trials}")
    print(f"  Device: {opt_args.device_no}")
    
    # Create Study
    # Direction: maximize (AUROC) or minimize (MAE/RMSE)?
    # We need to check the metric type from config
    # But main.py returns 'metric'. 
    # For classification (AUROC), maximize. For regression (MAE/RMSE), minimize.
    # Let's check dataset config.
    
    with open(opt_args.config_path, 'r') as f:
        config = json.load(f)
    dataset_info = config['datasets'][opt_args.dataset_name]
    metric_type = dataset_info['metric']
    
    direction = 'maximize' if metric_type == 'AUROC' else 'minimize'
    print(f"  Metric: {metric_type} ({direction})")
    
    # Pruner: Hyperband is good
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=200, reduction_factor=3)
    
    study = optuna.create_study(direction=direction, study_name=study_name, pruner=pruner)
    
    try:
        study.optimize(objective, n_trials=opt_args.n_trials)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")
        
    # Save Best Params
    best_params_path = os.path.join(save_dir, "best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
    print(f"Best params saved to {best_params_path}")
    
    # Save Full Study (Optional, as dataframe)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, "study_results.csv"), index=False)
    
    # --- Final Evaluation ---
    print("\nRunning Final Evaluation with Best Hyperparameters...")
    
    # Update args with best params
    best_params = study.best_trial.params
    args = copy.deepcopy(base_args_global)
    for key, value in best_params.items():
        setattr(args, key, value)
    
    # Ensure we are not in quiet mode for final run? Or maybe just print summary.
    args.quiet = True 
    
    test_metrics = []
    test_secondary_metrics = []
    
    for seed in range(5):
        args.seed = seed
        args.experiment_id = f"final_best_{study_name}_seed_{seed}"
        # We might want to save the model this time?
        args.no_save_model = False 
        args.output_dir = os.path.join(save_dir, "final_models")
        
        print(f"  Running Final Seed {seed}...")
        try:
            # main returns: val_acc, test_acc, test_secondary
            val_acc, test_acc, test_secondary = run_single_seed(args)
            test_metrics.append(test_acc)
            test_secondary_metrics.append(test_secondary)
            print(f"    Seed {seed} Test Metric: {test_acc}")
        except Exception as e:
            print(f"    Seed {seed} Failed: {e}")
            
    if test_metrics:
        mean_test = np.mean(test_metrics)
        std_test = np.std(test_metrics)
        print(f"\nFinal Test Results ({opt_args.dataset_name}):")
        print(f"  Mean Test Metric: {mean_test:.4f} ± {std_test:.4f}")
        
        # Save final results
        final_results = {
            "dataset": opt_args.dataset_name,
            "best_params": best_params,
            "test_metrics": test_metrics,
            "mean_test_metric": mean_test,
            "std_test_metric": std_test,
            "secondary_metrics": test_secondary_metrics
        }
        
        with open(os.path.join(save_dir, "final_test_results.json"), 'w') as f:
            json.dump(final_results, f, indent=4, cls=NumpyEncoder)
            
        print(f"Final results saved to {os.path.join(save_dir, 'final_test_results.json')}")
