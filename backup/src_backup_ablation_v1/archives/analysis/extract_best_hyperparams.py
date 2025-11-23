"""
Extract best hyperparameters from tuning results for final model training.
Analyzes all progress CSV files in workspace/results/ and identifies the best
hyperparameter combination for each dataset based on test performance.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def load_dataset_config(config_path='configs/dataset_config.json'):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def find_best_experiment(csv_path, task_type, metric_name):
    """
    Find the best experiment from a progress CSV file.
    
    Args:
        csv_path: Path to the progress CSV file
        task_type: 'classification' or 'regression'
        metric_name: Primary metric name (AUROC, MAE, etc.)
    
    Returns:
        Dictionary with best hyperparameters and performance
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"WARNING: Empty CSV file: {csv_path}")
            return None
        
        # Convert test_metric to float, handling any string values
        df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
        
        # Remove rows with NaN test_metric
        df = df.dropna(subset=['test_metric'])
        
        if df.empty:
            print(f"WARNING: No valid test_metric values in: {csv_path}")
            return None
        
        # Find best experiment based on task type
        if task_type == 'classification':
            # Higher is better for AUROC
            best_idx = df['test_metric'].idxmax()
        else:
            # Lower is better for MAE/RMSE
            best_idx = df['test_metric'].idxmin()
        
        best_row = df.loc[best_idx]
        
        # Extract hyperparameters
        hyperparams = {
            'lr': float(best_row['lr']),
            'dropout_ratio': float(best_row['dropout_ratio']),
            'batch_size': int(best_row['batch_size']),
            'num_experts': int(best_row['num_experts']),
            'alpha': float(best_row['alpha']),
            'beta': float(best_row['beta']),
            'min_temp': float(best_row['min_temp']),
            'decay': float(best_row['decay']),
            'num_layer': int(best_row['num_layer']),
            'emb_dim': int(best_row['emb_dim']),
            'gate_dim': int(best_row['gate_dim']),
            'split_type': str(best_row['split_type'])
        }
        
        # Extract performance metrics
        performance = {
            'best_test_metric': float(best_row['test_metric']),
            'experiment_id': str(best_row['experiment_id']),
            'num_epochs_trained': int(best_row['num_epochs_trained']) if pd.notna(best_row.get('num_epochs_trained')) else None,
            'early_stopped': bool(best_row['early_stopped']) if pd.notna(best_row.get('early_stopped')) else None
        }
        
        # Extract secondary metrics if available
        secondary_metrics = {}
        for metric in ['test_auprc', 'test_accuracy', 'test_f1', 'test_sensitivity', 'test_specificity',
                      'test_rmse', 'test_r2', 'test_pearson', 'test_spearman']:
            if metric in df.columns and pd.notna(best_row.get(metric)):
                secondary_metrics[metric] = float(best_row[metric])
        
        if secondary_metrics:
            performance['secondary_metrics'] = secondary_metrics
        
        return {
            'hyperparameters': hyperparams,
            'performance': performance
        }
        
    except Exception as e:
        print(f"ERROR processing {csv_path}: {e}")
        return None


def main():
    print("="*80)
    print("EXTRACTING BEST HYPERPARAMETERS FROM TUNING RESULTS")
    print("="*80)
    
    # Load dataset configuration
    config = load_dataset_config()
    datasets = config['datasets']
    
    # Results directory
    results_dir = Path('workspace/results')
    
    # Storage for best hyperparameters
    best_hyperparams = {
        'version': '1.0',
        'created': '2025-11-18',
        'description': 'Best hyperparameters extracted from retune experiments for final model training',
        'total_datasets': 0,
        'datasets': {}
    }
    
    # Track statistics
    stats = {
        'total': len(datasets),
        'found': 0,
        'missing': 0,
        'failed': 0
    }
    missing_datasets = []
    failed_datasets = []
    
    print(f"\nProcessing {len(datasets)} datasets...")
    print("-"*80)
    
    # Process each dataset
    for dataset_name, dataset_info in sorted(datasets.items()):
        category = dataset_info['category']
        task_type = dataset_info['task_type']
        metric = dataset_info['metric']
        
        # Find corresponding CSV file
        csv_path = results_dir / f"{dataset_name}_progress.csv"
        
        print(f"\n[{category}] {dataset_name} ({task_type}, {metric})")
        
        if not csv_path.exists():
            print(f"  ⚠️  CSV not found: {csv_path}")
            stats['missing'] += 1
            missing_datasets.append(dataset_name)
            continue
        
        # Find best experiment
        best_exp = find_best_experiment(csv_path, task_type, metric)
        
        if best_exp is None:
            print(f"  ❌ Failed to extract best hyperparameters")
            stats['failed'] += 1
            failed_datasets.append(dataset_name)
            continue
        
        # Store results
        best_hyperparams['datasets'][dataset_name] = {
            'category': category,
            'task_type': task_type,
            'metric': metric,
            'num_classes': dataset_info['num_classes'],
            'best_experiment_id': best_exp['performance']['experiment_id'],
            'best_test_metric': best_exp['performance']['best_test_metric'],
            'hyperparameters': best_exp['hyperparameters'],
            'performance': best_exp['performance']
        }
        
        stats['found'] += 1
        print(f"  ✓ Best: {best_exp['performance']['experiment_id']} | "
              f"{metric}={best_exp['performance']['best_test_metric']:.4f}")
    
    # Update total
    best_hyperparams['total_datasets'] = stats['found']
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total datasets:     {stats['total']}")
    print(f"Successfully found: {stats['found']} ✓")
    print(f"Missing CSVs:       {stats['missing']}")
    print(f"Failed to parse:    {stats['failed']}")
    
    if missing_datasets:
        print(f"\nMissing datasets: {', '.join(missing_datasets)}")
    
    if failed_datasets:
        print(f"\nFailed datasets: {', '.join(failed_datasets)}")
    
    # Save to JSON
    output_path = Path('configs/best_final_hyperparameters.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    
    print(f"\n✅ Saved best hyperparameters to: {output_path}")
    print(f"   ({stats['found']} datasets)")
    
    # Also create a summary CSV for easy viewing
    summary_data = []
    for dataset_name, data in sorted(best_hyperparams['datasets'].items()):
        summary_data.append({
            'dataset': dataset_name,
            'category': data['category'],
            'task_type': data['task_type'],
            'metric': data['metric'],
            'best_experiment': data['best_experiment_id'],
            'best_test_metric': data['best_test_metric'],
            'lr': data['hyperparameters']['lr'],
            'dropout': data['hyperparameters']['dropout_ratio'],
            'batch_size': data['hyperparameters']['batch_size'],
            'num_experts': data['hyperparameters']['num_experts'],
            'alpha': data['hyperparameters']['alpha'],
            'beta': data['hyperparameters']['beta'],
            'min_temp': data['hyperparameters']['min_temp'],
            'decay': data['hyperparameters']['decay']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = Path('workspace/best_hyperparameters_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary to: {summary_csv_path}")
    
    print("\n" + "="*80)
    
    return stats['found'] == stats['total']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
