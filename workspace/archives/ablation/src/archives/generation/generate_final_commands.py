"""
Generate training commands for final models with optimized structure for inference.

This script creates:
1. Training commands with best hyperparameters
2. Optimized directory structure for model loading
3. Model registry for unified prediction system
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_best_hyperparameters(config_path='configs/best_final_hyperparameters.json'):
    """Load best hyperparameters from JSON"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_dataset_config(config_path='configs/dataset_config.json'):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_command(dataset_name, dataset_info, hyperparams, use_pretrained=True):
    """
    Generate training command for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_info: Dataset metadata (category, task_type, etc.)
        hyperparams: Best hyperparameters dictionary
        use_pretrained: Whether to use pre-trained GIN model
    
    Returns:
        Training command string
    """
    category = dataset_info['category']
    
    # Base command
    cmd_parts = [
        'conda run -n ADMET python workspace/src/main.py',
        f'--category {category}',
        f'--dataset_name {dataset_name}',
        f'--experiment_id final_model',
        f'--lr {hyperparams["lr"]}',
        f'--dropout_ratio {hyperparams["dropout_ratio"]}',
        f'--batch_size {hyperparams["batch_size"]}',
        f'--num_experts {hyperparams["num_experts"]}',
        f'--alpha {hyperparams["alpha"]}',
        f'--beta {hyperparams["beta"]}',
        f'--min_temp {hyperparams["min_temp"]}',
        f'--decay {hyperparams["decay"]}',
        f'--num_layer {hyperparams["num_layer"]}',
        f'--emb_dim {hyperparams["emb_dim"]}',
        f'--gate_dim {hyperparams["gate_dim"]}',
        f'--split {hyperparams["split_type"]}',
        '--epochs 300',
        '--patience 50',
        '--use_combined_trainvalid',
        '--output_dir workspace/final_models'
    ]
    
    # Add pre-trained model if requested
    if use_pretrained:
        cmd_parts.append('--gin_pretrained_file workspace/src/pre-trained/supervised_contextpred.pth')
    
    return ' '.join(cmd_parts)


def create_model_registry(best_hyperparams, dataset_config):
    """
    Create a model registry JSON for unified prediction system.
    This registry contains all metadata needed to load and use models.
    
    Args:
        best_hyperparams: Best hyperparameters data
        dataset_config: Dataset configuration data
    
    Returns:
        Model registry dictionary
    """
    registry = {
        'version': '1.0',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Final trained models registry for unified ADMET prediction',
        'total_models': len(best_hyperparams['datasets']),
        'models': {}
    }
    
    # Group models by category
    categories = {}
    
    for dataset_name, data in sorted(best_hyperparams['datasets'].items()):
        category = data['category']
        task_type = data['task_type']
        metric = data['metric']
        
        # Model metadata for registry
        model_info = {
            'dataset_name': dataset_name,
            'category': category,
            'task_type': task_type,
            'metric': metric,
            'num_classes': data['num_classes'],
            'model_path': f'workspace/final_models/hyperparam/{category}/{dataset_name}/final_model/best_model.pt',
            'config_path': f'workspace/final_models/{category}/{dataset_name}/config.json',
            'performance_path': f'workspace/final_models/{category}/{dataset_name}/performance.json',
            'best_tuning_performance': data['best_test_metric'],
            'trained_with_combined_data': True,
            'hyperparameters': data['hyperparameters']
        }
        
        registry['models'][dataset_name] = model_info
        
        # Group by category for summary
        if category not in categories:
            categories[category] = []
        categories[category].append(dataset_name)
    
    # Add category summary
    registry['categories'] = {
        cat: {
            'count': len(datasets),
            'datasets': sorted(datasets)
        }
        for cat, datasets in sorted(categories.items())
    }
    
    return registry


def main():
    print("="*80)
    print("GENERATING FINAL MODEL TRAINING COMMANDS")
    print("="*80)
    
    # Load configurations
    print("\nLoading configurations...")
    best_hyperparams = load_best_hyperparameters()
    dataset_config = load_dataset_config()
    
    total_datasets = len(best_hyperparams['datasets'])
    print(f"  ✓ Loaded {total_datasets} dataset configurations")
    
    # Create output directory for commands
    commands_dir = Path('workspace/commands')
    commands_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate commands file
    commands_file = commands_dir / 'commands_final_model_training.txt'
    
    print(f"\nGenerating training commands...")
    print(f"  Output: {commands_file}")
    
    with open(commands_file, 'w') as f:
        # Write header
        f.write("# Final Model Training Commands\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total: {total_datasets} datasets\n")
        f.write("# Mode: Combined train+valid, test evaluation only\n")
        f.write("# Pre-trained: supervised_contextpred.pth\n")
        f.write("# Output: workspace/final_models/\n")
        f.write("#\n")
        f.write("# Model structure for inference:\n")
        f.write("#   workspace/final_models/hyperparam/{category}/{dataset_name}/final_model/best_model.pt\n")
        f.write("#\n")
        f.write("# Use with: simple_gpu_scheduler --gpus 0 1 2 3 < commands_final_model_training.txt\n")
        f.write("#\n\n")
        
        # Generate commands grouped by category
        for category in ['Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Toxicity']:
            category_datasets = [
                (name, data) for name, data in sorted(best_hyperparams['datasets'].items())
                if data['category'] == category
            ]
            
            if category_datasets:
                f.write(f"# {category} ({len(category_datasets)} datasets)\n")
                
                for dataset_name, data in category_datasets:
                    hyperparams = data['hyperparameters']
                    task_type = data['task_type']
                    metric = data['metric']
                    best_score = data['best_test_metric']
                    
                    # Add comment with metadata
                    f.write(f"# {dataset_name}: {task_type}, {metric}={best_score:.4f}\n")
                    
                    # Generate command
                    cmd = generate_command(dataset_name, data, hyperparams, use_pretrained=True)
                    f.write(f"{cmd}\n\n")
                
                f.write("\n")
    
    print(f"  ✓ Generated {total_datasets} training commands")
    
    # Create model registry
    print("\nCreating model registry...")
    registry = create_model_registry(best_hyperparams, dataset_config)
    
    registry_path = Path('workspace/final_models/model_registry.json')
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"  ✓ Model registry saved: {registry_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Commands file:  {commands_file}")
    print(f"Total commands: {total_datasets}")
    print(f"Model registry: {registry_path}")
    print("\nModels by category:")
    for category, info in sorted(registry['categories'].items()):
        print(f"  - {category}: {info['count']} models")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the generated commands:")
    print(f"   cat {commands_file}")
    print("\n2. Start training with GPU scheduler:")
    print(f"   simple_gpu_scheduler --gpus 0 1 2 3 < {commands_file}")
    print("\n3. Monitor progress:")
    print("   tail -f workspace/output/hyperparam/*/final_model.log")
    print("\n4. After training completes, collect results:")
    print("   python workspace/src/collect_final_results.py")
    print("="*80)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
