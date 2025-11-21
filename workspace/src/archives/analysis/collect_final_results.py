"""
Collect and analyze final model training results.

This script:
1. Collects results from all final model training runs
2. Compares performance with tuning results
3. Generates summary reports and visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

def load_model_registry(registry_path='workspace/final_models/model_registry.json'):
    """Load model registry"""
    with open(registry_path, 'r') as f:
        return json.load(f)

def load_dataset_config(config_path='configs/dataset_config.json'):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def collect_final_results(registry):
    """
    Collect results from all final model training runs.
    
    Args:
        registry: Model registry dictionary
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for dataset_name, model_info in sorted(registry['models'].items()):
        category = model_info['category']
        task_type = model_info['task_type']
        metric = model_info['metric']
        
        # Find CSV file with results
        csv_path = Path(f'workspace/output/hyperparam/{category}/{dataset_name}_progress.csv')
        
        if not csv_path.exists():
            print(f"  ⚠ No results CSV for {dataset_name}")
            results.append({
                'dataset': dataset_name,
                'category': category,
                'task_type': task_type,
                'metric': metric,
                'status': 'missing',
                'final_test_metric': None,
                'best_tuning_metric': model_info['best_tuning_performance'],
                'improvement': None
            })
            continue
        
        # Load CSV and find final_model row
        try:
            df = pd.read_csv(csv_path)
            final_rows = df[df['experiment_id'] == 'final_model']
            
            if final_rows.empty:
                print(f"  ⚠ No final_model results for {dataset_name}")
                results.append({
                    'dataset': dataset_name,
                    'category': category,
                    'task_type': task_type,
                    'metric': metric,
                    'status': 'not_run',
                    'final_test_metric': None,
                    'best_tuning_metric': model_info['best_tuning_performance'],
                    'improvement': None
                })
                continue
            
            # Get the last final_model entry (in case there are multiple)
            final_row = final_rows.iloc[-1]
            
            # Extract metrics
            final_test_metric = float(final_row['test_metric'])
            best_tuning_metric = model_info['best_tuning_performance']
            
            # Calculate improvement
            if task_type == 'classification':
                # Higher is better for AUROC
                improvement = final_test_metric - best_tuning_metric
                improvement_pct = (improvement / best_tuning_metric) * 100
            else:
                # Lower is better for MAE
                improvement = best_tuning_metric - final_test_metric
                improvement_pct = (improvement / best_tuning_metric) * 100
            
            # Extract secondary metrics if available
            secondary_metrics = {}
            for col in df.columns:
                if col.startswith('test_') and col != 'test_metric' and col != 'test_metric_std':
                    if pd.notna(final_row.get(col)):
                        secondary_metrics[col] = float(final_row[col])
            
            result = {
                'dataset': dataset_name,
                'category': category,
                'task_type': task_type,
                'metric': metric,
                'status': 'completed',
                'final_test_metric': final_test_metric,
                'best_tuning_metric': best_tuning_metric,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'num_epochs': int(final_row['num_epochs_trained']) if pd.notna(final_row.get('num_epochs_trained')) else None,
                'early_stopped': bool(final_row['early_stopped']) if pd.notna(final_row.get('early_stopped')) else None,
                'timestamp': str(final_row['timestamp']) if pd.notna(final_row.get('timestamp')) else None
            }
            
            # Add secondary metrics
            if secondary_metrics:
                result['secondary_metrics'] = secondary_metrics
            
            results.append(result)
            
            # Print status
            status_emoji = "✓" if improvement > 0 else ("=" if abs(improvement) < 0.001 else "↓")
            print(f"  {status_emoji} {dataset_name:35s} | {metric}={final_test_metric:.4f} | Δ={improvement:+.4f} ({improvement_pct:+.2f}%)")
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {e}")
            results.append({
                'dataset': dataset_name,
                'category': category,
                'task_type': task_type,
                'metric': metric,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def generate_summary_statistics(results):
    """
    Generate summary statistics from results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Summary statistics dictionary
    """
    completed = [r for r in results if r['status'] == 'completed']
    
    if not completed:
        return {'error': 'No completed results'}
    
    # Overall statistics
    improvements = [r['improvement'] for r in completed]
    improvement_pcts = [r['improvement_pct'] for r in completed]
    
    summary = {
        'total_datasets': len(results),
        'completed': len(completed),
        'missing': len([r for r in results if r['status'] == 'missing']),
        'not_run': len([r for r in results if r['status'] == 'not_run']),
        'errors': len([r for r in results if r['status'] == 'error']),
        'improvement': {
            'mean': np.mean(improvements),
            'std': np.std(improvements),
            'median': np.median(improvements),
            'min': np.min(improvements),
            'max': np.max(improvements),
            'mean_pct': np.mean(improvement_pcts),
            'positive_count': len([i for i in improvements if i > 0]),
            'negative_count': len([i for i in improvements if i < 0]),
            'neutral_count': len([i for i in improvements if abs(i) < 0.001])
        }
    }
    
    # By category
    categories = {}
    for result in completed:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    summary['by_category'] = {}
    for cat, cat_results in categories.items():
        cat_improvements = [r['improvement'] for r in cat_results]
        summary['by_category'][cat] = {
            'count': len(cat_results),
            'mean_improvement': np.mean(cat_improvements),
            'positive_count': len([i for i in cat_improvements if i > 0])
        }
    
    # By task type
    task_types = {}
    for result in completed:
        task = result['task_type']
        if task not in task_types:
            task_types[task] = []
        task_types[task].append(result)
    
    summary['by_task_type'] = {}
    for task, task_results in task_types.items():
        task_improvements = [r['improvement'] for r in task_results]
        summary['by_task_type'][task] = {
            'count': len(task_results),
            'mean_improvement': np.mean(task_improvements),
            'positive_count': len([i for i in task_improvements if i > 0])
        }
    
    # Top improvements
    sorted_by_improvement = sorted(completed, key=lambda x: x['improvement_pct'], reverse=True)
    summary['top_improvements'] = [
        {
            'dataset': r['dataset'],
            'improvement': r['improvement'],
            'improvement_pct': r['improvement_pct'],
            'metric': r['metric']
        }
        for r in sorted_by_improvement[:5]
    ]
    
    # Worst regressions
    summary['worst_regressions'] = [
        {
            'dataset': r['dataset'],
            'improvement': r['improvement'],
            'improvement_pct': r['improvement_pct'],
            'metric': r['metric']
        }
        for r in sorted_by_improvement[-5:]
    ]
    
    return summary


def main():
    print("="*80)
    print("COLLECTING FINAL MODEL TRAINING RESULTS")
    print("="*80)
    
    # Load registry
    print("\nLoading model registry...")
    registry = load_model_registry()
    print(f"  ✓ Found {registry['total_models']} models")
    
    # Collect results
    print("\nCollecting training results...")
    print("-"*80)
    results = collect_final_results(registry)
    
    # Generate summary
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    summary = generate_summary_statistics(results)
    
    # Print summary
    print(f"\nOverall Statistics:")
    print(f"  Total datasets:    {summary['total_datasets']}")
    print(f"  Completed:         {summary['completed']} ✓")
    print(f"  Not run yet:       {summary['not_run']}")
    print(f"  Missing:           {summary['missing']}")
    print(f"  Errors:            {summary['errors']}")
    
    if summary['completed'] > 0:
        print(f"\nImprovement Statistics:")
        print(f"  Mean improvement:  {summary['improvement']['mean']:+.4f} ({summary['improvement']['mean_pct']:+.2f}%)")
        print(f"  Median:            {summary['improvement']['median']:+.4f}")
        print(f"  Std dev:           {summary['improvement']['std']:.4f}")
        print(f"  Range:             [{summary['improvement']['min']:.4f}, {summary['improvement']['max']:.4f}]")
        print(f"  Improved:          {summary['improvement']['positive_count']} models")
        print(f"  Regressed:         {summary['improvement']['negative_count']} models")
        print(f"  Unchanged:         {summary['improvement']['neutral_count']} models")
        
        print(f"\nTop 5 Improvements:")
        for i, item in enumerate(summary['top_improvements'], 1):
            print(f"  {i}. {item['dataset']:30s} {item['improvement']:+.4f} ({item['improvement_pct']:+.2f}%)")
        
        if summary['improvement']['negative_count'] > 0:
            print(f"\nTop 5 Regressions:")
            for i, item in enumerate(reversed(summary['worst_regressions']), 1):
                print(f"  {i}. {item['dataset']:30s} {item['improvement']:+.4f} ({item['improvement_pct']:+.2f}%)")
        
        print(f"\nBy Category:")
        for cat, stats in sorted(summary['by_category'].items()):
            print(f"  {cat:15s}: {stats['count']:2d} models | "
                  f"Mean Δ={stats['mean_improvement']:+.4f} | "
                  f"{stats['positive_count']}/{stats['count']} improved")
        
        print(f"\nBy Task Type:")
        for task, stats in sorted(summary['by_task_type'].items()):
            print(f"  {task:15s}: {stats['count']:2d} models | "
                  f"Mean Δ={stats['mean_improvement']:+.4f} | "
                  f"{stats['positive_count']}/{stats['count']} improved")
    
    # Save results to files
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save detailed results CSV
    results_df = pd.DataFrame([
        {
            'dataset': r['dataset'],
            'category': r['category'],
            'task_type': r['task_type'],
            'metric': r['metric'],
            'status': r['status'],
            'final_test_metric': r.get('final_test_metric'),
            'best_tuning_metric': r.get('best_tuning_metric'),
            'improvement': r.get('improvement'),
            'improvement_pct': r.get('improvement_pct'),
            'num_epochs': r.get('num_epochs'),
            'early_stopped': r.get('early_stopped'),
            'timestamp': r.get('timestamp')
        }
        for r in results
    ])
    
    csv_path = Path('workspace/final_training_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Detailed results: {csv_path}")
    
    # Save summary JSON
    summary_path = Path('workspace/final_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary stats:    {summary_path}")
    
    # Save full results with secondary metrics
    full_results_path = Path('workspace/final_training_full_results.json')
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Full results:     {full_results_path}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    
    return summary['completed'] > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
