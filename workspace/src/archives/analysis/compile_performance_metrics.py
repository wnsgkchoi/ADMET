"""
Calculate and organize final performance metrics for all ADMET models.

This script:
1. Collects final model performance from training results
2. Calculates AUROC for binary classification tasks
3. Calculates R² for continuous regression tasks  
4. Organizes metrics by category and task type
5. Generates comprehensive performance reports
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, roc_auc_score
import sys

def load_dataset_config(config_path='configs/dataset_config.json'):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model_registry(registry_path='workspace/final_models/model_registry.json'):
    """Load model registry"""
    with open(registry_path, 'r') as f:
        return json.load(f)

def extract_performance_metrics(dataset_name, category, task_type, metric_type):
    """
    Extract performance metrics from CSV and checkpoint files.
    
    Returns:
        Dictionary with all available metrics
    """
    # Load from progress CSV
    csv_path = Path(f'workspace/output/hyperparam/{category}/{dataset_name}_progress.csv')
    
    metrics = {
        'dataset': dataset_name,
        'category': category,
        'task_type': task_type,
        'primary_metric': metric_type
    }
    
    if not csv_path.exists():
        metrics['status'] = 'missing'
        return metrics
    
    try:
        df = pd.read_csv(csv_path)
        final_rows = df[df['experiment_id'] == 'final_model']
        
        if final_rows.empty:
            metrics['status'] = 'not_trained'
            return metrics
        
        final_row = final_rows.iloc[-1]
        
        # Extract primary metric
        if task_type == 'classification':
            metrics['AUROC'] = float(final_row['test_metric'])
            metrics['primary_value'] = metrics['AUROC']
        else:  # regression
            metrics['MAE'] = float(final_row['test_metric'])
            metrics['primary_value'] = metrics['MAE']
        
        # Extract secondary metrics
        secondary = {}
        
        if task_type == 'classification':
            # Classification metrics
            for col in ['test_auprc', 'test_accuracy', 'test_f1', 
                       'test_sensitivity', 'test_specificity']:
                if col in df.columns and pd.notna(final_row.get(col)):
                    key = col.replace('test_', '')
                    secondary[key] = float(final_row[col])
        else:
            # Regression metrics
            for col in ['test_rmse', 'test_r2', 'test_pearson', 'test_spearman']:
                if col in df.columns and pd.notna(final_row.get(col)):
                    key = col.replace('test_', '')
                    secondary[key] = float(final_row[col])
        
        metrics['secondary_metrics'] = secondary
        
        # Training info
        metrics['num_epochs'] = int(final_row['num_epochs_trained']) if pd.notna(final_row.get('num_epochs_trained')) else None
        metrics['early_stopped'] = bool(final_row['early_stopped']) if pd.notna(final_row.get('early_stopped')) else None
        
        metrics['status'] = 'success'
        
    except Exception as e:
        metrics['status'] = 'error'
        metrics['error'] = str(e)
    
    return metrics

def calculate_category_statistics(metrics_list, task_type):
    """Calculate statistics for a category"""
    if not metrics_list:
        return None
    
    successful = [m for m in metrics_list if m['status'] == 'success']
    
    if not successful:
        return None
    
    if task_type == 'classification':
        aurocs = [m['AUROC'] for m in successful if 'AUROC' in m]
        
        stats = {
            'count': len(successful),
            'mean_AUROC': np.mean(aurocs) if aurocs else None,
            'std_AUROC': np.std(aurocs) if aurocs else None,
            'min_AUROC': np.min(aurocs) if aurocs else None,
            'max_AUROC': np.max(aurocs) if aurocs else None,
            'median_AUROC': np.median(aurocs) if aurocs else None
        }
        
        # Best and worst models
        if aurocs:
            best_idx = np.argmax(aurocs)
            worst_idx = np.argmin(aurocs)
            stats['best_model'] = successful[best_idx]['dataset']
            stats['best_AUROC'] = aurocs[best_idx]
            stats['worst_model'] = successful[worst_idx]['dataset']
            stats['worst_AUROC'] = aurocs[worst_idx]
    
    else:  # regression
        maes = [m['MAE'] for m in successful if 'MAE' in m]
        r2s = [m['secondary_metrics'].get('r2') for m in successful 
               if 'secondary_metrics' in m and m['secondary_metrics'].get('r2') is not None]
        
        stats = {
            'count': len(successful),
            'mean_MAE': np.mean(maes) if maes else None,
            'std_MAE': np.std(maes) if maes else None,
            'min_MAE': np.min(maes) if maes else None,
            'max_MAE': np.max(maes) if maes else None,
            'median_MAE': np.median(maes) if maes else None
        }
        
        if r2s:
            stats['mean_R2'] = np.mean(r2s)
            stats['std_R2'] = np.std(r2s)
            stats['min_R2'] = np.min(r2s)
            stats['max_R2'] = np.max(r2s)
            stats['median_R2'] = np.median(r2s)
        
        # Best and worst models (lower MAE is better)
        if maes:
            best_idx = np.argmin(maes)
            worst_idx = np.argmax(maes)
            stats['best_model'] = successful[best_idx]['dataset']
            stats['best_MAE'] = maes[best_idx]
            stats['worst_model'] = successful[worst_idx]['dataset']
            stats['worst_MAE'] = maes[worst_idx]
    
    return stats

def main():
    print("="*80)
    print("PERFORMANCE METRICS COMPILATION")
    print("="*80)
    
    # Load configurations
    config = load_dataset_config()
    registry = load_model_registry()
    
    print(f"\nProcessing {len(config['datasets'])} datasets...")
    print("-"*80)
    
    # Collect all metrics
    all_metrics = []
    metrics_by_category = {}
    metrics_by_task_type = {'classification': [], 'regression': []}
    
    for dataset_name, dataset_info in sorted(config['datasets'].items()):
        category = dataset_info['category']
        task_type = dataset_info['task_type']
        metric_type = dataset_info['metric']
        
        print(f"\n{dataset_name} [{category}] ({task_type})")
        
        metrics = extract_performance_metrics(dataset_name, category, task_type, metric_type)
        all_metrics.append(metrics)
        
        if category not in metrics_by_category:
            metrics_by_category[category] = []
        metrics_by_category[category].append(metrics)
        
        metrics_by_task_type[task_type].append(metrics)
        
        # Print status
        if metrics['status'] == 'success':
            if task_type == 'classification':
                auroc = metrics.get('AUROC', 'N/A')
                print(f"  ✓ AUROC = {auroc:.4f}" if auroc != 'N/A' else f"  ✓ {auroc}")
                
                # Show secondary metrics
                if 'secondary_metrics' in metrics:
                    sec = metrics['secondary_metrics']
                    if 'auprc' in sec:
                        print(f"    AUPRC = {sec['auprc']:.4f}")
                    if 'accuracy' in sec:
                        print(f"    Accuracy = {sec['accuracy']:.4f}")
            else:
                mae = metrics.get('MAE', 'N/A')
                r2 = metrics.get('secondary_metrics', {}).get('r2', 'N/A')
                print(f"  ✓ MAE = {mae:.4f}" if mae != 'N/A' else f"  ✓ {mae}")
                if r2 != 'N/A':
                    print(f"    R² = {r2:.4f}")
        else:
            print(f"  ✗ Status: {metrics['status']}")
    
    # Calculate statistics by category
    print("\n" + "="*80)
    print("STATISTICS BY CATEGORY")
    print("="*80)
    
    category_stats = {}
    for category, cat_metrics in sorted(metrics_by_category.items()):
        # Separate by task type
        class_metrics = [m for m in cat_metrics if m['task_type'] == 'classification']
        regr_metrics = [m for m in cat_metrics if m['task_type'] == 'regression']
        
        category_stats[category] = {}
        
        print(f"\n{category}")
        print("-"*80)
        
        if class_metrics:
            stats = calculate_category_statistics(class_metrics, 'classification')
            if stats:
                category_stats[category]['classification'] = stats
                print(f"  Classification ({stats['count']} models):")
                print(f"    Mean AUROC: {stats['mean_AUROC']:.4f} ± {stats['std_AUROC']:.4f}")
                print(f"    Range: [{stats['min_AUROC']:.4f}, {stats['max_AUROC']:.4f}]")
                print(f"    Best: {stats['best_model']} (AUROC={stats['best_AUROC']:.4f})")
        
        if regr_metrics:
            stats = calculate_category_statistics(regr_metrics, 'regression')
            if stats:
                category_stats[category]['regression'] = stats
                print(f"  Regression ({stats['count']} models):")
                print(f"    Mean MAE: {stats['mean_MAE']:.4f} ± {stats['std_MAE']:.4f}")
                if stats.get('mean_R2') is not None:
                    print(f"    Mean R²: {stats['mean_R2']:.4f} ± {stats['std_R2']:.4f}")
                print(f"    Best: {stats['best_model']} (MAE={stats['best_MAE']:.4f})")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    for task_type in ['classification', 'regression']:
        stats = calculate_category_statistics(metrics_by_task_type[task_type], task_type)
        if stats:
            print(f"\n{task_type.title()} ({stats['count']} models):")
            if task_type == 'classification':
                print(f"  Mean AUROC: {stats['mean_AUROC']:.4f} ± {stats['std_AUROC']:.4f}")
                print(f"  Median AUROC: {stats['median_AUROC']:.4f}")
                print(f"  Range: [{stats['min_AUROC']:.4f}, {stats['max_AUROC']:.4f}]")
            else:
                print(f"  Mean MAE: {stats['mean_MAE']:.4f} ± {stats['std_MAE']:.4f}")
                if stats.get('mean_R2') is not None:
                    print(f"  Mean R²: {stats['mean_R2']:.4f} ± {stats['std_R2']:.4f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path('workspace/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics
    detailed_path = output_dir / 'detailed_performance_metrics.json'
    with open(detailed_path, 'w') as f:
        json.dump({
            'all_metrics': all_metrics,
            'category_stats': category_stats,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    print(f"  ✓ Detailed metrics: {detailed_path}")
    
    # Save performance table
    perf_data = []
    for m in all_metrics:
        if m['status'] == 'success':
            row = {
                'Dataset': m['dataset'],
                'Category': m['category'],
                'Task_Type': m['task_type'],
                'Primary_Metric': m['primary_metric']
            }
            
            if m['task_type'] == 'classification':
                row['AUROC'] = m.get('AUROC')
                if 'secondary_metrics' in m:
                    row['AUPRC'] = m['secondary_metrics'].get('auprc')
                    row['Accuracy'] = m['secondary_metrics'].get('accuracy')
                    row['F1'] = m['secondary_metrics'].get('f1')
            else:
                row['MAE'] = m.get('MAE')
                if 'secondary_metrics' in m:
                    row['RMSE'] = m['secondary_metrics'].get('rmse')
                    row['R²'] = m['secondary_metrics'].get('r2')
                    row['Pearson'] = m['secondary_metrics'].get('pearson')
            
            perf_data.append(row)
    
    perf_df = pd.DataFrame(perf_data)
    perf_csv = output_dir / 'performance_metrics_table.csv'
    perf_df.to_csv(perf_csv, index=False)
    print(f"  ✓ Performance table: {perf_csv}")
    
    # Save summary statistics
    summary_path = output_dir / 'performance_summary.json'
    summary = {
        'total_models': len(all_metrics),
        'successful': len([m for m in all_metrics if m['status'] == 'success']),
        'classification': calculate_category_statistics(metrics_by_task_type['classification'], 'classification'),
        'regression': calculate_category_statistics(metrics_by_task_type['regression'], 'regression'),
        'by_category': category_stats
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary statistics: {summary_path}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
