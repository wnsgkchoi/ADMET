"""
Analyze data distribution for continuous (regression) ADMET datasets.

This script:
1. Identifies all continuous/regression datasets
2. Analyzes distribution across train/valid/test/all splits
3. Generates statistical summaries and visualizations
4. Checks for distribution shifts between splits
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import sys

# Disable plotting due to library issues
ENABLE_PLOTS = False

def load_dataset_config(config_path='configs/dataset_config.json'):
    """Load dataset configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def get_continuous_datasets(config):
    """Get list of continuous/regression datasets"""
    continuous = []
    for name, info in config['datasets'].items():
        if info['task_type'] == 'regression':
            continuous.append({
                'name': name,
                'category': info['category'],
                'path': info['path'],
                'metric': info['metric']
            })
    return continuous

def load_split_data(dataset_path, split_name):
    """Load data from a specific split"""
    csv_path = Path(dataset_path) / f"{split_name}.csv"
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    if 'Y' in df.columns:
        return df['Y'].values
    return None

def analyze_distribution(data, name):
    """Analyze distribution of a dataset"""
    if data is None or len(data) == 0:
        return None
    
    analysis = {
        'count': int(len(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data))
    }
    
    return analysis

def ks_test(data1, data2, name1, name2):
    """Perform Kolmogorov-Smirnov test between two distributions"""
    if data1 is None or data2 is None:
        return None
    
    statistic, pvalue = stats.ks_2samp(data1, data2)
    
    return {
        'comparison': f"{name1} vs {name2}",
        'ks_statistic': float(statistic),
        'p_value': float(pvalue),
        'significant': bool(pvalue < 0.05)
    }

def analyze_dataset_distribution(dataset_info):
    """Analyze distribution for a single dataset across all splits"""
    dataset_name = dataset_info['name']
    dataset_path = dataset_info['path']
    
    print(f"\nAnalyzing: {dataset_name}")
    print("-"*80)
    
    # Load all splits
    splits = {}
    for split_name in ['train', 'valid', 'test']:
        data = load_split_data(dataset_path, split_name)
        if data is not None:
            splits[split_name] = data
            print(f"  ✓ Loaded {split_name}: {len(data)} samples")
        else:
            print(f"  ✗ {split_name} not found")
    
    if not splits:
        print("  ⚠ No data loaded")
        return None
    
    # Combine all splits
    all_data = np.concatenate(list(splits.values()))
    splits['all'] = all_data
    
    # Analyze each split
    distributions = {}
    for split_name, data in splits.items():
        dist = analyze_distribution(data, split_name)
        if dist:
            distributions[split_name] = dist
    
    # Statistical tests between splits
    ks_tests = []
    if 'train' in splits and 'valid' in splits:
        ks_tests.append(ks_test(splits['train'], splits['valid'], 'train', 'valid'))
    if 'train' in splits and 'test' in splits:
        ks_tests.append(ks_test(splits['train'], splits['test'], 'train', 'test'))
    if 'valid' in splits and 'test' in splits:
        ks_tests.append(ks_test(splits['valid'], splits['test'], 'valid', 'test'))
    
    results = {
        'dataset': dataset_name,
        'category': dataset_info['category'],
        'distributions': distributions,
        'ks_tests': [t for t in ks_tests if t is not None]
    }
    
    # Print summary
    print(f"\n  Distribution Summary:")
    for split_name, dist in distributions.items():
        if split_name != 'all':
            print(f"    {split_name:6s}: n={dist['count']:4d} | "
                  f"mean={dist['mean']:7.3f} ± {dist['std']:6.3f} | "
                  f"range=[{dist['min']:7.3f}, {dist['max']:7.3f}]")
    
    if ks_tests:
        print(f"\n  Distribution Shift Tests (KS-test):")
        for test in ks_tests:
            if test:
                sig = "⚠ SIGNIFICANT" if test['significant'] else "✓ Similar"
                print(f"    {test['comparison']:20s}: p={test['p_value']:.4f} {sig}")
    
    return results

def create_distribution_plots(dataset_info, splits_data, output_dir='workspace/analysis/distributions'):
    """Create distribution visualization plots"""
    if not ENABLE_PLOTS:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset_info['name']
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} - Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Histograms for each split
    ax = axes[0, 0]
    for split_name in ['train', 'valid', 'test']:
        if split_name in splits_data:
            ax.hist(splits_data[split_name], alpha=0.5, label=split_name, bins=30)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    ax = axes[0, 1]
    data_to_plot = []
    labels = []
    for split_name in ['train', 'valid', 'test']:
        if split_name in splits_data:
            data_to_plot.append(splits_data[split_name])
            labels.append(split_name)
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Value')
    ax.set_title('Box Plot Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plots
    ax = axes[1, 0]
    if 'train' in splits_data and 'test' in splits_data:
        # Q-Q plot: train vs test
        sorted_train = np.sort(splits_data['train'])
        sorted_test = np.sort(splits_data['test'])
        
        # Interpolate to same length
        if len(sorted_train) != len(sorted_test):
            min_len = min(len(sorted_train), len(sorted_test))
            train_qq = np.interp(np.linspace(0, 1, min_len), 
                                 np.linspace(0, 1, len(sorted_train)), sorted_train)
            test_qq = np.interp(np.linspace(0, 1, min_len),
                               np.linspace(0, 1, len(sorted_test)), sorted_test)
        else:
            train_qq = sorted_train
            test_qq = sorted_test
        
        ax.scatter(train_qq, test_qq, alpha=0.5)
        min_val = min(train_qq.min(), test_qq.min())
        max_val = max(train_qq.max(), test_qq.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax.set_xlabel('Train Quantiles')
        ax.set_ylabel('Test Quantiles')
        ax.set_title('Q-Q Plot: Train vs Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Density plots
    ax = axes[1, 1]
    for split_name in ['train', 'valid', 'test']:
        if split_name in splits_data:
            data = splits_data[split_name]
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), label=split_name, linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{dataset_name}_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved plot: {plot_path}")

def main():
    print("="*80)
    print("CONTINUOUS DATASET DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load configuration
    config = load_dataset_config()
    continuous_datasets = get_continuous_datasets(config)
    
    print(f"\nFound {len(continuous_datasets)} continuous (regression) datasets")
    print("-"*80)
    
    # Analyze each dataset
    all_results = []
    
    for dataset_info in continuous_datasets:
        result = analyze_dataset_distribution(dataset_info)
        if result:
            all_results.append(result)
            
            # Create visualization
            dataset_path = dataset_info['path']
            splits_data = {}
            for split_name in ['train', 'valid', 'test']:
                data = load_split_data(dataset_path, split_name)
                if data is not None:
                    splits_data[split_name] = data
            
            if splits_data:
                create_distribution_plots(dataset_info, splits_data)
    
    # Save results to JSON
    output_path = Path('workspace/analysis/continuous_distribution_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Analyzed: {len(all_results)} datasets")
    print(f"Results saved: {output_path}")
    print(f"Plots saved: workspace/analysis/distributions/")
    
    # Create summary table
    summary_data = []
    for result in all_results:
        dataset = result['dataset']
        if 'all' in result['distributions']:
            dist = result['distributions']['all']
            
            # Check for significant distribution shifts
            significant_shifts = sum(1 for t in result['ks_tests'] if t['significant'])
            
            summary_data.append({
                'dataset': dataset,
                'category': result['category'],
                'total_samples': dist['count'],
                'mean': dist['mean'],
                'std': dist['std'],
                'min': dist['min'],
                'max': dist['max'],
                'skewness': dist['skewness'],
                'distribution_shifts': significant_shifts
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = Path('workspace/analysis/continuous_distribution_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table: {summary_csv}")
    
    print("="*80)


if __name__ == '__main__':
    main()
