import pandas as pd
import numpy as np
import os

def analyze_ames_results():
    csv_path = "workspace/output/hyperparam/Toxicity/AMES_progress.csv"
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total experiments: {len(df)}")
    
    # Filter out rows where test_metric might be missing or empty
    df = df.dropna(subset=['test_metric'])
    # Convert test_metric to numeric, coercing errors
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    df = df.dropna(subset=['test_metric'])

    print(f"Valid experiments: {len(df)}")
    
    if len(df) == 0:
        print("No valid data to analyze.")
        return

    # Hyperparameters to analyze
    hyperparams = ['lr', 'dropout_ratio', 'batch_size', 'num_experts', 'alpha', 'beta', 'num_layer', 'decay']
    
    summary_list = []

    print("\n" + "="*50)
    print("Average Test Metric by Hyperparameter")
    print("="*50)

    for param in hyperparams:
        if param in df.columns:
            print(f"\n--- {param} ---")
            stats = df.groupby(param)['test_metric'].agg(['mean', 'std', 'count', 'max']).sort_values('mean', ascending=False)
            print(stats)
            
            # Prepare for saving
            stats_reset = stats.reset_index()
            stats_reset['parameter'] = param
            stats_reset = stats_reset.rename(columns={param: 'value'})
            summary_list.append(stats_reset)
        else:
            print(f"\nWarning: {param} not found in columns")

    # Save summary to CSV
    if summary_list:
        summary_df = pd.concat(summary_list, ignore_index=True)
        # Reorder columns
        summary_df = summary_df[['parameter', 'value', 'mean', 'std', 'count', 'max']]
        summary_save_path = os.path.join(os.path.dirname(csv_path), "AMES_analysis_summary.csv")
        summary_df.to_csv(summary_save_path, index=False)
        print(f"\nSaved summary analysis to {summary_save_path}")

    print("\n" + "="*50)
    print("Top Configurations")
    print("="*50)
    
    # Sort by test_metric descending
    df_sorted = df.sort_values('test_metric', ascending=False)
    top_10 = df_sorted.head(10)
    
    cols_to_show = ['experiment_id'] + hyperparams + ['test_metric', 'val_metric', 'train_metric', 'num_epochs_trained']
    # Filter cols that exist
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    print(top_10[cols_to_show].to_string(index=False))
    
    # Save all sorted results to a new CSV for easy viewing
    top_save_path = os.path.join(os.path.dirname(csv_path), "AMES_sorted_results.csv")
    df_sorted[cols_to_show].to_csv(top_save_path, index=False)
    print(f"Saved sorted results to {top_save_path}")

if __name__ == "__main__":
    analyze_ames_results()
