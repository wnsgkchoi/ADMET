import pandas as pd
import numpy as np
import json

# Load performance data
csv_path = '/home/choi0425/workspace/ADMET/workspace/enhanced_features_performance.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("Low Performance Dataset Identification")
print("=" * 80)

# Separate classification and regression datasets
classification_df = df[df['Task_Type'] == 'classification'].copy()
regression_df = df[df['Task_Type'] == 'regression'].copy()

print(f"\nTotal datasets: {len(df)}")
print(f"  - Classification: {len(classification_df)}")
print(f"  - Regression: {len(regression_df)}")

# Classification: AUROC < 75
print("\n" + "=" * 80)
print("Classification Datasets (AUROC < 75)")
print("=" * 80)

classification_low = classification_df[classification_df['Performance'] < 75].copy()
classification_low = classification_low.sort_values('Performance')

if len(classification_low) > 0:
    print(f"\nFound {len(classification_low)} low-performance classification datasets:\n")
    for idx, row in classification_low.iterrows():
        print(f"  {row['Dataset']:40s} | AUROC: {row['Performance']:6.2f} | Category: {row['Category']}")
else:
    print("\nNo low-performance classification datasets found.")

# Regression: Top 25% highest MAE
print("\n" + "=" * 80)
print("Regression Datasets (Top 25% MAE - Highest Error)")
print("=" * 80)

# Sort by performance (MAE) in descending order and take top 25%
regression_sorted = regression_df.sort_values('Performance', ascending=False)
top_25_count = max(1, int(len(regression_sorted) * 0.25))  # At least 1
regression_low = regression_sorted.head(top_25_count)

print(f"\nTop 25% ({top_25_count} out of {len(regression_df)}) regression datasets with highest MAE:\n")
for idx, row in regression_low.iterrows():
    print(f"  {row['Dataset']:40s} | MAE: {row['Performance']:7.4f} | Category: {row['Category']}")

# Combine all low-performance datasets
all_low_performance = []

for idx, row in classification_low.iterrows():
    all_low_performance.append({
        'dataset': row['Dataset'],
        'category': row['Category'],
        'task_type': 'classification',
        'metric_type': 'AUROC',
        'performance': row['Performance'],
        'reason': f'AUROC < 75 (current: {row["Performance"]:.2f})'
    })

for idx, row in regression_low.iterrows():
    all_low_performance.append({
        'dataset': row['Dataset'],
        'category': row['Category'],
        'task_type': 'regression',
        'metric_type': 'MAE',
        'performance': row['Performance'],
        'reason': f'Top 25% MAE (current: {row["Performance"]:.4f})'
    })

# Summary
print("\n" + "=" * 80)
print("Summary of Datasets Requiring Retuning")
print("=" * 80)
print(f"\nTotal datasets for retuning: {len(all_low_performance)}")
print(f"  - Classification (AUROC < 75): {len(classification_low)}")
print(f"  - Regression (Top 25% MAE): {len(regression_low)}")

# Save to JSON for next step
output_path = '/home/choi0425/workspace/ADMET/workspace/low_performance_datasets.json'
with open(output_path, 'w') as f:
    json.dump(all_low_performance, f, indent=2)

print(f"\n✅ Saved low-performance dataset list to: {output_path}")

# Display all datasets to retune
print("\n" + "=" * 80)
print("All Datasets to Retune:")
print("=" * 80)
for i, dataset_info in enumerate(all_low_performance, 1):
    print(f"\n{i}. {dataset_info['dataset']}")
    print(f"   Category: {dataset_info['category']}")
    print(f"   Task: {dataset_info['task_type']}")
    print(f"   Current {dataset_info['metric_type']}: {dataset_info['performance']:.4f}")
    print(f"   Reason: {dataset_info['reason']}")

print("\n" + "=" * 80)
