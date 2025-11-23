"""
Classification 데이터셋의 클래스 분포 분석
Train/Valid/Test split별 Y distribution 조사
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import sys
from pathlib import Path

# Add src root to path
src_root = Path(__file__).parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from loader import load_admet_dataset

# Load dataset configuration
config_path = 'configs/dataset_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print("=" * 100)
print("Classification Dataset Class Distribution Analysis")
print("=" * 100)

results = []

# Filter only classification datasets
classification_datasets = {
    name: info for name, info in config['datasets'].items()
    if info['task_type'] == 'classification'
}

print(f"\nTotal classification datasets: {len(classification_datasets)}")

for dataset_name in sorted(classification_datasets.keys()):
    dataset_info = classification_datasets[dataset_name]
    category = dataset_info['category']
    
    print(f"\n{'=' * 100}")
    print(f"Dataset: {dataset_name}")
    print(f"Category: {category}")
    print(f"{'=' * 100}")
    
    try:
        # Load dataset
        data_dict = load_admet_dataset(
            category=category,
            dataset_name=dataset_name,
            config_path=config_path
        )
        
        # Get data splits
        train_df = data_dict['train']
        val_df = data_dict['valid']
        test_df = data_dict['test']
        
        # Extract labels
        train_labels = train_df['Y'].tolist()
        val_labels = val_df['Y'].tolist()
        test_labels = test_df['Y'].tolist()
        all_labels = train_labels + val_labels + test_labels
        
        # Count distributions
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        test_counter = Counter(test_labels)
        all_counter = Counter(all_labels)
        
        # Total counts
        total_samples = len(all_labels)
        train_total = len(train_labels)
        val_total = len(val_labels)
        test_total = len(test_labels)
        
        # Get unique classes
        unique_classes = sorted(all_counter.keys())
        num_classes = len(unique_classes)
        
        # Calculate percentages for each class
        for cls in unique_classes:
            # Overall distribution
            overall_count = all_counter.get(cls, 0)
            overall_pct = (overall_count / total_samples * 100) if total_samples > 0 else 0
            
            # Train distribution
            train_count = train_counter.get(cls, 0)
            train_pct = (train_count / train_total * 100) if train_total > 0 else 0
            
            # Validation distribution
            val_count = val_counter.get(cls, 0)
            val_pct = (val_count / val_total * 100) if val_total > 0 else 0
            
            # Test distribution
            test_count = test_counter.get(cls, 0)
            test_pct = (test_count / test_total * 100) if test_total > 0 else 0
            
            results.append({
                'Dataset': dataset_name,
                'Category': category,
                'Num_Classes': num_classes,
                'Class': int(cls),
                'Total_Samples': total_samples,
                'Train_Total': train_total,
                'Valid_Total': val_total,
                'Test_Total': test_total,
                'Overall_Class_Count': overall_count,
                'Overall_Class_Pct': f"{overall_pct:.2f}",
                'Train_Class_Count': train_count,
                'Train_Class_Pct': f"{train_pct:.2f}",
                'Valid_Class_Count': val_count,
                'Valid_Class_Pct': f"{val_pct:.2f}",
                'Test_Class_Count': test_count,
                'Test_Class_Pct': f"{test_pct:.2f}",
            })
            
            print(f"\n  Class {int(cls)}:")
            print(f"    Overall: {overall_count:5d} ({overall_pct:6.2f}%)")
            print(f"    Train:   {train_count:5d} ({train_pct:6.2f}%)")
            print(f"    Valid:   {val_count:5d} ({val_pct:6.2f}%)")
            print(f"    Test:    {test_count:5d} ({test_pct:6.2f}%)")
        
        # Calculate imbalance ratio (majority class / minority class)
        class_counts = [all_counter.get(cls, 0) for cls in unique_classes]
        if len(class_counts) >= 2 and min(class_counts) > 0:
            imbalance_ratio = max(class_counts) / min(class_counts)
            print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        continue

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
output_csv = '/home/choi0425/workspace/ADMET/workspace/classification_class_distribution.csv'
df.to_csv(output_csv, index=False)

print(f"\n{'=' * 100}")
print(f"✅ Saved class distribution analysis to: {output_csv}")
print(f"{'=' * 100}")

# Summary statistics
print(f"\n{'=' * 100}")
print("Summary Statistics")
print(f"{'=' * 100}")

# Group by dataset and calculate imbalance metrics
summary_data = []
for dataset_name in df['Dataset'].unique():
    dataset_df = df[df['Dataset'] == dataset_name]
    
    # Get overall class percentages
    class_pcts = dataset_df['Overall_Class_Pct'].astype(float).values
    
    if len(class_pcts) >= 2:
        max_pct = max(class_pcts)
        min_pct = min(class_pcts)
        imbalance = max_pct / min_pct if min_pct > 0 else float('inf')
        
        category = dataset_df['Category'].iloc[0]
        num_classes = dataset_df['Num_Classes'].iloc[0]
        total_samples = dataset_df['Total_Samples'].iloc[0]
        
        summary_data.append({
            'Dataset': dataset_name,
            'Category': category,
            'Num_Classes': num_classes,
            'Total_Samples': total_samples,
            'Majority_Class_Pct': f"{max_pct:.2f}",
            'Minority_Class_Pct': f"{min_pct:.2f}",
            'Imbalance_Ratio': f"{imbalance:.2f}"
        })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Imbalance_Ratio', ascending=False)

# Save summary
summary_csv = '/home/choi0425/workspace/ADMET/workspace/classification_imbalance_summary.csv'
summary_df.to_csv(summary_csv, index=False)

print(f"\n✅ Saved imbalance summary to: {summary_csv}")
print("\nTop 10 Most Imbalanced Datasets:")
print(summary_df.head(10).to_string(index=False))

print(f"\n{'=' * 100}")
print("Analysis Complete!")
print(f"{'=' * 100}")
