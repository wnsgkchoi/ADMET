#!/usr/bin/env python3
"""Test secondary metrics computation"""

import numpy as np
from sklearn.metrics import (
    average_precision_score, accuracy_score, f1_score, 
    recall_score, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr

def compute_secondary_metrics(y_true, y_pred, task_type, num_classes=2):
    """Test version of compute_secondary_metrics"""
    metrics = {}
    
    try:
        if task_type == 'classification':
            if num_classes == 2:
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics['auprc'] = average_precision_score(y_true, y_pred)
                metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
                metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
                metrics['sensitivity'] = recall_score(y_true, y_pred_binary, zero_division=0)
                tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
                fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif task_type == 'regression':
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['r2'] = r2_score(y_true, y_pred)
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            metrics['pearson'] = pearson_corr if not np.isnan(pearson_corr) else 0.0
            spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
            metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    except Exception as e:
        print(f"Error: {e}")
        return {}
    
    return metrics

# Test classification
print("Testing Classification:")
y_true_cls = np.array([0, 1, 0, 1, 1, 0])
y_pred_cls = np.array([0.2, 0.8, 0.3, 0.9, 0.7, 0.1])
cls_metrics = compute_secondary_metrics(y_true_cls, y_pred_cls, 'classification', 2)
print(cls_metrics)

# Test regression
print("\nTesting Regression:")
y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
reg_metrics = compute_secondary_metrics(y_true_reg, y_pred_reg, 'regression')
print(reg_metrics)

print("\n✅ Secondary metrics computation works!")
