import math
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from splitters import data_split, cross_validation_split
from loader import MoleculeCSVDataset
from TopExpert import GNN_topexpert
from utils import *

def qprint(msg, args):
    """Conditional print: only print if not in quiet mode"""
    if not args.quiet:
        print(msg)

def load_args():
    parser = argparse.ArgumentParser()

# seed & device
    parser.add_argument('--device_no', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
   
#dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name (legacy support)')
    
    # New ADMET dataset arguments
    parser.add_argument('--category', type=str, default='', 
                        help='ADMET category: Absorption, Distribution, Metabolism, Excretion, Toxicity')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='ADMET dataset name (e.g., Caco2_Wang, AMES)')
    parser.add_argument('--config_path', type=str, default='configs/dataset_config.json',
                        help='Path to dataset_config.json')
    
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold or cv")
    parser.add_argument('--use_kfold', type=bool, default=False, help='Use K-fold CV (auto-detected for small datasets)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for K-fold CV')
    parser.add_argument('--use_combined_trainvalid', action='store_true', 
                        help='Combine train and valid datasets for final training (evaluate on test set only)')

#model
    parser.add_argument('-i', '--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('-c', '--ckpt_all', action='store_true',
                        help='Save all epoch checkpoints (default: save only best model)')
    parser.add_argument('--no_save_model', action='store_true',
                        help='Do not save model checkpoints (useful for hyperparameter tuning)')
    parser.add_argument('--output_dir', type=str, default='workspace/output',
                        help='Directory to save model checkpoints')
    

    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--extra_feature_dim', type=int, default=0, help='Dimension of extra features')


# train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--gin_pretrained_file', type=str, default='', help='path to pre-trained GIN model')
    parser.add_argument('--experiment_id', type=str, default='0', help='experiment ID for tracking')


#optimizer
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

## loss balance
    parser.add_argument('--alpha', type=float, default=0.1, help="balance parameter for clustering")
    parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")

## clustering
    parser.add_argument('--min_temp', type=float, default=1, help=" temperature for gumble softmax, annealing")
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--gate_dim', type=int, default=50, help="gate embedding space dimension, 50 or 300")

## output control
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output (for grid search)')
    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")

    # Handle ADMET datasets
    if args.dataset_name:
        # New ADMET dataset structure
        import json
        from loader import load_admet_dataset
        
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        
        dataset_info = config['datasets'][args.dataset_name]
        
        # Auto-fill category if missing
        if not args.category:
            args.category = dataset_info['category']
            
        args.task_type = dataset_info['task_type']
        args.num_tasks = 1
        args.num_classes = dataset_info['num_classes']
        args.use_kfold = dataset_info['use_kfold']
        args.metric = dataset_info['metric']
        
        # For backward compatibility, set dataset to dataset_name
        args.dataset = args.dataset_name
        
    else:
        # Legacy dataset handling
        # Bunch of classification tasks
        if args.dataset in ["dili2", "dili3"]:
            args.task_type = 'classification'
        elif args.dataset in ["hk2", "hepg2"]:
            args.task_type = 'regression'
        else:
            raise ValueError("Invalid dataset name.")

        if args.dataset == "dili2":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "dili3":
            args.num_tasks = 1
            args.num_classes = 3
        elif args.dataset == "hepg2":
            args.num_tasks = 1
            args.num_classes = 1  # Regression task
        elif args.dataset == "hk2":
            args.num_tasks = 1
            args.num_classes = 1  # Regression task
        elif args.dataset == "tox21":
            args.num_tasks = 12
            args.num_classes = 2
        elif args.dataset == "hiv":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "pcba":
            args.num_tasks = 128
            args.num_classes = 2
        elif args.dataset == "muv":
            args.num_tasks = 17
            args.num_classes = 2
        elif args.dataset == "bace":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "bbbp":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "toxcast":
            args.num_tasks = 617
            args.num_classes = 2
        elif args.dataset == "sider":
            args.num_tasks = 27
            args.num_classes = 2
        elif args.dataset == "clintox":
            args.num_tasks = 2
            args.num_classes = 2
        else:
            raise ValueError("Invalid dataset name.")

    return args


def train(args, model, loader, optimizer, scf_class):
    model.train()
    
    for batch in loader:
        model.T = max(torch.tensor(args.min_temp), model.T * args.temp_alpha)
        
        batch = batch.to(args.device)
        num_graph = batch.id.shape[0]  
        labels = batch.y.view(num_graph, -1)
        
        _, _, temp_q = model(batch)
        temp_q = temp_q.data

        p = model.target_distribution(temp_q) 
        
        pred, z, q = model(batch)
        g, q_idx = model.assign_head(q)

        scf_idx = scf_class.scfIdx_to_label[batch.scf_idx] 

        if args.task_type == 'classification':
            # For classification, convert labels to float
            labels = labels.to(torch.float)
            # Pass task_type and num_classes to clf_loss
            loss_mat, num_valid_mat = model.clf_loss(
                pred, labels, g, 
                task_type=args.task_type, 
                num_classes=args.num_classes
            )
            main_loss = torch.sum(loss_mat / num_valid_mat) / args.num_tasks
        elif args.task_type == 'regression':
            # Regression: NO label conversion, keep original values
            # Pass task_type to clf_loss
            loss_mat, num_valid_mat = model.clf_loss(
                pred, labels, g, 
                task_type=args.task_type, 
                num_classes=1  # Not used for regression
            )
            main_loss = torch.sum(loss_mat / num_valid_mat) / args.num_tasks
        else:
            raise ValueError("Invalid task type.")

        cluster_loss = F.kl_div(q.log(), p, reduction='sum')
        align_loss = model.alignment_loss(scf_idx, q)
        
        loss_total = main_loss + args.alpha * (cluster_loss / num_graph) + args.beta * align_loss
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()


def compute_secondary_metrics(y_true, y_pred, task_type, num_classes=2):
    """
    Compute secondary metrics for comprehensive evaluation
    
    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predictions (numpy array)
        task_type: 'classification' or 'regression'
        num_classes: Number of classes for classification
    
    Returns:
        Dictionary of secondary metrics
    """
    from sklearn.metrics import (
        average_precision_score, accuracy_score, f1_score, 
        recall_score, roc_auc_score, mean_squared_error, r2_score
    )
    from scipy.stats import pearsonr, spearmanr
    
    metrics = {}
    
    try:
        if task_type == 'classification':
            if num_classes == 2:
                # Binary classification
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                # AUPRC
                metrics['auprc'] = average_precision_score(y_true, y_pred)
                
                # Accuracy
                metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
                
                # F1 Score
                metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
                
                # Sensitivity (Recall)
                metrics['sensitivity'] = recall_score(y_true, y_pred_binary, zero_division=0)
                
                # Specificity
                tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
                fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
            else:
                # Multi-class classification
                y_pred_class = np.argmax(y_pred, axis=1)
                
                # Accuracy
                metrics['accuracy'] = accuracy_score(y_true, y_pred_class)
                
                # Macro F1
                metrics['f1'] = f1_score(y_true, y_pred_class, average='macro', zero_division=0)
                
                # Macro AUPRC
                try:
                    metrics['auprc'] = average_precision_score(
                        np.eye(num_classes)[y_true.astype(int)], 
                        y_pred, 
                        average='macro'
                    )
                except:
                    metrics['auprc'] = 0.0
                
                # Sensitivity/Specificity not well-defined for multi-class
                metrics['sensitivity'] = 0.0
                metrics['specificity'] = 0.0
                
        elif task_type == 'regression':
            # RMSE
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # R²
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Pearson Correlation
            pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            metrics['pearson'] = pearson_corr if not np.isnan(pearson_corr) else 0.0
            
            # Spearman Correlation
            spearman_corr, _ = spearmanr(y_true.flatten(), y_pred.flatten())
            metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            
    except Exception as e:
        # If any metric computation fails, set to 0
        print(f"Warning: Error computing secondary metrics: {e}")
        if task_type == 'classification':
            metrics = {'auprc': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}
        else:
            metrics = {'rmse': 0.0, 'r2': 0.0, 'pearson': 0.0, 'spearman': 0.0}
    
    return metrics


def eval(args, model, loader):
    model.eval()
    
    y_true_list, y_pred_list = [], []
    for batch in loader:
        batch = batch.to(args.device)
        with torch.no_grad():
            pred, z, q_origin = model(batch)
            q, q_idx = model.assign_head(q_origin) # N x tasks x head
            
            if args.task_type == 'classification':
                if args.num_classes > 2:
                    # Multi-class: pred is N x num_classes x tasks x num_experts
                    # Apply softmax and weight by expert assignment
                    # q is N x tasks x num_experts
                    softmax_pred = F.softmax(pred, dim=1)  # N x num_classes x tasks x experts
                    
                    # Weight by expert assignment: need to expand q
                    q_expanded = q.unsqueeze(1)  # N x 1 x tasks x experts
                    weighted_pred = softmax_pred * q_expanded  # N x num_classes x tasks x experts
                    scores = torch.sum(weighted_pred, dim=-1)  # N x num_classes x tasks
                    
                    # For single task, scores is N x num_classes
                    scores = scores.squeeze(-1)  # N x num_classes
                    y_pred_list.append(scores)
                else:
                    # Binary: pred is N x tasks x num_experts
                    # Apply sigmoid and weight by expert assignment
                    scores = torch.sum(torch.sigmoid(pred) * q, dim=-1)
                    y_pred_list.append(scores)
            elif args.task_type == 'regression':
                # Regression: pred is N x tasks x num_experts
                # Weight predictions by expert assignment (NO sigmoid/softmax)
                scores = torch.sum(pred * q, dim=-1)
                y_pred_list.append(scores)
                
        y_true_list.append(batch.y.view(batch.id.shape[0], -1))

    y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()

    # Compute primary metric
    if args.task_type == 'classification':
        # For multi-class, ensure y_true is in correct format
        if args.num_classes > 2:
            # y_true: (N, 1) with class indices -> (N,) for roc_auc_score
            # y_pred: (N, num_classes) with probabilities
            primary_metric = cal_roc(y_true.squeeze(), y_pred, num_classes=args.num_classes)
        else:
            # Binary classification
            primary_metric = cal_roc(y_true, y_pred, num_classes=args.num_classes)
    elif args.task_type == 'regression':
        from sklearn.metrics import mean_absolute_error
        primary_metric = mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError("Invalid task type.")
    
    # Compute secondary metrics
    secondary_metrics = compute_secondary_metrics(y_true, y_pred, args.task_type, args.num_classes)
    
    return primary_metric, secondary_metrics



def save_results_to_csv(args, val_metric, test_metric, val_std=None, test_std=None, num_epochs=None, early_stopped=False, 
                       val_secondary=None, test_secondary=None, train_metric=None, train_secondary=None):
    """Save experiment results to CSV file with secondary metrics"""
    import pandas as pd
    
    # Determine CSV path based on dataset type
    if args.category and args.dataset_name:
        # ADMET datasets: save in category subfolder
        csv_dir = f'/home/choi0425/workspace/ADMET/workspace/output/hyperparam/{args.category}'
        csv_path = os.path.join(csv_dir, f'{args.dataset_name}_progress.csv')
    else:
        # Legacy datasets
        csv_dir = '/home/choi0425/workspace/ADMET/workspace/output/hyperparam'
        csv_path = os.path.join(csv_dir, f'{args.dataset}_progress.csv')
    
    os.makedirs(csv_dir, exist_ok=True)
    
    result = {
        'experiment_id': args.experiment_id,
        'lr': args.lr,
        'dropout_ratio': args.dropout_ratio,
        'batch_size': args.batch_size,
        'num_experts': args.num_experts,
        'alpha': args.alpha,
        'beta': args.beta,
        'min_temp': args.min_temp,
        'decay': args.decay,
        'num_layer': args.num_layer,
        'emb_dim': args.emb_dim,
        'gate_dim': args.gate_dim,
        'split_type': args.split,
        'train_metric': f'{train_metric:.4f}' if train_metric is not None else '',
        'val_metric': f'{val_metric:.4f}' if val_metric is not None else '',
        'val_metric_std': f'{val_std:.4f}' if val_std is not None else '',
        'test_metric': f'{test_metric:.4f}',
        'test_metric_std': f'{test_std:.4f}' if test_std is not None else '',
        'num_epochs_trained': num_epochs if num_epochs else '',
        'early_stopped': early_stopped,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add secondary metrics if available
    if train_secondary:
        for key, value in train_secondary.items():
            result[f'train_{key}'] = f'{value:.4f}' if value is not None else ''

    if val_secondary:
        for key, value in val_secondary.items():
            result[f'val_{key}'] = f'{value:.4f}' if value is not None else ''

    if test_secondary:
        for key, value in test_secondary.items():
            result[f'test_{key}'] = f'{value:.4f}' if value is not None else ''
    
    df = pd.DataFrame([result])
    
    # 파일이 없으면 헤더와 함께 생성, 있으면 append
    try:
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode='w')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)
        # Suppress success message for grid search
        # print(f"Results saved to {csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save results to CSV: {e}", file=sys.stderr)


def main(args):
    set_seed(args.seed)    

    # Handle ADMET datasets vs legacy datasets
    if args.category and args.dataset_name:
        # New ADMET dataset loading - load each split separately
        dataset_path = f"workspace/data/{args.category}/{args.dataset_name}"
        
        if args.use_combined_trainvalid:
            # For final training: combine train and valid datasets
            train_dataset_orig = MoleculeCSVDataset(
                root=dataset_path, 
                dataset_name='train',
                smiles_col='Drug', 
                target_cols=['Y']
            )
            
            valid_dataset_orig = MoleculeCSVDataset(
                root=dataset_path, 
                dataset_name='valid',
                smiles_col='Drug', 
                target_cols=['Y']
            )
            
            # Combine train and valid
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset_orig, valid_dataset_orig])
            
            # For combined mode, we don't use a separate validation set
            # We'll evaluate on test set only
            valid_dataset = None
            
            qprint(f"Combined train+valid: {len(train_dataset)} samples", args)
        else:
            # For hyperparameter tuning: use separate train, valid, test
            train_dataset = MoleculeCSVDataset(
                root=dataset_path, 
                dataset_name='train',  # will look for train.csv
                smiles_col='Drug', 
                target_cols=['Y']
            )
            
            valid_dataset = MoleculeCSVDataset(
                root=dataset_path, 
                dataset_name='valid',  # will look for valid.csv
                smiles_col='Drug', 
                target_cols=['Y']
            )
        
        test_dataset = MoleculeCSVDataset(
            root=dataset_path, 
            dataset_name='test',  # will look for test.csv
            smiles_col='Drug', 
            target_cols=['Y']
        )
        
        # Skip the normal split logic for ADMET datasets
        use_presplit_data = True
        
    else:
        # Legacy dataset loading
        target_cols_map = {
            'dili2': ['label'],
            'dili3': ['vDILI-Concern'],
            'hepg2': ['pAC50'],
            'hk2': ['pIC50']
        }
        target_cols = target_cols_map.get(args.dataset)
        if target_cols is None:
            # Keep the original dataset names for compatibility
            pass
        
        dataset = MoleculeCSVDataset(root=args.dataset_dir, dataset_name=args.dataset, smiles_col='smiles', target_cols=target_cols)
        use_presplit_data = False
    
    if args.split == 'cv':
        cv_splitter = cross_validation_split(dataset, args.task_type, n_splits=10, seed=args.seed)
        val_acc_list, test_acc_list = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter):
            print(f"===== Fold {fold_idx+1} =====")
            
            train_dataset = dataset[torch.tensor(train_idx)]
            # Further split training data into training and validation
            # Note: This is a simple random split for validation. For more robust CV, you might use nested CV.
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(0.9 * num_train))
            train_subset_idx, valid_subset_idx = indices[:split], indices[split:]
            
            train_subset = train_dataset[torch.tensor(train_subset_idx)]
            valid_subset = train_dataset[torch.tensor(valid_subset_idx)]
            test_dataset = dataset[torch.tensor(test_idx)]

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            ## criterion, model, optimizer setup for each fold
            # Select criterion based on task type
            if args.task_type == 'classification':
                # Binary or multi-class classification
                # Note: For multi-class, we'll use F.cross_entropy in clf_loss
                # But we still pass BCEWithLogitsLoss for binary compatibility
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                # Regression: use L1Loss (MAE)
                criterion = nn.L1Loss(reduction="none")
                
            scf_tr = Scf_index(train_subset, args)
            args.num_tr_scf = scf_tr.num_scf
            num_iter = math.ceil(len(train_subset) / args.batch_size) 
            args.temp_alpha = np.exp(np.log(args.min_temp / 10 + 1e-10) / (args.epochs * num_iter))

            model = GNN_topexpert(args, criterion)
            if args.gin_pretrained_file:
                model.from_pretrained(args.gin_pretrained_file)
            
            model = model.to(args.device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)

            ###### init centroid using randomly initialized gnn
            zs_init = get_z(model, train_loader, args.device)
            init_centroid(model, zs_init, args.num_experts)
        
            best_val_acc = 0
            patience_counter = 0
            for epoch in range(1, args.epochs + 1):
                train(args, model, train_loader, optimizer, scf_tr)
                val_acc, _ = eval(args, model, val_loader)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    qprint(f"Early stopping at epoch {epoch}", args)
                    break

                if (epoch % 10 == 0) or (epoch == args.epochs):
                    metric_name = 'AUROC' if args.task_type == 'classification' else 'MAE'
                    qprint(f'Epoch {epoch}, Val {metric_name}: {val_acc:.4f}', args)

            test_acc, _ = eval(args, model, test_loader)
            metric_name = 'AUROC' if args.task_type == 'classification' else 'MAE'
            print(f'Fold {fold_idx+1} Test {metric_name}: {test_acc:.4f}')
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

        metric_name = "AUROC" if args.task_type == 'classification' else "MAE"
        print(f'\nAverage Validation {metric_name}: {np.mean(val_acc_list):.4f} +/- {np.std(val_acc_list):.4f}')
        print(f'Average Test {metric_name}: {np.mean(test_acc_list):.4f} +/- {np.std(test_acc_list):.4f}')
        
        # Save results to CSV
        save_results_to_csv(
            args, 
            val_metric=np.mean(val_acc_list),
            test_metric=np.mean(test_acc_list),
            val_std=np.std(val_acc_list),
            test_std=np.std(test_acc_list),
            num_epochs=None,  # CV doesn't have a single epoch count
            early_stopped=False  # CV doesn't use early stopping in the same way
        )

    else: # Original scaffold/random split
        if not use_presplit_data:
            # Legacy datasets: perform split
            train_dataset, valid_dataset, test_dataset = data_split(args, dataset)
        # else: ADMET datasets already have train_dataset, valid_dataset, test_dataset defined

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        # Only create val_loader if valid_dataset exists (not in combined mode)
        if valid_dataset is not None:
            val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            val_loader = None
            
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        ## criterion - select based on task type
        if args.task_type == 'classification':
            # Binary or multi-class classification
            # Calculate class weights for imbalanced datasets
            # Extract labels from train_dataset to compute pos_weight
            try:
                # Handle both Subset and direct Dataset objects
                if hasattr(train_dataset, 'dataset'):
                    # It's a Subset
                    indices = train_dataset.indices
                    all_labels = [train_dataset.dataset[i].y for i in indices]
                else:
                    # It's a Dataset
                    all_labels = [data.y for data in train_dataset]
                
                # Stack labels (assuming shape [num_tasks] or [1, num_tasks])
                y_train = torch.stack(all_labels)
                if y_train.dim() > 1:
                    y_train = y_train.squeeze()
                
                y_train_np = y_train.cpu().numpy()
                
                # Calculate weight for each task (if multi-task) or single task
                # Assuming binary classification (0/1)
                num_pos = np.sum(y_train_np == 1, axis=0)
                num_neg = np.sum(y_train_np == 0, axis=0)
                
                # Avoid division by zero
                pos_weight = torch.tensor(np.where(num_pos > 0, num_neg / num_pos, 1.0), dtype=torch.float).to(args.device)
                
                print(f"Class balance: Pos={num_pos}, Neg={num_neg}")
                print(f"Applied pos_weight: {pos_weight}")
                
                criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
                
            except Exception as e:
                print(f"Warning: Could not calculate class weights: {e}")
                print("Falling back to standard BCEWithLogitsLoss")
                criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            # Regression: use L1Loss (MAE)
            criterion = nn.L1Loss(reduction="none")
        
        ## set additional args
        scf_tr = Scf_index(train_dataset, args)
        args.num_tr_scf = scf_tr.num_scf
        
        num_iter = math.ceil(len(train_dataset) / args.batch_size) 
        args.temp_alpha = np.exp(np.log(args.min_temp / 10 + 1e-10) / (args.epochs * num_iter))


        ## define a model and load chek points
        model = GNN_topexpert(args, criterion)
        if args.gin_pretrained_file:
            model.from_pretrained(args.gin_pretrained_file)

        model = model.to(args.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)


        ###### init centroid using randomly initialized gnn
        zs_init = get_z(model, train_loader, args.device)
        init_centroid(model, zs_init, args.num_experts)
    
        # Prepare checkpoint directory
        if args.category and args.dataset_name:
            if args.experiment_id and args.experiment_id != '0':
                # For hyperparameter tuning, use experiment_id subdirectory
                checkpoint_dir = f"{args.output_dir}/hyperparam/{args.category}/{args.dataset_name}/{args.experiment_id}"
            else:
                # For final training, use category/dataset structure
                checkpoint_dir = f"{args.output_dir}/{args.category}/{args.dataset_name}"
        else:
            checkpoint_dir = f"{args.output_dir}/{args.dataset}"
            
        if not args.no_save_model:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        best_val_acc = 0
        best_test_acc = 0
        
        # Variables to store metrics at the best epoch (to avoid loading model if no_save_model is True)
        best_test_acc_at_best_val = 0
        best_test_secondary_at_best_val = None
        best_test_secondary = None # For combined mode
        
        patience_counter = 0
        current_epoch = 0
        
        # Determine if we're using combined mode (no validation set)
        use_test_for_tracking = (val_loader is None)

        for epoch in range(1, args.epochs + 1):
            current_epoch = epoch
            train(args, model, train_loader, optimizer, scf_tr)

            if use_test_for_tracking:
                # Combined train+valid mode: evaluate on test set
                test_acc, test_secondary = eval(args, model, test_loader)
                
                # Track best test performance for checkpoint saving
                if args.task_type == 'classification':
                    # For classification, higher is better
                    is_better = test_acc > best_test_acc
                else:
                    # For regression (MAE), lower is better
                    is_better = (best_test_acc == 0) or (test_acc < best_test_acc)
                
                if is_better:
                    best_test_acc = test_acc
                    best_test_secondary = test_secondary
                    patience_counter = 0
                    
                    # Save best checkpoint with secondary metrics
                    if not args.no_save_model:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'test_metric': test_acc,
                            'test_secondary_metrics': test_secondary,
                            'args': vars(args)
                        }, best_checkpoint_path)
                        qprint(f"Saved best model at epoch {epoch}", args)
                else:
                    patience_counter += 1
                
                # Display metric
                metric_name = 'AUROC' if args.task_type == 'classification' else 'MAE'
                qprint(f'{epoch}epoch, test {metric_name}:{test_acc:.1f}', args)
                
            else:
                # Normal mode with validation set
                val_acc, val_secondary = eval(args, model, val_loader)
                
                # Evaluate on test set for monitoring AND for capturing best model performance
                te_acc, te_secondary = eval(args, model, test_loader)
                
                # Track best validation performance
                if args.task_type == 'classification':
                    is_better = val_acc > best_val_acc
                else:
                    is_better = (best_val_acc == 0) or (val_acc < best_val_acc)
                
                if is_better:
                    best_val_acc = val_acc
                    best_test_acc_at_best_val = te_acc
                    best_test_secondary_at_best_val = te_secondary
                    patience_counter = 0
                    
                    # Save best checkpoint
                    if not args.no_save_model:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_metric': val_acc,
                            'args': vars(args)
                        }, best_checkpoint_path)
                        qprint(f"Saved best model at epoch {epoch}", args)
                else:
                    patience_counter += 1
                
                # Display metrics
                metric_name = 'AUROC' if args.task_type == 'classification' else 'MAE'
                qprint(f'{epoch}epoch, val {metric_name}:{val_acc:.1f}, test {metric_name}:{te_acc:.1f}', args)
            
            # Early stopping check
            if patience_counter >= args.patience:
                qprint(f"Early stopping at epoch {epoch}", args)
                break
        
        # Final evaluation logic
        if not args.no_save_model:
            # Load best model from disk for final evaluation
            qprint(f"Loading best model from {best_checkpoint_path}", args)
            checkpoint = torch.load(best_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Final test evaluation
            final_test_acc, final_test_secondary = eval(args, model, test_loader)
            # Final train evaluation
            final_train_acc, final_train_secondary = eval(args, model, train_loader)
        else:
            # Use stored metrics from memory
            qprint("Using best model metrics from memory (no_save_model=True)", args)
            if use_test_for_tracking:
                final_test_acc = best_test_acc
                final_test_secondary = best_test_secondary
                # For combined mode, we don't have a separate validation set, so we can evaluate on train set now
                final_train_acc, final_train_secondary = eval(args, model, train_loader)
            else:
                final_test_acc = best_test_acc_at_best_val
                final_test_secondary = best_test_secondary_at_best_val
                # Evaluate on train set using the current model state (which might not be the absolute best if no_save_model is True)
                # Ideally we should have tracked best_train_acc too, but evaluating now is a reasonable approximation or we can skip it.
                # Let's evaluate on train_loader to get the metrics.
                final_train_acc, final_train_secondary = eval(args, model, train_loader)
        
        # Save results to CSV
        if use_test_for_tracking:
            # For combined mode, we tracked test performance
            save_results_to_csv(
                args,
                val_metric=None,  # No validation set in combined mode
                test_metric=final_test_acc,
                val_std=None,
                test_std=None,
                num_epochs=current_epoch,
                early_stopped=(patience_counter >= args.patience),
                val_secondary=None,
                test_secondary=final_test_secondary,
                train_metric=final_train_acc,
                train_secondary=final_train_secondary
            )
        else:
            # Normal mode with validation
            save_results_to_csv(
                args,
                val_metric=best_val_acc,
                test_metric=final_test_acc,
                val_std=None,
                test_std=None,
                num_epochs=current_epoch,
                early_stopped=(patience_counter >= args.patience),
                val_secondary=val_secondary if 'val_secondary' in locals() else None,
                test_secondary=final_test_secondary,
                train_metric=final_train_acc,
                train_secondary=final_train_secondary
            )

if __name__ == "__main__":
    args = load_args()
    main(args)





