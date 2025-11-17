import math
import argparse
import os
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

def load_args():
    parser = argparse.ArgumentParser()

# seed & device
    parser.add_argument('--device_no', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
   
#dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold or cv")

#model
    parser.add_argument('-i', '--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('-c', '--ckpt_all', type=str, default='',
                        help='filename to read the model ')
    

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


# train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--gin_pretrained_file', type=str, default='', help='path to pre-trained GIN model')
    parser.add_argument('--experiment_id', type=int, default=0, help='experiment ID for tracking')


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


    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")

    # Bunch of classification tasks
    if args.dataset in ["dili2", "dili3", "hepg2"]:
        args.task_type = 'classification'
    elif args.dataset in ["hk2"]:
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
        args.num_classes = 2
    elif args.dataset == "hk2":
        args.num_tasks = 1
        args.num_classes = 1 # Regression task
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
            # BCEWithLogitsLoss expects float labels
            labels = labels.to(torch.float)
            loss_mat, num_valid_mat = model.clf_loss(pred, labels, g)
            main_loss = torch.sum(loss_mat / num_valid_mat) / args.num_tasks
        elif args.task_type == 'regression':
            # L1Loss (MAE) for regression
            loss_fn = nn.L1Loss()
            main_loss = loss_fn(pred, labels)
        else:
            raise ValueError("Invalid task type.")

        cluster_loss = F.kl_div(q.log(), p, reduction='sum')
        align_loss = model.alignment_loss(scf_idx, q)
        
        loss_total = main_loss + args.alpha * (cluster_loss / num_graph) + args.beta * align_loss
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()


def eval(args, model, loader):
    model.eval()
    
    y_true_list, y_pred_list = [], []
    for batch in loader:
        batch = batch.to(args.device)
        with torch.no_grad():
            pred, z, q_origin = model(batch)
            q, q_idx = model.assign_head(q_origin) # N x tasks x head
            
            if args.task_type == 'classification':
                # For classification, apply sigmoid and weight by expert assignment
                scores = torch.sum(torch.sigmoid(pred) * q, dim=-1)
                y_pred_list.append(scores)
            elif args.task_type == 'regression':
                # For regression, weight predictions by expert assignment
                scores = torch.sum(pred * q, dim=-1)
                y_pred_list.append(scores)
                
        y_true_list.append(batch.y.view(batch.id.shape[0], -1))

    y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred_list, dim=0).cpu().numpy()

    if args.task_type == 'classification':
        # Ensure y_true is in the correct format for roc_auc_score
        if args.num_classes > 2: # multiclass
            # For multi-class, y_true should be one-hot encoded or handled by roc_auc_score with multi_class='ovr'
            # Assuming y_pred has shape (n_samples, n_classes)
            return cal_roc(y_true, y_pred, num_classes=args.num_classes)
        else: # binary
            return cal_roc(y_true, y_pred)
    elif args.task_type == 'regression':
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError("Invalid task type.")


def save_results_to_csv(args, val_metric, test_metric, val_std=None, test_std=None, num_epochs=None, early_stopped=False):
    """Save experiment results to CSV file"""
    import pandas as pd
    
    csv_dir = '/home/choi0425/workspace/ADMET/workspace/output/hyperparam'
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{args.dataset}_progress.csv')
    
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
        'val_metric': f'{val_metric:.4f}',
        'val_metric_std': f'{val_std:.4f}' if val_std is not None else '',
        'test_metric': f'{test_metric:.4f}',
        'test_metric_std': f'{test_std:.4f}' if test_std is not None else '',
        'num_epochs_trained': num_epochs if num_epochs else '',
        'early_stopped': early_stopped,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    df = pd.DataFrame([result])
    
    # 파일이 없으면 헤더와 함께 생성, 있으면 append
    try:
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False, mode='w')
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)
    except Exception as e:
        print(f"Warning: Failed to save results to CSV: {e}")


def main(args):
    set_seed(args.seed)    

    # dataset split & data loader
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
        # raise ValueError("Invalid dataset name for target columns.")
    
    dataset = MoleculeCSVDataset(root=args.dataset_dir, dataset_name=args.dataset, smiles_col='smiles', target_cols=target_cols)
    
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
            criterion = nn.BCEWithLogitsLoss(reduction="none")
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
                val_acc = eval(args, model, val_loader)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                if (epoch % 10 == 0) or (epoch == args.epochs):
                    print(f'Epoch {epoch}, Val Acc: {val_acc:.4f}')

            test_acc = eval(args, model, test_loader)
            print(f'Fold {fold_idx+1} Test Acc: {test_acc:.4f}')
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

        metric_name = "Accuracy" if args.task_type == 'classification' else "MAE"
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
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        ## criterion
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        
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
    
        best_val_acc = 0
        patience_counter = 0
        current_epoch = 0

        for epoch in range(1, args.epochs + 1):
            current_epoch = epoch
            train(args, model, train_loader, optimizer, scf_tr)

            val_acc = eval(args, model, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            te_acc = eval(args, model, test_loader)

            print(f'{epoch}epoch, val acc:{val_acc:.1f}, test acc:{te_acc:.1f} ')
        
        # Final test evaluation
        final_test_acc = eval(args, model, test_loader)
        
        # Save results to CSV
        save_results_to_csv(
            args,
            val_metric=best_val_acc,
            test_metric=final_test_acc,
            val_std=None,
            test_std=None,
            num_epochs=current_epoch,
            early_stopped=(patience_counter >= args.patience)
        )

if __name__ == "__main__":
    args = load_args()
    main(args)





