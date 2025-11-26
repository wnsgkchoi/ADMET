import os
import argparse
import pickle
import numpy as np
import pandas as pd
import optuna
import sys
import joblib
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Add scripts directory to path for target_datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from workspace.benchmark.utils import get_dataset_path
from target_datasets import ALL_TARGET_DATASETS
from baseline_models import (
    create_objective, 
    sanitize_params_for_estimator, 
    evaluate_model_classification, 
    evaluate_model_regression
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_task_type(dataset_name):
    if dataset_name in ALL_TARGET_DATASETS:
        return ALL_TARGET_DATASETS[dataset_name]['type']
    return 'classification'

def load_data(dataset_name, base_data_dir):
    # Try to find the directory
    possible_paths = [
        os.path.join(base_data_dir, dataset_name),
        os.path.join(base_data_dir, 'Tox', dataset_name),
        os.path.join(base_data_dir, 'ADME', dataset_name)
    ]
    
    dataset_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_dir = p
            break
            
    if dataset_dir is None:
        # Fallback to get_dataset_path logic
        dataset_dir = get_dataset_path(base_data_dir, dataset_name)
    
    # Fallback check for features file in base dir (legacy)
    if not os.path.exists(os.path.join(dataset_dir, f"{dataset_name}_features.pkl")):
        old_path = os.path.join(base_data_dir, f"{dataset_name}_features.pkl")
        if os.path.exists(old_path):
            dataset_dir = base_data_dir

    # Load features
    feat_path = os.path.join(dataset_dir, f"{dataset_name}_features.pkl")
    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f)
    
    features = feat_data['features']
    smiles = feat_data['smiles']
    
    # Load splits
    split_path = os.path.join(dataset_dir, f"{dataset_name}_splits.pkl")
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
        
    # Load raw data for labels
    data_path = os.path.join(dataset_dir, f"{dataset_name}_data.csv")
    df = pd.read_csv(data_path)
    
    # Align labels
    # Assuming df is aligned with features (which it should be as features were generated from it)
    labels = df['Y'].values
    
    return features, labels, splits

def run_baseline(dataset_name, data_dir, output_dir, n_trials=20, num_seeds=5, seed_index=None, model_name=None):
    features, labels, splits = load_data(dataset_name, data_dir)
    task_type = get_task_type(dataset_name)
    print(f"Task type for {dataset_name}: {task_type}")
    
    results = []
    
    # Filter splits if seed_index is provided
    if seed_index is not None:
        target_seed = f"seed_{seed_index + 1}"
        if target_seed not in splits:
             print(f"Seed {target_seed} not found.")
             return
        splits = {target_seed: splits[target_seed]}
        print(f"Running specific seed: {target_seed}")

    seed_count = 0
    for seed_name, split_indices in splits.items():
        if seed_index is None and seed_count >= num_seeds:
            break
        seed_count += 1
        
        print(f"\n{'='*50}")
        print(f"Running {seed_name}...")
        print(f"{'='*50}")
        
        train_idx = split_indices['train']
        valid_idx = split_indices['valid']
        test_idx = split_indices['test']
        
        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_valid = features[valid_idx]
        y_valid = labels[valid_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]
        
        # Imputation
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_valid_imp = imputer.transform(X_valid)
        X_test_imp = imputer.transform(X_test)
        
        if model_name:
            model_list = [model_name]
        else:
            model_list = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
        
        seed_results = {}
        
        for m_name in model_list:
            print(f"  Optimizing {m_name}...")
            
            if m_name == 'RandomForest':
                X_tr, X_val, X_te = X_train_imp, X_valid_imp, X_test_imp
            else:
                X_tr, X_val, X_te = X_train, X_valid, X_test
                
            objective = create_objective(X_tr, y_train, X_val, y_valid, task_type)
            direction = 'maximize' if task_type == 'classification' else 'minimize'
            study = optuna.create_study(direction=direction)
            study.optimize(lambda trial: objective(trial, m_name), n_trials=n_trials, show_progress_bar=True)
            
            best_trial = study.best_trial
            metric_label = 'AUROC' if task_type == 'classification' else 'MAE'
            print(f"    Best Valid {metric_label}: {best_trial.value:.4f}")
            
            # Re-instantiate model with best params
            best_params = study.best_params
            
<<<<<<< HEAD
            # Sanitize params
            if m_name == 'XGBoost':
                final_params = sanitize_params_for_estimator(XGBClassifier, best_params)
                # Recalculate scale_pos_weight for full train if needed, but let's use the one from train
                # Actually, let's just use the params as is.
                model = XGBClassifier(**final_params, objective='binary:logistic', random_state=42, device='cpu', tree_method='hist')
            elif m_name == 'LightGBM':
                final_params = sanitize_params_for_estimator(LGBMClassifier, best_params)
                model = LGBMClassifier(**final_params, objective='binary', random_state=42, verbosity=-1, device='cpu')
            elif m_name == 'CatBoost':
                final_params = sanitize_params_for_estimator(CatBoostClassifier, best_params)
                model = CatBoostClassifier(**final_params, loss_function='Logloss', random_state=42, logging_level='Silent', task_type='CPU')
            else:
                final_params = sanitize_params_for_estimator(RandomForestClassifier, best_params)
                model = RandomForestClassifier(**final_params, random_state=42, class_weight='balanced')
            
            # Fit on Train + Valid (No Early Stopping, use best params)
            # We assume the best params include n_estimators which was tuned or is sufficient.
            # For production/benchmark, we should retrain on full data.
            # However, without a validation set, we cannot use early stopping.
            # We will trust the n_estimators from the best trial or use the default/tuned value.
            
            # Combine Train + Valid
=======
            if task_type == 'classification':
                if m_name == 'XGBoost':
                    final_params = sanitize_params_for_estimator(XGBClassifier, best_params)
                    model = XGBClassifier(**final_params, objective='binary:logistic', random_state=42, device='cpu', tree_method='hist')
                elif m_name == 'LightGBM':
                    final_params = sanitize_params_for_estimator(LGBMClassifier, best_params)
                    model = LGBMClassifier(**final_params, objective='binary', random_state=42, verbosity=-1, device='cpu')
                elif m_name == 'CatBoost':
                    final_params = sanitize_params_for_estimator(CatBoostClassifier, best_params)
                    model = CatBoostClassifier(**final_params, loss_function='Logloss', random_state=42, logging_level='Silent', task_type='CPU')
                else:
                    final_params = sanitize_params_for_estimator(RandomForestClassifier, best_params)
                    model = RandomForestClassifier(**final_params, random_state=42, class_weight='balanced')
            else: # Regression
                if m_name == 'XGBoost':
                    final_params = sanitize_params_for_estimator(XGBRegressor, best_params)
                    model = XGBRegressor(**final_params, objective='reg:squarederror', random_state=42, device='cpu', tree_method='hist')
                elif m_name == 'LightGBM':
                    final_params = sanitize_params_for_estimator(LGBMRegressor, best_params)
                    model = LGBMRegressor(**final_params, objective='regression', random_state=42, verbosity=-1, device='cpu')
                elif m_name == 'CatBoost':
                    final_params = sanitize_params_for_estimator(CatBoostRegressor, best_params)
                    model = CatBoostRegressor(**final_params, loss_function='RMSE', random_state=42, logging_level='Silent', task_type='CPU')
                else:
                    final_params = sanitize_params_for_estimator(RandomForestRegressor, best_params)
                    model = RandomForestRegressor(**final_params, random_state=42)

            # Fit on Train + Valid
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
            X_full = np.vstack([X_tr, X_val])
            y_full = np.concatenate([y_train, y_valid])
            
            model.fit(X_full, y_full)
            
            # Evaluate on Test
            if task_type == 'classification':
                if hasattr(model, "predict_proba"):
                    test_preds = model.predict_proba(X_te)[:, 1]
                else:
                    test_preds = model.predict(X_te)
                metrics = evaluate_model_classification(y_test, test_preds)
                print(f"    Test AUROC: {metrics['AUROC']:.4f}")
            else:
                test_preds = model.predict(X_te)
                metrics = evaluate_model_regression(y_test, test_preds)
                print(f"    Test MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
                
<<<<<<< HEAD
            metrics = evaluate_model_classification(y_test, test_proba)
            print(f"    Test AUROC: {metrics['AUROC']:.4f}")
            
=======
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
            metrics['best_params'] = best_params
            seed_results[m_name] = metrics
            
        results.append({'seed': seed_name, 'results': seed_results})
        
        # Log best model for this seed to CSV
<<<<<<< HEAD
        best_model_name = max(seed_results, key=lambda k: seed_results[k]['AUROC'])
        best_metric = seed_results[best_model_name]['AUROC']
=======
        if task_type == 'classification':
            best_model_name = max(seed_results, key=lambda k: seed_results[k]['AUROC'])
            best_metric = seed_results[best_model_name]['AUROC']
            # Prepare CSV row with all metrics
            # Seed,Model,AUROC,F1,Accuracy,Precision,Recall,Specificity,AUPRC,Best_Params
            header = "Seed,Model,AUROC,F1,Accuracy,Precision,Recall,Specificity,AUPRC,Best_Params"
        else:
            best_model_name = min(seed_results, key=lambda k: seed_results[k]['MAE'])
            best_metric = seed_results[best_model_name]['MAE']
            # Prepare CSV row with all metrics
            # Seed,Model,MAE,RMSE,R2,Best_Params
            header = "Seed,Model,MAE,RMSE,R2,Best_Params"
            
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
        best_params_str = str(seed_results[best_model_name]['best_params'])
        
        progress_csv = os.path.join(output_dir, f"{dataset_name}_progress.csv")
        file_exists = os.path.isfile(progress_csv)
        
        with open(progress_csv, 'a') as f:
            if not file_exists:
<<<<<<< HEAD
                f.write("Seed,Model,Test_AUROC,Best_Params\n")
            f.write(f"{seed_name},{best_model_name},{best_metric:.4f},\"{best_params_str}\"\n")
=======
                f.write(header + "\n")
            # Re-construct row_str with correct best_params_str
            if task_type == 'classification':
                res = seed_results[best_model_name]
                row_str = f"{seed_name},{best_model_name},{res['AUROC']:.4f},{res['F1']:.4f},{res['Accuracy']:.4f},{res['Precision']:.4f},{res['Recall(Sensitivity)']:.4f},{res['Specificity']:.4f},{res['AUPRC']:.4f},\"{best_params_str}\""
            else:
                res = seed_results[best_model_name]
                row_str = f"{seed_name},{best_model_name},{res['MAE']:.4f},{res['RMSE']:.4f},{res['R2']:.4f},\"{best_params_str}\""
            f.write(row_str + "\n")
>>>>>>> 06c0ab56d1c4430ff3e23ab6e23fb5f18b88c717
        print(f"Logged best result for {seed_name} to {progress_csv}")
        
    # Save results
    if seed_index is not None or model_name is not None:
        suffix = ""
        if seed_index is not None:
            suffix += f"_seed{seed_index+1}"
        if model_name is not None:
            suffix += f"_{model_name}"
        res_path = os.path.join(output_dir, f"{dataset_name}_baseline{suffix}.pkl")
    else:
        res_path = os.path.join(output_dir, f"{dataset_name}_baseline_results.pkl")
        
    with open(res_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved baseline results to {res_path}")
    
    # Print Summary only if running full suite
    if seed_index is None and model_name is None:
        print("\nSummary:")
        progress_csv = os.path.join(output_dir, f"{dataset_name}_progress.csv")
        
        best_values = []
        for r in results:
            s_res = r['results']
            if task_type == 'classification':
                val = max(s_res[m]['AUROC'] for m in s_res)
            else:
                val = min(s_res[m]['MAE'] for m in s_res)
            best_values.append(val)
            
        mean_val = np.mean(best_values)
        std_val = np.std(best_values)
        
        with open(progress_csv, 'a') as f:
            f.write(f"\nFinal Summary (Average of Best Models across {len(best_values)} seeds):\n")
            metric_label = 'AUROC' if task_type == 'classification' else 'MAE'
            f.write(f"Mean Best {metric_label}: {mean_val:.4f} ± {std_val:.4f}\n")
        print(f"Mean Best {metric_label}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AMES')
    parser.add_argument('--data_dir', type=str, default='workspace/benchmark/data')
    parser.add_argument('--output_dir', type=str, default='workspace/benchmark/results')
    parser.add_argument('--n_trials', type=int, default=10) # Small number for quick testing
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use (e.g., 0, 1, 2, 3)')
    parser.add_argument('--seed_index', type=int, default=None, help='Index of seed to run (0-4)')
    parser.add_argument('--model_name', type=str, default=None, help='Specific model to run')
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if gpu_id is provided
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    run_baseline(args.dataset, args.data_dir, args.output_dir, args.n_trials, args.num_seeds, args.seed_index, args.model_name)
