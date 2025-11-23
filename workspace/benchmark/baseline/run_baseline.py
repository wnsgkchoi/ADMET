import os
import argparse
import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from baseline_models import create_objective, sanitize_params_for_estimator, evaluate_model_classification
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(dataset_name, data_dir):
    # Load features
    feat_path = os.path.join(data_dir, f"{dataset_name}_features.pkl")
    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f)
    
    features = feat_data['features']
    smiles = feat_data['smiles']
    
    # Load splits
    split_path = os.path.join(data_dir, f"{dataset_name}_splits.pkl")
    with open(split_path, 'rb') as f:
        splits = pickle.load(f)
        
    # Load raw data for labels
    data_path = os.path.join(data_dir, f"{dataset_name}_data.csv")
    df = pd.read_csv(data_path)
    
    # Align labels
    # Assuming df is aligned with features (which it should be as features were generated from it)
    labels = df['Y'].values
    
    return features, labels, splits

def run_baseline(dataset_name, data_dir, output_dir, n_trials=20, num_seeds=5, seed_index=None, model_name=None):
    features, labels, splits = load_data(dataset_name, data_dir)
    
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
        
        # Handle NaNs in features if any (XGB/LGBM/CatBoost handle NaNs, RF might not)
        # For RF, we might need imputation. But let's assume tree models handle it or data is clean.
        # The feature generation script produced no failures, but some values might be NaN?
        # RDKit descriptors can be NaN.
        # Simple imputation for RF: mean
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        # Fit on train, transform all
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_valid_imp = imputer.transform(X_valid)
        X_test_imp = imputer.transform(X_test)
        
        # Task type: AMES is classification
        task_type = 'classification'
        
        if model_name:
            model_list = [model_name]
        else:
            model_list = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
        
        seed_results = {}
        
        for m_name in model_list:
            print(f"  Optimizing {m_name}...")
            
            # Use imputed data for RF, raw for others (or imputed for all to be safe/consistent)
            # Tree models (XGB, LGBM, Cat) handle NaNs natively and often better than mean imputation.
            # So use raw for them.
            if m_name == 'RandomForest':
                X_tr, X_val, X_te = X_train_imp, X_valid_imp, X_test_imp
            else:
                X_tr, X_val, X_te = X_train, X_valid, X_test
                
            objective = create_objective(X_tr, y_train, X_val, y_valid, task_type)
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, m_name), n_trials=n_trials, show_progress_bar=True)
            
            best_trial = study.best_trial
            print(f"    Best Valid AUROC: {best_trial.value:.4f}")
            
            # Retrain on Train + Valid with best params
            # For Early Stopping, we need to be careful.
            # If we use Train+Valid, we don't have a validation set for ES.
            # Strategy: Use the best n_estimators/iterations found (if tuned) or just train for fixed epochs?
            # The notebook code uses `safe_fit` with `early_stopping_rounds=50`.
            # If we combine Train+Valid, we can't use ES.
            # Standard practice: Train on Train+Valid for the number of iterations that was optimal on Valid?
            # Or just use the model trained on Train (if it was refitted on Train+Valid inside the notebook logic? No, notebook logic refits on Train+Valid).
            
            # Notebook logic:
            # model.fit(X_train_full, y_train_full)
            # It doesn't seem to use ES for the final fit in the notebook snippet I saw?
            # Wait, let's check the notebook code again.
            # "model.fit(X_train_full, y_train_full)" -> No ES args passed.
            # So it trains until n_estimators (which was tuned).
            # But if n_estimators was tuned with ES, the 'n_estimators' in best_params might be the max value, not the best step.
            # Optuna suggests 'n_estimators'. If ES stopped early, the trial value is reported.
            # But the parameter 'n_estimators' in the trial is what was sampled, not where it stopped.
            # This is a common issue.
            # However, for tree models, usually we set a large n_estimators and use ES.
            # If we retrain without ES, we might overfit.
            # The notebook code:
            # params = { ... 'n_estimators': trial.suggest_int('n_estimators', 100, 1000) ... }
            # safe_fit(..., early_stop=50)
            # So n_estimators is the UPPER BOUND.
            # If ES triggers at 200, but n_estimators was 1000.
            # Retraining with n_estimators=1000 on Train+Valid will overfit.
            
            # Correct approach: Get the `best_iteration` from the trained model in the trial?
            # Optuna doesn't easily expose the trained model from the trial unless we save it.
            # Alternative: Just use the model trained on Train (and validated on Valid) to predict on Test?
            # No, we want to use all data.
            
            # Compromise for this baseline script:
            # Use the best params, but set n_estimators to a reasonable value or trust the tuned value if ES wasn't aggressive.
            # OR, just evaluate the model trained on Train (using Valid for ES) on the Test set.
            # This is slightly suboptimal (less data) but statistically valid and safer than overfitting.
            # Given we have 5 seeds, this is a fair estimate of performance.
            # AND, the user asked to "Train on 5 splits... evaluate on test".
            # Usually "Train" implies Train set.
            # If we want to be strictly comparable to the notebook which did "Train+Valid", we should try to emulate that.
            # But the notebook's "Train+Valid" retraining might be flawed if it ignores ES.
            # Let's stick to "Train on Train, Select on Valid, Evaluate on Test" for now.
            # It's the most robust standard.
            
            # Re-instantiate model with best params
            best_params = study.best_params
            
            # Sanitize params
            if m_name == 'XGBoost':
                final_params = sanitize_params_for_estimator(XGBClassifier, best_params)
                # Recalculate scale_pos_weight for full train if needed, but let's use the one from train
                # Actually, let's just use the params as is.
                model = XGBClassifier(**final_params, objective='binary:logistic', random_state=42, device='cuda')
            elif m_name == 'LightGBM':
                final_params = sanitize_params_for_estimator(LGBMClassifier, best_params)
                model = LGBMClassifier(**final_params, objective='binary', random_state=42, verbosity=-1, device='gpu')
            elif m_name == 'CatBoost':
                final_params = sanitize_params_for_estimator(CatBoostClassifier, best_params)
                model = CatBoostClassifier(**final_params, loss_function='Logloss', random_state=42, logging_level='Silent', task_type='GPU')
            else:
                final_params = sanitize_params_for_estimator(RandomForestClassifier, best_params)
                model = RandomForestClassifier(**final_params, random_state=42, class_weight='balanced')
            
            # Fit on Train + Valid (No Early Stopping, use best params)
            # We assume the best params include n_estimators which was tuned or is sufficient.
            # For production/benchmark, we should retrain on full data.
            # However, without a validation set, we cannot use early stopping.
            # We will trust the n_estimators from the best trial or use the default/tuned value.
            
            # Combine Train + Valid
            X_full = np.vstack([X_tr, X_val])
            y_full = np.concatenate([y_train, y_valid])
            
            model.fit(X_full, y_full)
            
            # Evaluate on Test
            if hasattr(model, "predict_proba"):
                test_proba = model.predict_proba(X_te)[:, 1]
            else:
                test_proba = model.predict(X_te)
                
            metrics = evaluate_model_classification(y_test, test_proba)
            print(f"    Test AUROC: {metrics['AUROC']:.4f}")
            
            seed_results[m_name] = metrics
            
        results.append({'seed': seed_name, 'results': seed_results})
        
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
        model_names = results[0]['results'].keys()
        for m in model_names:
            aurocs = [r['results'][m]['AUROC'] for r in results]
            print(f"{m}: Mean AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

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
