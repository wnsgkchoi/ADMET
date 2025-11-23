import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import inspect

# -------------------------
# safe_fit: fit 호출을 안전하게 래핑
# -------------------------
def safe_fit(model, X_train, y_train, X_valid, y_valid, early_stop=None, verbose=False):
    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs = {}
    if 'eval_set' in fit_sig:
        fit_kwargs['eval_set'] = [(X_valid, y_valid)]
    if early_stop is not None and 'early_stopping_rounds' in fit_sig:
        fit_kwargs['early_stopping_rounds'] = int(early_stop)
    if 'verbose' in fit_sig:
        fit_kwargs['verbose'] = verbose
    # 일부 래퍼는 use_best_model, callbacks 등 다르므로 최소한으로 전달
    return model.fit(X_train, y_train, **fit_kwargs)

# -------------------------
# sanitize_params_for_estimator: 생성자에 허용된 키만 전달
# -------------------------
def sanitize_params_for_estimator(estimator_cls, params):
    try:
        valid_keys = set(estimator_cls().get_params().keys())
    except Exception:
        valid_keys = set(inspect.signature(estimator_cls.__init__).parameters.keys())
        valid_keys.discard('self')
    return {k: v for k, v in params.items() if k in valid_keys}

def evaluate_model_classification(y_true, y_pred_proba):
    """분류 모델 종합 평가 (AUPRC 추가)"""
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    
    return {
        'AUROC': roc_auc_score(y_true, y_pred_proba),
        'F1': f1_score(y_true, y_pred_class, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred_class),
        'Precision': precision_score(y_true, y_pred_class, zero_division=0),
        'Recall(Sensitivity)': recall_score(y_true, y_pred_class, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'AUPRC': average_precision_score(y_true, y_pred_proba),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

def create_objective(X_train, y_train, X_valid, y_valid, task_type):
    def objective_func(trial, model_name):
        try:
            if task_type == 'classification':
                y_train_int = y_train.astype(int)
                y_valid_int = y_valid.astype(int)
                unique, counts = np.unique(y_train_int, return_counts=True)
                class_counts = dict(zip(unique, counts))
                pos = class_counts.get(1, 0)
                neg = class_counts.get(0, 0)
                scale_pos_weight = (neg / pos) if pos > 0 else 1.0

                if model_name == 'XGBoost':
                    # booster = trial.suggest_categorical('xgb_booster', ['gbtree','dart'])
                    booster = 'gbtree' # ADMET_Ver4.ipynb uses default (gbtree)
                    params = {
                        'objective': 'binary:logistic', 'random_state': 42, 'n_jobs': -1,
                        'scale_pos_weight': scale_pos_weight, 'tree_method':'hist', 'device': 'cuda',
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'booster': booster
                    }
                    # if booster == 'dart':
                    #     params.update({
                    #         'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
                    #         'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
                    #         'sample_type': trial.suggest_categorical('sample_type', ['uniform','weighted']),
                    #         'normalize_type': trial.suggest_categorical('normalize_type', ['tree','forest'])
                    #     })
                    model = XGBClassifier(**params)
                    try:
                        safe_fit(model, X_train, y_train_int, X_valid, y_valid_int, early_stop=50, verbose=False)
                    except Exception:
                        model.fit(X_train, y_train_int)

                elif model_name == 'LightGBM':
                    # boosting_type = trial.suggest_categorical('lgb_boosting_type', ['gbdt','dart'])
                    boosting_type = 'gbdt' # ADMET_Ver4.ipynb uses default (gbdt)
                    params = {
                        'objective':'binary','metric':'auc','random_state':42,'verbosity':-1,'n_jobs':-1,
                        'scale_pos_weight': scale_pos_weight, 'device': 'cpu', # GPU overhead is high for small data / varying max_bin
                        'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),
                        'n_estimators':trial.suggest_int('n_estimators',100,1000),
                        'num_leaves':trial.suggest_int('num_leaves',20,300),
                        'max_depth':trial.suggest_int('max_depth',3,15),
                        'min_child_samples':trial.suggest_int('min_child_samples',5,100),
                        'subsample':trial.suggest_float('subsample',0.5,1.0),
                        'colsample_bytree':trial.suggest_float('colsample_bytree',0.5,1.0),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                        'min_split_gain':trial.suggest_float('min_split_gain',0.0,1.0),
                        'max_bin': trial.suggest_int('max_bin', 63, 255),
                        'boosting_type': boosting_type
                    }
                    # if boosting_type == 'dart':
                    #     params.update({
                    #         'drop_rate': trial.suggest_float('drop_rate', 0.0, 0.5),
                    #         'max_drop': trial.suggest_int('max_drop', 1, 50),
                    #         'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5)
                    #     })
                    model = LGBMClassifier(**params)
                    try:
                        safe_fit(model, X_train, y_train_int, X_valid, y_valid_int, early_stop=50, verbose=False)
                    except Exception:
                        model.fit(X_train, y_train_int)

                elif model_name == 'CatBoost':
                    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
                    params = {
                        'random_state':42,'logging_level':'Silent','loss_function':'Logloss','scale_pos_weight':scale_pos_weight,
                        'task_type': 'GPU',
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'iterations': trial.suggest_int('iterations', 100, 1000),
                        'depth': trial.suggest_int('depth', 3, 12),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
                        'bootstrap_type': bootstrap_type,
                        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0)
                    }
                    if bootstrap_type == 'Bayesian':
                        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 2.0)
                    else:
                        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                    model = CatBoostClassifier(**params)
                    try:
                        safe_fit(model, X_train, y_train_int, X_valid, y_valid_int, early_stop=50, verbose=False)
                    except Exception:
                        model.fit(X_train, y_train_int)

                else:  # RandomForestClassifier
                    bootstrap = trial.suggest_categorical('rf_bootstrap', [True, False])
                    params = {
                        'random_state':42,'n_jobs':-1,'class_weight':'balanced',
                        'n_estimators':trial.suggest_int('n_estimators',100,1000),
                        'max_depth':trial.suggest_int('max_depth',3,20),
                        'min_samples_split':trial.suggest_int('min_samples_split',2,30),
                        'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,20),
                        'max_features':trial.suggest_categorical('max_features',['sqrt','log2']),
                        'ccp_alpha':trial.suggest_float('ccp_alpha',0.0,0.1),
                        'bootstrap': bootstrap
                    }
                    if bootstrap:
                        use_max_samples = trial.suggest_categorical('rf_use_max_samples', [False, True])
                        if use_max_samples:
                            params['max_samples'] = trial.suggest_float('rf_max_samples', 0.5, 1.0)
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train_int)

                # compute AUROC for train & valid (predict_proba preferred)
                def get_scores_proba(est, Xtr, ytr, Xv, yv):
                    if hasattr(est, "predict_proba"):
                        trp = est.predict_proba(Xtr)[:, 1]; vp = est.predict_proba(Xv)[:, 1]
                    elif hasattr(est, "decision_function"):
                        trp = est.decision_function(Xtr); vp = est.decision_function(Xv)
                    else:
                        trp = est.predict(Xtr); vp = est.predict(Xv)
                    tr_auc = float(roc_auc_score(ytr, trp))
                    val_auc = float(roc_auc_score(yv, vp))
                    return tr_auc, val_auc

                train_auroc, val_auroc = get_scores_proba(model, X_train, y_train_int, X_valid, y_valid_int)
                trial.set_user_attr('train_auroc', train_auroc)
                trial.set_user_attr('val_auroc', val_auroc)
                return float(val_auroc)

        except Exception as e:
            print(f"  - ⚠️ 최적화 예외 발생 (model={model_name}, task={task_type}): {e}")
            return 0.0

    return objective_func
