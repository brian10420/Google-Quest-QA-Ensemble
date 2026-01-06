"""
Step 3: Train Stacking Meta-Learner
===================================
Run this locally after optimizing weights.

This script trains a two-level stacking model:
- Level 1: OOF predictions from RoBERTa, DeBERTa, Mamba (90 features)
- Level 2: Meta features (length, category, disagreement, etc.)

Models:
- Ridge Regression (per target)
- LightGBM (per target)
- Final ensemble of both

Output:
- ridge_models/: Trained Ridge models for each target
- lgb_models/: Trained LightGBM models for each target  
- stacker_config.json: Configuration and feature info
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'oof_dir': './oof_data/',
    'params_dir': './optimized_params/',
    'output_dir': './stacker_models/',
    
    'n_stacker_folds': 5,  # CV folds for stacker
    'random_seed': 42,
    
    'ridge_alpha': 1.0,
    
    'lgb_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_estimators': 200,
        'early_stopping_rounds': 30,
    },
    
    'target_cols': [
        'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
        'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
        'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
        'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
        'question_type_compare', 'question_type_consequence', 'question_type_definition',
        'question_type_entity', 'question_type_instructions', 'question_type_procedure',
        'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
        'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
        'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
        'answer_type_reason_explanation', 'answer_well_written'
    ]
}


# ==========================================
# Feature Engineering
# ==========================================
def create_stacking_features(oof_roberta, oof_deberta, oof_mamba, meta_features):
    """
    Create feature matrix for stacking model.
    
    Features:
    - 30 RoBERTa predictions
    - 30 DeBERTa predictions
    - 30 Mamba predictions
    - Model disagreement features (std, range)
    - Meta features (length, category, etc.)
    """
    n_samples = oof_roberta.shape[0]
    
    # Stack base predictions
    features = np.hstack([oof_roberta, oof_deberta, oof_mamba])  # (N, 90)
    
    # Add model disagreement features
    all_preds = np.stack([oof_roberta, oof_deberta, oof_mamba], axis=2)  # (N, 30, 3)
    
    # Per-target disagreement
    pred_std = np.std(all_preds, axis=2)  # (N, 30)
    pred_range = np.max(all_preds, axis=2) - np.min(all_preds, axis=2)  # (N, 30)
    pred_mean = np.mean(all_preds, axis=2)  # (N, 30)
    
    # Global disagreement (across all targets)
    global_std = np.mean(pred_std, axis=1, keepdims=True)  # (N, 1)
    global_range = np.mean(pred_range, axis=1, keepdims=True)  # (N, 1)
    
    # Meta features
    meta_array = meta_features[['original_token_length', 'log_token_length', 
                                'exceeds_512', 'exceeds_1024',
                                'q_a_ratio', 'q_len', 'a_len', 'category']].values
    
    # One-hot category features
    cat_cols = [c for c in meta_features.columns if c.startswith('cat_')]
    if cat_cols:
        cat_onehot = meta_features[cat_cols].values
        meta_array = np.hstack([meta_array, cat_onehot])
    
    # Combine all features
    features = np.hstack([
        features,           # 90 base predictions
        pred_std,          # 30 disagreement std
        pred_range,        # 30 disagreement range
        pred_mean,         # 30 ensemble mean
        global_std,        # 1 global std
        global_range,      # 1 global range
        meta_array         # meta features
    ])
    
    # Create feature names
    feature_names = []
    for name in ['roberta', 'deberta', 'mamba']:
        for col in CONFIG['target_cols']:
            feature_names.append(f'{name}_{col}')
    
    for col in CONFIG['target_cols']:
        feature_names.append(f'std_{col}')
    for col in CONFIG['target_cols']:
        feature_names.append(f'range_{col}')
    for col in CONFIG['target_cols']:
        feature_names.append(f'mean_{col}')
    
    feature_names.extend(['global_std', 'global_range'])
    feature_names.extend(['token_length', 'log_token_length', 'exceeds_512', 'exceeds_1024',
                         'q_a_ratio', 'q_len', 'a_len', 'category'])
    feature_names.extend(cat_cols)
    
    print(f"  Created {features.shape[1]} stacking features")
    return features, feature_names


def create_target_specific_features(oof_roberta, oof_deberta, oof_mamba, target_idx, meta_features):
    """
    Create features for a specific target column.
    Only includes relevant predictions and disagreement.
    """
    n_samples = oof_roberta.shape[0]
    
    # Base predictions for this target
    r_pred = oof_roberta[:, target_idx:target_idx+1]
    d_pred = oof_deberta[:, target_idx:target_idx+1]
    m_pred = oof_mamba[:, target_idx:target_idx+1]
    
    # Disagreement for this target
    all_preds = np.stack([r_pred[:, 0], d_pred[:, 0], m_pred[:, 0]], axis=1)
    target_std = np.std(all_preds, axis=1, keepdims=True)
    target_range = (np.max(all_preds, axis=1) - np.min(all_preds, axis=1)).reshape(-1, 1)
    target_mean = np.mean(all_preds, axis=1, keepdims=True)
    
    # Meta features
    meta_array = meta_features[['original_token_length', 'log_token_length',
                                'exceeds_512', 'exceeds_1024',
                                'q_a_ratio', 'q_len', 'a_len', 'category']].values
    
    # One-hot category
    cat_cols = [c for c in meta_features.columns if c.startswith('cat_')]
    if cat_cols:
        cat_onehot = meta_features[cat_cols].values
        meta_array = np.hstack([meta_array, cat_onehot])
    
    features = np.hstack([
        r_pred, d_pred, m_pred,  # 3
        target_std, target_range, target_mean,  # 3
        meta_array  # meta features
    ])
    
    return features


# ==========================================
# Training Functions
# ==========================================
def train_ridge_stacker(X, y, n_folds=5, alpha=1.0):
    """Train Ridge regression with CV for a single target"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_preds = np.zeros(len(y))
    models = []
    scalers = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train
        model = Ridge(alpha=alpha, random_state=CONFIG['random_seed'])
        model.fit(X_train_scaled, y_train)
        
        # Predict
        oof_preds[val_idx] = model.predict(X_val_scaled)
        
        models.append(model)
        scalers.append(scaler)
    
    # Clip predictions to [0, 1]
    oof_preds = np.clip(oof_preds, 0, 1)
    
    return models, scalers, oof_preds


def train_lgb_stacker(X, y, n_folds=5):
    """Train LightGBM with CV for a single target"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=CONFIG['random_seed'])
    
    oof_preds = np.zeros(len(y))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        model = lgb.train(
            CONFIG['lgb_params'],
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(CONFIG['lgb_params']['early_stopping_rounds'], verbose=False)]
        )
        
        # Predict
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)
    
    # Clip predictions to [0, 1]
    oof_preds = np.clip(oof_preds, 0, 1)
    
    return models, oof_preds


def compute_spearman_single(preds, targets):
    """Compute Spearman for single column"""
    if np.std(preds) < 1e-9 or np.std(targets) < 1e-9:
        return 0
    corr = spearmanr(preds, targets).correlation
    return corr if not np.isnan(corr) else 0


# ==========================================
# Main Training Loop
# ==========================================
def train_all_stackers(oof_roberta, oof_deberta, oof_mamba, targets, meta_features):
    """Train all stacking models"""
    
    os.makedirs(os.path.join(CONFIG['output_dir'], 'ridge_models'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['output_dir'], 'lgb_models'), exist_ok=True)
    
    n_targets = len(CONFIG['target_cols'])
    
    # Store results
    ridge_oof = np.zeros_like(targets)
    lgb_oof = np.zeros_like(targets)
    ensemble_oof = np.zeros_like(targets)
    
    all_ridge_models = {}
    all_ridge_scalers = {}
    all_lgb_models = {}
    
    print("\n" + "="*60)
    print("Training Stacking Models")
    print("="*60)
    
    for i, col in enumerate(CONFIG['target_cols']):
        # Create features for this target
        X = create_target_specific_features(
            oof_roberta, oof_deberta, oof_mamba, i, meta_features
        )
        y = targets[:, i]
        
        # Train Ridge
        ridge_models, ridge_scalers, ridge_preds = train_ridge_stacker(
            X, y, CONFIG['n_stacker_folds'], CONFIG['ridge_alpha']
        )
        ridge_oof[:, i] = ridge_preds
        all_ridge_models[col] = ridge_models
        all_ridge_scalers[col] = ridge_scalers
        
        # Train LightGBM
        lgb_models, lgb_preds = train_lgb_stacker(
            X, y, CONFIG['n_stacker_folds']
        )
        lgb_oof[:, i] = lgb_preds
        all_lgb_models[col] = lgb_models
        
        # Ensemble Ridge + LGB
        ensemble_oof[:, i] = (ridge_preds + lgb_preds) / 2
        
        # Progress
        if (i + 1) % 10 == 0:
            ridge_score = compute_spearman_single(ridge_preds, y)
            lgb_score = compute_spearman_single(lgb_preds, y)
            ens_score = compute_spearman_single(ensemble_oof[:, i], y)
            print(f"  [{i+1}/{n_targets}] {col}: Ridge={ridge_score:.4f}, LGB={lgb_score:.4f}, Ens={ens_score:.4f}")
    
    # Save models
    print("\n  Saving models...")
    
    with open(os.path.join(CONFIG['output_dir'], 'ridge_models', 'models.pkl'), 'wb') as f:
        pickle.dump({'models': all_ridge_models, 'scalers': all_ridge_scalers}, f)
    
    with open(os.path.join(CONFIG['output_dir'], 'lgb_models', 'models.pkl'), 'wb') as f:
        pickle.dump(all_lgb_models, f)
    
    return ridge_oof, lgb_oof, ensemble_oof


# ==========================================
# Evaluation
# ==========================================
def sigmoid(x):
    """Numerically stable sigmoid"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def compute_length_dependent_weights(lengths, a, b, roberta_base, deberta_base):
    """Compute per-sample weights based on token length."""
    normalized_lengths = (lengths - 512) / 512
    w_mamba = sigmoid(a * normalized_lengths + b)
    remaining = 1 - w_mamba
    total_base = roberta_base + deberta_base + 1e-9
    w_roberta = remaining * (roberta_base / total_base)
    w_deberta = remaining * (deberta_base / total_base)
    return w_roberta, w_deberta, w_mamba


def apply_weighted_ensemble(oof_roberta, oof_deberta, oof_mamba, w_roberta, w_deberta, w_mamba):
    """Apply per-sample weights to create ensemble predictions"""
    w_r = w_roberta[:, np.newaxis]
    w_d = w_deberta[:, np.newaxis]
    w_m = w_mamba[:, np.newaxis]
    return w_r * oof_roberta + w_d * oof_deberta + w_m * oof_mamba


def evaluate_stackers(oof_roberta, oof_deberta, oof_mamba, targets, 
                     ridge_oof, lgb_oof, ensemble_oof, best_params):
    """Compare all methods"""
    
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    def mean_spearman(preds, targets):
        scores = []
        for i in range(preds.shape[1]):
            score = compute_spearman_single(preds[:, i], targets[:, i])
            scores.append(score)
        return np.mean(scores)
    
    # Individual models
    score_r = mean_spearman(oof_roberta, targets)
    score_d = mean_spearman(oof_deberta, targets)
    score_m = mean_spearman(oof_mamba, targets)
    
    # Simple baselines
    simple_avg = (oof_roberta + oof_deberta + oof_mamba) / 3
    score_simple = mean_spearman(simple_avg, targets)
    
    # Optimized weights (from step 2)
    meta = pd.read_csv(os.path.join(CONFIG['oof_dir'], 'meta_features.csv'))
    lengths = meta['original_token_length'].values
    
    global_params = best_params['global_params']
    w_r, w_d, w_m = compute_length_dependent_weights(
        lengths,
        global_params['sigmoid_slope'],
        global_params['sigmoid_intercept'],
        global_params['roberta_base'],
        global_params['deberta_base']
    )
    optimized_ensemble = apply_weighted_ensemble(oof_roberta, oof_deberta, oof_mamba, w_r, w_d, w_m)
    score_optimized = mean_spearman(optimized_ensemble, targets)
    
    # Stacker scores
    score_ridge = mean_spearman(ridge_oof, targets)
    score_lgb = mean_spearman(lgb_oof, targets)
    score_stacker_ens = mean_spearman(ensemble_oof, targets)
    
    # Best combination: Optimized weights + Stacker ensemble
    final_ensemble = (optimized_ensemble + ensemble_oof) / 2
    score_final = mean_spearman(final_ensemble, targets)
    
    print("\n  Model Performance:")
    print("  " + "-"*50)
    print(f"  {'Method':<30} | {'Spearman':>10}")
    print("  " + "-"*50)
    print(f"  {'RoBERTa Only':<30} | {score_r:>10.5f}")
    print(f"  {'DeBERTa Only':<30} | {score_d:>10.5f}")
    print(f"  {'Mamba Only':<30} | {score_m:>10.5f}")
    print("  " + "-"*50)
    print(f"  {'Simple Average':<30} | {score_simple:>10.5f}")
    print(f"  {'Length-Weighted (Optimized)':<30} | {score_optimized:>10.5f}")
    print("  " + "-"*50)
    print(f"  {'Stacker (Ridge)':<30} | {score_ridge:>10.5f}")
    print(f"  {'Stacker (LightGBM)':<30} | {score_lgb:>10.5f}")
    print(f"  {'Stacker (Ridge+LGB)':<30} | {score_stacker_ens:>10.5f}")
    print("  " + "-"*50)
    print(f"  {'FINAL (Optimized + Stacker)':<30} | {score_final:>10.5f}")
    print("  " + "-"*50)
    
    # Save final OOF for analysis
    np.save(os.path.join(CONFIG['output_dir'], 'final_oof.npy'), final_ensemble)
    
    return {
        'roberta': score_r,
        'deberta': score_d,
        'mamba': score_m,
        'simple_avg': score_simple,
        'optimized': score_optimized,
        'ridge': score_ridge,
        'lgb': score_lgb,
        'stacker_ensemble': score_stacker_ens,
        'final': score_final
    }


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    oof_roberta = np.load(os.path.join(CONFIG['oof_dir'], 'oof_roberta.npy'))
    oof_deberta = np.load(os.path.join(CONFIG['oof_dir'], 'oof_deberta.npy'))
    oof_mamba = np.load(os.path.join(CONFIG['oof_dir'], 'oof_mamba.npy'))
    targets = np.load(os.path.join(CONFIG['oof_dir'], 'oof_targets.npy'))
    meta_features = pd.read_csv(os.path.join(CONFIG['oof_dir'], 'meta_features.csv'))
    
    with open(os.path.join(CONFIG['params_dir'], 'best_params.json'), 'r') as f:
        best_params = json.load(f)
    
    print(f"  Loaded {len(targets)} samples with {targets.shape[1]} targets")
    
    # Train stackers
    ridge_oof, lgb_oof, ensemble_oof = train_all_stackers(
        oof_roberta, oof_deberta, oof_mamba, targets, meta_features
    )
    
    # Evaluate
    scores = evaluate_stackers(
        oof_roberta, oof_deberta, oof_mamba, targets,
        ridge_oof, lgb_oof, ensemble_oof, best_params
    )
    
    # Save config
    config_to_save = {
        'scores': scores,
        'ridge_alpha': CONFIG['ridge_alpha'],
        'lgb_params': CONFIG['lgb_params'],
        'n_folds': CONFIG['n_stacker_folds'],
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'stacker_config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"\n  Saved all models to {CONFIG['output_dir']}/")
    print("\n>>> Step 3 Complete! Run step4_inference_kaggle.py on Kaggle next.")


if __name__ == "__main__":
    main()
