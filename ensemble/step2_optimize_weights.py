"""
Step 2: Optimize Length-Dependent Ensemble Weights
==================================================
Run this locally after generating OOF predictions.

This script uses Optuna to find optimal parameters for a sigmoid-based
length-dependent weighting scheme:

    w_mamba(length) = sigmoid(a * length + b)
    w_roberta = (1 - w_mamba) * r / (r + d)
    w_deberta = (1 - w_mamba) * d / (r + d)

Where 'a', 'b', 'r', 'd' are learnable parameters.

Output:
- best_params.json: Optimized parameters
- optimization_history.png: Optimization visualization
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.stats import spearmanr
from scipy.optimize import minimize, differential_evolution
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'oof_dir': './oof_data/',
    'output_dir': './optimized_params/',
    'n_trials': 500,  # Number of Optuna trials
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
# Utility Functions
# ==========================================
def sigmoid(x):
    """Numerically stable sigmoid"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def compute_spearman(preds, targets):
    """Compute mean Spearman correlation across all targets"""
    scores = []
    for i in range(preds.shape[1]):
        if np.std(preds[:, i]) > 1e-9 and np.std(targets[:, i]) > 1e-9:
            corr = spearmanr(preds[:, i], targets[:, i]).correlation
            if not np.isnan(corr):
                scores.append(corr)
    return np.mean(scores) if scores else 0


def compute_length_dependent_weights(lengths, a, b, roberta_base, deberta_base):
    """
    Compute per-sample weights based on token length.
    
    Args:
        lengths: Array of original token lengths
        a: Slope parameter for sigmoid
        b: Intercept parameter for sigmoid  
        roberta_base: Base weight ratio for RoBERTa
        deberta_base: Base weight ratio for DeBERTa
    
    Returns:
        w_roberta, w_deberta, w_mamba: Arrays of shape (N,)
    """
    # Normalize length for numerical stability
    normalized_lengths = (lengths - 512) / 512  # Center around 512
    
    # Sigmoid determines how much weight goes to Mamba
    w_mamba = sigmoid(a * normalized_lengths + b)
    
    # Remaining weight is split between RoBERTa and DeBERTa
    remaining = 1 - w_mamba
    total_base = roberta_base + deberta_base + 1e-9
    
    w_roberta = remaining * (roberta_base / total_base)
    w_deberta = remaining * (deberta_base / total_base)
    
    return w_roberta, w_deberta, w_mamba


def apply_weighted_ensemble(oof_roberta, oof_deberta, oof_mamba, w_roberta, w_deberta, w_mamba):
    """Apply per-sample weights to create ensemble predictions"""
    # Expand weights to match prediction shape (N, 30)
    w_r = w_roberta[:, np.newaxis]
    w_d = w_deberta[:, np.newaxis]
    w_m = w_mamba[:, np.newaxis]
    
    ensemble = w_r * oof_roberta + w_d * oof_deberta + w_m * oof_mamba
    return ensemble


# ==========================================
# Optuna Optimization
# ==========================================
class WeightOptimizer:
    def __init__(self, oof_roberta, oof_deberta, oof_mamba, targets, lengths):
        self.oof_roberta = oof_roberta
        self.oof_deberta = oof_deberta
        self.oof_mamba = oof_mamba
        self.targets = targets
        self.lengths = lengths
        
    def objective(self, trial):
        """Optuna objective function"""
        # Sample parameters
        a = trial.suggest_float('sigmoid_slope', 0.0, 5.0)  # How quickly Mamba weight increases
        b = trial.suggest_float('sigmoid_intercept', -3.0, 3.0)  # Shift point
        roberta_base = trial.suggest_float('roberta_base', 0.1, 2.0)
        deberta_base = trial.suggest_float('deberta_base', 0.1, 2.0)
        
        # Compute weights
        w_roberta, w_deberta, w_mamba = compute_length_dependent_weights(
            self.lengths, a, b, roberta_base, deberta_base
        )
        
        # Apply ensemble
        ensemble = apply_weighted_ensemble(
            self.oof_roberta, self.oof_deberta, self.oof_mamba,
            w_roberta, w_deberta, w_mamba
        )
        
        # Compute score
        score = compute_spearman(ensemble, self.targets)
        return score


def run_optuna_optimization(oof_roberta, oof_deberta, oof_mamba, targets, lengths, n_trials=500):
    """Run Optuna optimization"""
    print("\n" + "="*60)
    print("Running Optuna Optimization")
    print("="*60)
    
    optimizer = WeightOptimizer(oof_roberta, oof_deberta, oof_mamba, targets, lengths)
    
    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # Optimize
    study.optimize(optimizer.objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n  Best Score: {study.best_value:.5f}")
    print(f"  Best Params: {study.best_params}")
    
    return study


# ==========================================
# Alternative: Scipy Differential Evolution
# ==========================================
def run_scipy_optimization(oof_roberta, oof_deberta, oof_mamba, targets, lengths):
    """Run Scipy differential evolution optimization"""
    print("\n" + "="*60)
    print("Running Scipy Differential Evolution")
    print("="*60)
    
    def objective(params):
        a, b, roberta_base, deberta_base = params
        w_roberta, w_deberta, w_mamba = compute_length_dependent_weights(
            lengths, a, b, roberta_base, deberta_base
        )
        ensemble = apply_weighted_ensemble(
            oof_roberta, oof_deberta, oof_mamba,
            w_roberta, w_deberta, w_mamba
        )
        score = compute_spearman(ensemble, targets)
        return -score  # Minimize negative score
    
    bounds = [
        (0.0, 5.0),    # a: sigmoid_slope
        (-3.0, 3.0),   # b: sigmoid_intercept
        (0.1, 2.0),    # roberta_base
        (0.1, 2.0),    # deberta_base
    ]
    
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=300,
        workers=1,  # Use all CPU cores
        disp=True
    )
    
    best_params = {
        'sigmoid_slope': result.x[0],
        'sigmoid_intercept': result.x[1],
        'roberta_base': result.x[2],
        'deberta_base': result.x[3],
    }
    
    print(f"\n  Best Score: {-result.fun:.5f}")
    print(f"  Best Params: {best_params}")
    
    return best_params, -result.fun


# ==========================================
# Per-Target Optimization (Advanced)
# ==========================================
def optimize_per_target(oof_roberta, oof_deberta, oof_mamba, targets, lengths):
    """Optimize weights separately for each target column"""
    print("\n" + "="*60)
    print("Running Per-Target Optimization")
    print("="*60)
    
    per_target_params = {}
    per_target_scores = []
    
    for i, col in enumerate(CONFIG['target_cols']):
        def objective(params):
            a, b, roberta_base, deberta_base = params
            w_roberta, w_deberta, w_mamba = compute_length_dependent_weights(
                lengths, a, b, roberta_base, deberta_base
            )
            
            # Single column ensemble
            ensemble = (w_roberta * oof_roberta[:, i] + 
                       w_deberta * oof_deberta[:, i] + 
                       w_mamba * oof_mamba[:, i])
            
            if np.std(ensemble) < 1e-9:
                return 0
            
            corr = spearmanr(ensemble, targets[:, i]).correlation
            return -corr if not np.isnan(corr) else 0
        
        bounds = [(0.0, 5.0), (-3.0, 3.0), (0.1, 2.0), (0.1, 2.0)]
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=100, disp=False)
        
        per_target_params[col] = {
            'sigmoid_slope': result.x[0],
            'sigmoid_intercept': result.x[1],
            'roberta_base': result.x[2],
            'deberta_base': result.x[3],
            'score': -result.fun
        }
        per_target_scores.append(-result.fun)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(CONFIG['target_cols'])} targets")
    
    avg_score = np.mean(per_target_scores)
    print(f"\n  Per-Target Average Score: {avg_score:.5f}")
    
    return per_target_params, avg_score


# ==========================================
# Visualization
# ==========================================
def visualize_weights(lengths, best_params, output_dir):
    """Visualize the learned weight function"""
    plt.figure(figsize=(14, 5))
    
    # Sort lengths for plotting
    sorted_idx = np.argsort(lengths)
    sorted_lengths = lengths[sorted_idx]
    
    w_roberta, w_deberta, w_mamba = compute_length_dependent_weights(
        sorted_lengths,
        best_params['sigmoid_slope'],
        best_params['sigmoid_intercept'],
        best_params['roberta_base'],
        best_params['deberta_base']
    )
    
    # Plot 1: Weight distribution by length
    plt.subplot(1, 2, 1)
    plt.plot(sorted_lengths, w_roberta, label='RoBERTa', color='blue', alpha=0.8)
    plt.plot(sorted_lengths, w_deberta, label='DeBERTa', color='green', alpha=0.8)
    plt.plot(sorted_lengths, w_mamba, label='Mamba', color='red', alpha=0.8)
    plt.axvline(x=512, color='gray', linestyle='--', label='BERT Limit (512)')
    plt.xlabel('Original Token Length')
    plt.ylabel('Weight')
    plt.title('Length-Dependent Model Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Token length histogram with weight zones
    plt.subplot(1, 2, 2)
    plt.hist(lengths, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=512, color='red', linestyle='--', label='BERT Limit (512)')
    
    # Find the crossover point where Mamba weight = 0.5
    crossover = (0.5 - best_params['sigmoid_intercept']) / (best_params['sigmoid_slope'] / 512) + 512
    if 0 < crossover < 5000:
        plt.axvline(x=crossover, color='orange', linestyle='--', label=f'Mamba 50% Point ({crossover:.0f})')
    
    plt.xlabel('Original Token Length')
    plt.ylabel('Count')
    plt.title('Token Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_visualization.png'), dpi=150)
    plt.close()
    print(f"  Saved weight visualization to {output_dir}/weight_visualization.png")


# ==========================================
# Main
# ==========================================
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load OOF data
    print("Loading OOF data...")
    oof_roberta = np.load(os.path.join(CONFIG['oof_dir'], 'oof_roberta.npy'))
    oof_deberta = np.load(os.path.join(CONFIG['oof_dir'], 'oof_deberta.npy'))
    oof_mamba = np.load(os.path.join(CONFIG['oof_dir'], 'oof_mamba.npy'))
    targets = np.load(os.path.join(CONFIG['oof_dir'], 'oof_targets.npy'))
    meta = pd.read_csv(os.path.join(CONFIG['oof_dir'], 'meta_features.csv'))
    lengths = meta['original_token_length'].values
    
    print(f"  RoBERTa OOF: {oof_roberta.shape}")
    print(f"  DeBERTa OOF: {oof_deberta.shape}")
    print(f"  Mamba OOF: {oof_mamba.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Baselines
    print("\n" + "="*60)
    print("Baseline Scores")
    print("="*60)
    
    score_r = compute_spearman(oof_roberta, targets)
    score_d = compute_spearman(oof_deberta, targets)
    score_m = compute_spearman(oof_mamba, targets)
    score_simple = compute_spearman((oof_roberta + oof_deberta + oof_mamba) / 3, targets)
    
    print(f"  RoBERTa Only: {score_r:.5f}")
    print(f"  DeBERTa Only: {score_d:.5f}")
    print(f"  Mamba Only: {score_m:.5f}")
    print(f"  Simple Average: {score_simple:.5f}")
    
    # Method 1: Optuna (global params)
    study = run_optuna_optimization(
        oof_roberta, oof_deberta, oof_mamba, targets, lengths,
        n_trials=CONFIG['n_trials']
    )
    optuna_params = study.best_params
    optuna_score = study.best_value
    
    # Method 2: Scipy (global params) - usually faster convergence
    scipy_params, scipy_score = run_scipy_optimization(
        oof_roberta, oof_deberta, oof_mamba, targets, lengths
    )
    
    # Method 3: Per-target optimization
    per_target_params, per_target_score = optimize_per_target(
        oof_roberta, oof_deberta, oof_mamba, targets, lengths
    )
    
    # Compare and select best
    print("\n" + "="*60)
    print("Optimization Summary")
    print("="*60)
    print(f"  Simple Average:     {score_simple:.5f}")
    print(f"  Optuna (Global):    {optuna_score:.5f}")
    print(f"  Scipy (Global):     {scipy_score:.5f}")
    print(f"  Per-Target Average: {per_target_score:.5f}")
    
    # Use best global params
    if optuna_score >= scipy_score:
        best_global_params = optuna_params
        best_global_score = optuna_score
        best_method = 'optuna'
    else:
        best_global_params = scipy_params
        best_global_score = scipy_score
        best_method = 'scipy'
    
    print(f"\n  Best Global Method: {best_method} ({best_global_score:.5f})")
    
    # Save results
    results = {
        'baselines': {
            'roberta': score_r,
            'deberta': score_d,
            'mamba': score_m,
            'simple_average': score_simple
        },
        'global_params': best_global_params,
        'global_score': best_global_score,
        'global_method': best_method,
        'per_target_params': per_target_params,
        'per_target_score': per_target_score,
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'best_params.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved results to {CONFIG['output_dir']}/best_params.json")
    
    # Visualize
    visualize_weights(lengths, best_global_params, CONFIG['output_dir'])
    
    # Show example weights at different lengths
    print("\n" + "="*60)
    print("Example Weights at Different Lengths")
    print("="*60)
    
    example_lengths = np.array([200, 400, 512, 800, 1000, 1500, 2000])
    w_r, w_d, w_m = compute_length_dependent_weights(
        example_lengths,
        best_global_params['sigmoid_slope'],
        best_global_params['sigmoid_intercept'],
        best_global_params['roberta_base'],
        best_global_params['deberta_base']
    )
    
    print(f"  {'Length':>8} | {'RoBERTa':>8} | {'DeBERTa':>8} | {'Mamba':>8}")
    print("  " + "-"*44)
    for i, length in enumerate(example_lengths):
        print(f"  {length:>8} | {w_r[i]:>8.3f} | {w_d[i]:>8.3f} | {w_m[i]:>8.3f}")
    
    print("\n>>> Step 2 Complete! Run step3_train_stacker.py next.")


if __name__ == "__main__":
    main()
