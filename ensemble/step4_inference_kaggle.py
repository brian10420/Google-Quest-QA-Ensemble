"""
Step 4: Final Inference for Kaggle
==================================
Upload this script to Kaggle along with:
- Trained model weights (.bin files)
- Optimized parameters (best_params.json)
- Stacker models (ridge_models/models.pkl, lgb_models/models.pkl)

This script combines:
1. Length-dependent weighted ensemble
2. Ridge + LightGBM stacking
3. Rank-based post-processing
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import sys
import glob
import json
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. Environment Setup
# ==========================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. Path Configuration (MODIFY FOR KAGGLE)
# ==========================================
# Model configs
ROBERTA_CONFIG_PATH = '/kaggle/input/roberta-config/roberta_base_config/'
DEBERTA_CONFIG_PATH = '/kaggle/input/deberta-v3-base-config/deberta_v3_base_config/'
MAMBA_CONFIG_PATH = '/kaggle/input/mamba-config/mamba_1.4b_config/'  # Adjust based on your upload

# Model weights
ROBERTA_WEIGHTS_DIR = '/kaggle/input/models-roberta/'
DEBERTA_WEIGHTS_DIR = '/kaggle/input/quest-deberta-models/'
MAMBA_WEIGHTS_DIR = '/kaggle/input/models-mamba/'  # Upload your mamba models

# Optimization params and stacker models
PARAMS_PATH = '/kaggle/input/quest-optimized-params/best_params.json'  # Upload this
RIDGE_MODELS_PATH = '/kaggle/input/quest-stacker-models/ridge_models/models.pkl'
LGB_MODELS_PATH = '/kaggle/input/quest-stacker-models/lgb_models/models.pkl'

# Competition data
TEST_DATA_PATH = '/kaggle/input/google-quest-challenge/test.csv'
TRAIN_DATA_PATH = '/kaggle/input/google-quest-challenge/train.csv'

# ==========================================
# 2. Config
# ==========================================
CONFIG = {
    'roberta_max_len': 512,
    'deberta_max_len': 512,
    'mamba_max_len': 2048,
    'batch_size': 32,
    'mamba_batch_size': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cat_emb_dim': 16,
    'num_workers': 2,
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
# 3. Dataset Classes
# ==========================================
class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, cat_map, return_token_type_ids=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cat_map = cat_map
        self.return_token_type_ids = return_token_type_ids
        
        # Compute original token lengths
        self.original_lengths = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = str(row['question_title']) + " " + str(row['question_body']) + " " + str(row['answer'])
            tokens = tokenizer.encode(text, add_special_tokens=False)
            self.original_lengths.append(len(tokens))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_a = str(row['question_title']) + " " + str(row['question_body'])
        text_b = str(row['answer'])
        
        inputs = self.tokenizer.encode_plus(
            text_a, text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=self.return_token_type_ids
        )
        
        cat_val = self.cat_map.get(row['category'], 0)
        
        result = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(cat_val, dtype=torch.long)
        }
        
        if self.return_token_type_ids and 'token_type_ids' in inputs:
            result['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        
        return result


# ==========================================
# 4. Model Classes
# ==========================================
class QuestModel(nn.Module):
    """Generic model for RoBERTa/DeBERTa"""
    def __init__(self, config_path, num_cats, cat_emb_dim=16):
        super(QuestModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config_path)
        self.backbone = AutoModel.from_config(self.config)
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)
        else:
            outputs = self.backbone(ids, attention_mask=mask)
        
        last_hidden_state = outputs.last_hidden_state
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_feature = sum_embeddings / sum_mask
        
        cat_feature = self.cat_embedding(cats)
        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        return torch.cat((out_q, out_a), dim=1)


class QuestMamba(nn.Module):
    """Mamba model (no token_type_ids)"""
    def __init__(self, config_path, num_cats, cat_emb_dim=16):
        super(QuestMamba, self).__init__()
        self.config = AutoConfig.from_pretrained(config_path)
        self.backbone = AutoModel.from_config(self.config)
        self.hidden_size = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.d_model
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats):
        outputs = self.backbone(input_ids=ids)
        last_hidden_state = outputs.last_hidden_state
        
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_feature = sum_embeddings / sum_mask
        
        cat_feature = self.cat_embedding(cats)
        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        return torch.cat((out_q, out_a), dim=1)


# ==========================================
# 5. Weight Functions
# ==========================================
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_length_dependent_weights(lengths, params):
    """Compute per-sample weights based on optimized parameters"""
    normalized_lengths = (lengths - 512) / 512
    
    w_mamba = sigmoid(params['sigmoid_slope'] * normalized_lengths + params['sigmoid_intercept'])
    remaining = 1 - w_mamba
    total_base = params['roberta_base'] + params['deberta_base'] + 1e-9
    
    w_roberta = remaining * (params['roberta_base'] / total_base)
    w_deberta = remaining * (params['deberta_base'] / total_base)
    
    return w_roberta, w_deberta, w_mamba


def apply_weighted_ensemble(preds_r, preds_d, preds_m, w_r, w_d, w_m):
    """Apply per-sample weights"""
    return (w_r[:, np.newaxis] * preds_r + 
            w_d[:, np.newaxis] * preds_d + 
            w_m[:, np.newaxis] * preds_m)


# ==========================================
# 6. Post-Processing
# ==========================================
def get_best_buckets(train_df, col):
    values = train_df[col].values
    unique_vals = np.sort(np.unique(values))
    counts = {v: np.sum(values == v) for v in unique_vals}
    total = len(values)
    mapping = []
    cum_prop = 0
    for v in unique_vals:
        prop = counts[v] / total
        cum_prop += prop
        mapping.append({'val': v, 'cum_prop': cum_prop})
    return mapping


def apply_buckets(preds, mapping):
    n = len(preds)
    sorted_idx = np.argsort(preds)
    new_preds = np.zeros(n)
    start_idx = 0
    for m in mapping:
        end_idx = int(m['cum_prop'] * n)
        end_idx = min(end_idx, n)
        if end_idx > start_idx:
            new_preds[sorted_idx[start_idx:end_idx]] = m['val']
        start_idx = end_idx
    if start_idx < n:
        new_preds[sorted_idx[start_idx:]] = mapping[-1]['val']
    return new_preds


# ==========================================
# 7. Stacker Features
# ==========================================
def create_stacker_features(preds_r, preds_d, preds_m, target_idx, meta_features):
    """Create features for stacker inference"""
    r_pred = preds_r[:, target_idx:target_idx+1]
    d_pred = preds_d[:, target_idx:target_idx+1]
    m_pred = preds_m[:, target_idx:target_idx+1]
    
    all_preds = np.stack([r_pred[:, 0], d_pred[:, 0], m_pred[:, 0]], axis=1)
    target_std = np.std(all_preds, axis=1, keepdims=True)
    target_range = (np.max(all_preds, axis=1) - np.min(all_preds, axis=1)).reshape(-1, 1)
    target_mean = np.mean(all_preds, axis=1, keepdims=True)
    
    meta_array = meta_features[['original_token_length', 'log_token_length',
                                'exceeds_512', 'exceeds_1024',
                                'q_a_ratio', 'q_len', 'a_len', 'category']].values
    
    cat_cols = [c for c in meta_features.columns if c.startswith('cat_')]
    if cat_cols:
        cat_onehot = meta_features[cat_cols].values
        meta_array = np.hstack([meta_array, cat_onehot])
    
    features = np.hstack([
        r_pred, d_pred, m_pred,
        target_std, target_range, target_mean,
        meta_array
    ])
    
    return features


def compute_meta_features(test_df, original_lengths, cat_map):
    """Compute meta features for test data"""
    meta = pd.DataFrame()
    meta['original_token_length'] = original_lengths
    meta['exceeds_512'] = (meta['original_token_length'] > 512).astype(int)
    meta['exceeds_1024'] = (meta['original_token_length'] > 1024).astype(int)
    meta['log_token_length'] = np.log1p(meta['original_token_length'])
    
    q_lens = test_df.apply(lambda x: len(str(x['question_title']) + str(x['question_body'])), axis=1)
    a_lens = test_df.apply(lambda x: len(str(x['answer'])), axis=1)
    meta['q_a_ratio'] = q_lens / (a_lens + 1)
    meta['q_len'] = q_lens
    meta['a_len'] = a_lens
    
    meta['category'] = test_df['category'].map(cat_map).fillna(0).astype(int)
    
    # One-hot category
    for cat, idx in cat_map.items():
        meta[f'cat_{cat}'] = (test_df['category'] == cat).astype(int)
    
    return meta


# ==========================================
# 8. Prediction Functions
# ==========================================
def predict_roberta(test_df, cat_map, num_cats):
    """Get RoBERTa predictions"""
    if not os.path.exists(ROBERTA_WEIGHTS_DIR):
        return None, None
    
    weight_files = sorted(glob.glob(os.path.join(ROBERTA_WEIGHTS_DIR, '*.bin')))
    if len(weight_files) == 0:
        return None, None
    
    print(f"\n>>> Predicting with {len(weight_files)} RoBERTa models")
    
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CONFIG_PATH)
    dataset = QuestDataset(test_df, tokenizer, CONFIG['roberta_max_len'], cat_map, return_token_type_ids=False)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    all_preds = np.zeros((len(test_df), 30))
    
    for f in weight_files:
        print(f"  Processing: {os.path.basename(f)}")
        model = QuestModel(ROBERTA_CONFIG_PATH, num_cats, CONFIG['cat_emb_dim'])
        model.load_state_dict(torch.load(f, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in loader:
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                out = model(ids, mask, cats)
                fold_preds.append(torch.sigmoid(out).cpu().numpy())
        
        all_preds += np.concatenate(fold_preds)
        del model
        torch.cuda.empty_cache()
    
    all_preds /= len(weight_files)
    return all_preds, dataset.original_lengths


def predict_deberta(test_df, cat_map, num_cats):
    """Get DeBERTa predictions"""
    if not os.path.exists(DEBERTA_WEIGHTS_DIR):
        return None
    
    weight_files = sorted(glob.glob(os.path.join(DEBERTA_WEIGHTS_DIR, '*.bin')))
    if len(weight_files) == 0:
        return None
    
    print(f"\n>>> Predicting with {len(weight_files)} DeBERTa models")
    
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_CONFIG_PATH, use_fast=False)
    dataset = QuestDataset(test_df, tokenizer, CONFIG['deberta_max_len'], cat_map, return_token_type_ids=True)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    all_preds = np.zeros((len(test_df), 30))
    
    for f in weight_files:
        print(f"  Processing: {os.path.basename(f)}")
        model = QuestModel(DEBERTA_CONFIG_PATH, num_cats, CONFIG['cat_emb_dim'])
        model.load_state_dict(torch.load(f, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in loader:
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                token_type_ids = data['token_type_ids'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                out = model(ids, mask, cats, token_type_ids)
                fold_preds.append(torch.sigmoid(out).cpu().numpy())
        
        all_preds += np.concatenate(fold_preds)
        del model
        torch.cuda.empty_cache()
    
    all_preds /= len(weight_files)
    return all_preds


def predict_mamba(test_df, cat_map, num_cats):
    """Get Mamba predictions"""
    if not os.path.exists(MAMBA_WEIGHTS_DIR):
        return None
    
    weight_files = sorted(glob.glob(os.path.join(MAMBA_WEIGHTS_DIR, '*.bin')))
    if len(weight_files) == 0:
        return None
    
    print(f"\n>>> Predicting with {len(weight_files)} Mamba models")
    
    tokenizer = AutoTokenizer.from_pretrained(MAMBA_CONFIG_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = QuestDataset(test_df, tokenizer, CONFIG['mamba_max_len'], cat_map, return_token_type_ids=False)
    loader = DataLoader(dataset, batch_size=CONFIG['mamba_batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    all_preds = np.zeros((len(test_df), 30))
    
    for f in weight_files:
        print(f"  Processing: {os.path.basename(f)}")
        model = QuestMamba(MAMBA_CONFIG_PATH, num_cats, CONFIG['cat_emb_dim'])
        model.load_state_dict(torch.load(f, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in loader:
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                out = model(ids, mask, cats)
                fold_preds.append(torch.sigmoid(out).cpu().numpy())
        
        all_preds += np.concatenate(fold_preds)
        del model
        torch.cuda.empty_cache()
    
    all_preds /= len(weight_files)
    return all_preds


# ==========================================
# 9. Main Inference
# ==========================================
def main():
    print(f"Device: {CONFIG['device']}")
    
    # Load data
    test_df = pd.read_csv(TEST_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH) if os.path.exists(TRAIN_DATA_PATH) else None
    
    # Build category map
    if train_df is not None:
        unique_cats = sorted(train_df['category'].unique())
        cat_map = {c: i for i, c in enumerate(unique_cats)}
    else:
        cat_map = {'CULTURE': 0, 'LIFE_ARTS': 1, 'SCIENCE': 2, 'STACKOVERFLOW': 3, 'TECHNOLOGY': 4}
    
    num_cats = len(cat_map)
    print(f"Categories: {cat_map}")
    
    # Get predictions from all models
    preds_roberta, original_lengths = predict_roberta(test_df, cat_map, num_cats)
    preds_deberta = predict_deberta(test_df, cat_map, num_cats)
    preds_mamba = predict_mamba(test_df, cat_map, num_cats)
    
    # Check which models are available
    available_models = []
    if preds_roberta is not None:
        available_models.append(('roberta', preds_roberta))
    if preds_deberta is not None:
        available_models.append(('deberta', preds_deberta))
    if preds_mamba is not None:
        available_models.append(('mamba', preds_mamba))
    
    if len(available_models) == 0:
        print("ERROR: No model predictions available!")
        return
    
    print(f"\n>>> Available models: {[m[0] for m in available_models]}")
    
    # ===============================
    # Strategy 1: Length-Weighted Ensemble (if we have all 3 models)
    # ===============================
    if len(available_models) == 3 and os.path.exists(PARAMS_PATH):
        print("\n>>> Applying length-dependent weighted ensemble...")
        
        with open(PARAMS_PATH, 'r') as f:
            params = json.load(f)
        
        global_params = params['global_params']
        lengths = np.array(original_lengths)
        
        w_r, w_d, w_m = compute_length_dependent_weights(lengths, global_params)
        weighted_preds = apply_weighted_ensemble(preds_roberta, preds_deberta, preds_mamba, w_r, w_d, w_m)
        
        print(f"  Weight ranges: R=[{w_r.min():.3f}, {w_r.max():.3f}], "
              f"D=[{w_d.min():.3f}, {w_d.max():.3f}], M=[{w_m.min():.3f}, {w_m.max():.3f}]")
    else:
        # Fallback: simple average
        print("\n>>> Using simple average ensemble...")
        weighted_preds = np.mean([p for _, p in available_models], axis=0)
    
    # ===============================
    # Strategy 2: Stacker Ensemble (if available)
    # ===============================
    stacker_preds = None
    
    if (len(available_models) == 3 and 
        os.path.exists(RIDGE_MODELS_PATH) and 
        os.path.exists(LGB_MODELS_PATH)):
        
        print("\n>>> Applying stacker ensemble...")
        
        # Compute meta features
        meta_features = compute_meta_features(test_df, original_lengths, cat_map)
        
        # Load stacker models
        with open(RIDGE_MODELS_PATH, 'rb') as f:
            ridge_data = pickle.load(f)
        with open(LGB_MODELS_PATH, 'rb') as f:
            lgb_models = pickle.load(f)
        
        ridge_models = ridge_data['models']
        ridge_scalers = ridge_data['scalers']
        
        # Predict with stackers
        stacker_preds = np.zeros((len(test_df), 30))
        
        for i, col in enumerate(CONFIG['target_cols']):
            # Create features for this target
            X = create_stacker_features(preds_roberta, preds_deberta, preds_mamba, i, meta_features)
            
            # Ridge predictions (average across folds)
            ridge_pred = np.zeros(len(X))
            for model, scaler in zip(ridge_models[col], ridge_scalers[col]):
                X_scaled = scaler.transform(X)
                ridge_pred += model.predict(X_scaled)
            ridge_pred /= len(ridge_models[col])
            
            # LightGBM predictions (average across folds)
            lgb_pred = np.zeros(len(X))
            for model in lgb_models[col]:
                lgb_pred += model.predict(X)
            lgb_pred /= len(lgb_models[col])
            
            # Ensemble Ridge + LGB
            stacker_preds[:, i] = (ridge_pred + lgb_pred) / 2
        
        # Clip to valid range
        stacker_preds = np.clip(stacker_preds, 0, 1)
        print("  Stacker predictions complete.")
    
    # ===============================
    # Combine All Strategies
    # ===============================
    if stacker_preds is not None:
        # Final: Average of weighted ensemble and stacker
        final_preds = (weighted_preds + stacker_preds) / 2
        print("\n>>> Final ensemble: (Weighted + Stacker) / 2")
    else:
        final_preds = weighted_preds
        print("\n>>> Final: Weighted ensemble only")
    
    # ===============================
    # Post-Processing
    # ===============================
    if train_df is not None:
        print("\n>>> Applying rank-based post-processing...")
        optimized_preds = final_preds.copy()
        
        for i, col in enumerate(CONFIG['target_cols']):
            mapping = get_best_buckets(train_df, col)
            optimized_preds[:, i] = apply_buckets(final_preds[:, i], mapping)
        
        final_preds = optimized_preds
    
    # ===============================
    # Create Submission
    # ===============================
    submission = pd.DataFrame(final_preds, columns=CONFIG['target_cols'])
    submission.insert(0, 'qa_id', test_df['qa_id'])
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "="*60)
    print(">>> submission.csv saved successfully!")
    print("="*60)
    
    # Show prediction statistics
    print(f"\nPrediction statistics:")
    print(f"  Shape: {final_preds.shape}")
    print(f"  Min: {final_preds.min():.4f}")
    print(f"  Max: {final_preds.max():.4f}")
    print(f"  Mean: {final_preds.mean():.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    sys.exit(0)
