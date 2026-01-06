"""
Step 1: Generate OOF (Out-of-Fold) Predictions
===============================================
Run this locally with your 4090 GPU.

This script loads your trained models and generates OOF predictions
that will be used for weight optimization and stacking.

Output files:
- oof_roberta.npy: (N, 30) array of RoBERTa OOF predictions
- oof_deberta.npy: (N, 30) array of DeBERTa OOF predictions  
- oof_mamba.npy: (N, 30) array of Mamba OOF predictions
- meta_features.csv: Meta features for each sample
- oof_targets.npy: Ground truth labels
"""

import os
# Suppress warnings BEFORE importing other libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Fix memory fragmentation

import warnings
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message=".*tokenizers.*")

import gc  # Add garbage collection
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, logging as hf_logging
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'roberta_model_name': 'roberta-base',
    'deberta_model_name': 'microsoft/deberta-v3-base',
    'mamba_model_name': 'state-spaces/mamba-1.4b-hf',
    
    'roberta_max_len': 512,
    'deberta_max_len': 512,
    'mamba_max_len': 2048,
    
    'batch_size': 32,
    'mamba_batch_size': 4,  # Reduced for 1.4B model memory
    'n_folds': 5,
    'num_workers': 0,  # Set to 0 to avoid tokenizer forking warnings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cat_emb_dim': 16,
    
    # Paths - MODIFY THESE TO YOUR LOCAL PATHS
    'train_data_path': './train.csv',
    'roberta_weights_dir': './models_Roberta/',
    'deberta_weights_dir': './models_deberta/',
    'mamba_weights_dir': './models_mamba/',
    'output_dir': './oof_data/',
    
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
# Dataset Classes
# ==========================================
class QuestDataset(Dataset):
    """Generic dataset for RoBERTa/DeBERTa"""
    def __init__(self, df, tokenizer, max_len, cat_encoder, return_token_type_ids=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cat_encoder = cat_encoder
        self.return_token_type_ids = return_token_type_ids
        self.categories = cat_encoder.transform(df['category'].values)
        
        # Store original token lengths for meta features
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

        result = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(self.categories[idx], dtype=torch.long),
        }
        
        if self.return_token_type_ids and 'token_type_ids' in inputs:
            result['token_type_ids'] = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            
        return result


class MambaDataset(Dataset):
    """Dataset for Mamba (no token_type_ids)"""
    def __init__(self, df, tokenizer, max_len, cat_encoder):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.categories = cat_encoder.transform(df['category'].values)

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
            return_token_type_ids=False
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(self.categories[idx], dtype=torch.long),
        }


# ==========================================
# Model Classes
# ==========================================
class QuestRoberta(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestRoberta, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats, token_type_ids=None):
        outputs = self.backbone(ids, attention_mask=mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_feature = sum_embeddings / sum_mask
        
        cat_feature = self.cat_embedding(cats)
        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        return torch.cat((out_q, out_a), dim=1)


class QuestDeberta(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestDeberta, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, token_type_ids, cats):
        outputs = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)
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
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestMamba, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
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
# OOF Generation Functions
# ==========================================
def generate_oof_roberta(df, cat_encoder, gkf, groups):
    """Generate OOF predictions for RoBERTa models"""
    print("\n" + "="*60)
    print("Generating RoBERTa OOF Predictions")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['roberta_model_name'])
    num_cats = len(cat_encoder.classes_)
    
    oof_preds = np.zeros((len(df), 30))
    original_lengths = None
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        print(f"\nFold {fold + 1}/{CONFIG['n_folds']}")
        
        weight_path = os.path.join(CONFIG['roberta_weights_dir'], f'model_fold{fold}.bin')
        if not os.path.exists(weight_path):
            print(f"  Warning: {weight_path} not found, skipping...")
            continue
        
        val_df = df.iloc[val_idx]
        val_dataset = QuestDataset(val_df, tokenizer, CONFIG['roberta_max_len'], cat_encoder, return_token_type_ids=False)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        
        # Store original lengths from first fold
        if original_lengths is None:
            full_dataset = QuestDataset(df, tokenizer, CONFIG['roberta_max_len'], cat_encoder, return_token_type_ids=False)
            original_lengths = full_dataset.original_lengths
        
        # Load model
        model = QuestRoberta(CONFIG['roberta_model_name'], num_cats, CONFIG['cat_emb_dim'])
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"  Predicting"):
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                
                outputs = model(ids, mask, cats)
                outputs = torch.sigmoid(outputs)
                fold_preds.append(outputs.cpu().numpy())
        
        fold_preds = np.concatenate(fold_preds)
        oof_preds[val_idx] = fold_preds
        
        del model
        torch.cuda.empty_cache()
    
    return oof_preds, original_lengths


def generate_oof_deberta(df, cat_encoder, gkf, groups):
    """Generate OOF predictions for DeBERTa models"""
    print("\n" + "="*60)
    print("Generating DeBERTa OOF Predictions")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['deberta_model_name'])
    num_cats = len(cat_encoder.classes_)
    
    oof_preds = np.zeros((len(df), 30))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        print(f"\nFold {fold + 1}/{CONFIG['n_folds']}")
        
        weight_path = os.path.join(CONFIG['deberta_weights_dir'], f'model_fold{fold}.bin')
        if not os.path.exists(weight_path):
            print(f"  Warning: {weight_path} not found, skipping...")
            continue
        
        val_df = df.iloc[val_idx]
        val_dataset = QuestDataset(val_df, tokenizer, CONFIG['deberta_max_len'], cat_encoder, return_token_type_ids=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        
        model = QuestDeberta(CONFIG['deberta_model_name'], num_cats, CONFIG['cat_emb_dim'])
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"  Predicting"):
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                token_type_ids = data['token_type_ids'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                
                outputs = model(ids, mask, token_type_ids, cats)
                outputs = torch.sigmoid(outputs)
                fold_preds.append(outputs.cpu().numpy())
        
        fold_preds = np.concatenate(fold_preds)
        oof_preds[val_idx] = fold_preds
        
        del model
        torch.cuda.empty_cache()
    
    return oof_preds


def generate_oof_mamba(df, cat_encoder, gkf, groups):
    """Generate OOF predictions for Mamba models"""
    print("\n" + "="*60)
    print("Generating Mamba OOF Predictions")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['mamba_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_cats = len(cat_encoder.classes_)
    
    oof_preds = np.zeros((len(df), 30))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        print(f"\nFold {fold + 1}/{CONFIG['n_folds']}")
        
        # Clean memory BEFORE loading new model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        weight_path = os.path.join(CONFIG['mamba_weights_dir'], f'model_fold{fold}.bin')
        if not os.path.exists(weight_path):
            print(f"  Warning: {weight_path} not found, skipping...")
            continue
        
        val_df = df.iloc[val_idx]
        val_dataset = MambaDataset(val_df, tokenizer, CONFIG['mamba_max_len'], cat_encoder)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['mamba_batch_size'], shuffle=False, num_workers=0, pin_memory=False)
        
        # Load model with explicit device mapping
        model = QuestMamba(CONFIG['mamba_model_name'], num_cats, CONFIG['cat_emb_dim'])
        state_dict = torch.load(weight_path, map_location='cpu')  # Load to CPU first
        model.load_state_dict(state_dict)
        del state_dict  # Free state dict memory
        gc.collect()
        
        model.to(CONFIG['device'])
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"  Predicting"):
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                
                outputs = model(ids, mask, cats)
                outputs = torch.sigmoid(outputs)
                fold_preds.append(outputs.cpu().numpy())
                
                # Clear intermediate tensors
                del ids, mask, cats, outputs
        
        fold_preds = np.concatenate(fold_preds)
        oof_preds[val_idx] = fold_preds
        
        # Aggressive cleanup after each fold
        del model, val_loader, val_dataset, fold_preds
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  Memory cleaned. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return oof_preds


def compute_meta_features(df, original_lengths, cat_encoder):
    """Compute meta features for stacking"""
    print("\n" + "="*60)
    print("Computing Meta Features")
    print("="*60)
    
    meta = pd.DataFrame()
    meta['original_token_length'] = original_lengths
    meta['exceeds_512'] = (meta['original_token_length'] > 512).astype(int)
    meta['exceeds_1024'] = (meta['original_token_length'] > 1024).astype(int)
    meta['log_token_length'] = np.log1p(meta['original_token_length'])
    
    # Question/Answer length ratio
    q_lens = df.apply(lambda x: len(str(x['question_title']) + str(x['question_body'])), axis=1)
    a_lens = df.apply(lambda x: len(str(x['answer'])), axis=1)
    meta['q_a_ratio'] = q_lens / (a_lens + 1)
    meta['q_len'] = q_lens
    meta['a_len'] = a_lens
    
    # Category encoding
    meta['category'] = cat_encoder.transform(df['category'].values)
    
    # One-hot category
    for i, cat in enumerate(cat_encoder.classes_):
        meta[f'cat_{cat}'] = (df['category'] == cat).astype(int)
    
    print(f"  Meta features shape: {meta.shape}")
    return meta


# ==========================================
# Main
# ==========================================
def main():
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading training data...")
    df = pd.read_csv(CONFIG['train_data_path'])
    print(f"  Loaded {len(df)} samples")
    
    # Setup encoder and fold splitter
    cat_encoder = LabelEncoder()
    cat_encoder.fit(df['category'])
    print(f"  Categories: {cat_encoder.classes_}")
    
    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    groups = df['question_body']
    
    # Generate OOF predictions (with resume capability)
    roberta_path = os.path.join(CONFIG['output_dir'], 'oof_roberta.npy')
    deberta_path = os.path.join(CONFIG['output_dir'], 'oof_deberta.npy')
    mamba_path = os.path.join(CONFIG['output_dir'], 'oof_mamba.npy')
    
    # RoBERTa
    if os.path.exists(roberta_path):
        print(f"\n>>> Loading existing RoBERTa OOF from {roberta_path}")
        oof_roberta = np.load(roberta_path)
        # Still need to compute original_lengths
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['roberta_model_name'])
        original_lengths = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = str(row['question_title']) + " " + str(row['question_body']) + " " + str(row['answer'])
            tokens = tokenizer.encode(text, add_special_tokens=False)
            original_lengths.append(len(tokens))
    else:
        oof_roberta, original_lengths = generate_oof_roberta(df, cat_encoder, gkf, groups)
        np.save(roberta_path, oof_roberta)
        print(f"  Saved RoBERTa OOF to {roberta_path}")
    
    # DeBERTa
    if os.path.exists(deberta_path):
        print(f"\n>>> Loading existing DeBERTa OOF from {deberta_path}")
        oof_deberta = np.load(deberta_path)
    else:
        oof_deberta = generate_oof_deberta(df, cat_encoder, gkf, groups)
        np.save(deberta_path, oof_deberta)
        print(f"  Saved DeBERTa OOF to {deberta_path}")
    
    # Mamba - Clean memory before running
    gc.collect()
    torch.cuda.empty_cache()
    
    if os.path.exists(mamba_path):
        print(f"\n>>> Loading existing Mamba OOF from {mamba_path}")
        oof_mamba = np.load(mamba_path)
    else:
        oof_mamba = generate_oof_mamba(df, cat_encoder, gkf, groups)
        np.save(mamba_path, oof_mamba)
        print(f"  Saved Mamba OOF to {mamba_path}")
    
    # Compute meta features
    meta_features = compute_meta_features(df, original_lengths, cat_encoder)
    
    # Save remaining files (targets, meta, encoder)
    print("\n" + "="*60)
    print("Saving Additional Data")
    print("="*60)
    
    np.save(os.path.join(CONFIG['output_dir'], 'oof_targets.npy'), df[CONFIG['target_cols']].values)
    meta_features.to_csv(os.path.join(CONFIG['output_dir'], 'meta_features.csv'), index=False)
    
    # Save category encoder classes
    with open(os.path.join(CONFIG['output_dir'], 'cat_classes.json'), 'w') as f:
        json.dump(list(cat_encoder.classes_), f)
    
    print(f"\n  Saved to {CONFIG['output_dir']}:")
    print(f"    - oof_roberta.npy: {oof_roberta.shape}")
    print(f"    - oof_deberta.npy: {oof_deberta.shape}")
    print(f"    - oof_mamba.npy: {oof_mamba.shape}")
    print(f"    - oof_targets.npy: {df[CONFIG['target_cols']].values.shape}")
    print(f"    - meta_features.csv: {meta_features.shape}")
    
    # Quick validation
    from scipy.stats import spearmanr
    targets = df[CONFIG['target_cols']].values
    
    print("\n" + "="*60)
    print("Quick Validation (Mean Spearman)")
    print("="*60)
    
    def calc_spearman(preds, targets):
        scores = []
        for i in range(30):
            if np.std(preds[:, i]) > 1e-9 and np.std(targets[:, i]) > 1e-9:
                scores.append(spearmanr(preds[:, i], targets[:, i]).correlation)
        return np.mean(scores)
    
    print(f"  RoBERTa OOF Spearman: {calc_spearman(oof_roberta, targets):.4f}")
    print(f"  DeBERTa OOF Spearman: {calc_spearman(oof_deberta, targets):.4f}")
    print(f"  Mamba OOF Spearman: {calc_spearman(oof_mamba, targets):.4f}")
    
    # Simple average
    simple_avg = (oof_roberta + oof_deberta + oof_mamba) / 3
    print(f"  Simple Average Spearman: {calc_spearman(simple_avg, targets):.4f}")
    
    print("\n>>> Step 1 Complete! Run step2_optimize_weights.py next.")


if __name__ == "__main__":
    main()