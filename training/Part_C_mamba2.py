import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import os
import sys
import re
import gc
import bitsandbytes as bnb

# ==========================================
# 0. Environment & Config
# ==========================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

CONFIG = {
    'model_name': 'state-spaces/mamba2-780m',
    'tokenizer_name': 'EleutherAI/gpt-neox-20b', 
    'max_len': 2048,          
    'train_batch_size': 8,    
    'valid_batch_size': 16,
    'accumulation_steps': 4,  # Effective Batch = 32
    'epochs': 7,
    'lr': 5e-5,               # [Lowered] For better convergence (was 1e-4)
    'head_lr': 1e-4,          # [New] Higher LR for the new classification head
    'weight_decay': 0.01,               
    'n_folds': 5,
    'num_workers': 8,         
    'device': 'cuda',
    'cat_emb_dim': 16,
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
# 1. Preprocessing (Text Cleaning)
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    
    # 1. Replace Code Blocks (```...```)
    text = re.sub(r'```.*?```', '[CODE]', text, flags=re.DOTALL)
    # 2. Replace Inline Code (`...`)
    text = re.sub(r'`.*?`', '[CODE]', text)
    # 3. Replace LaTeX Math ($...$ or $$...$$)
    text = re.sub(r'\$\$.*?\$\$', '[MATH]', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '[MATH]', text)
    # 4. Replace URLs
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    
    return text

# ==========================================
# 2. Dataset
# ==========================================
class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, cat_encoder):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = df[CONFIG['target_cols']].values
        self.categories = cat_encoder.transform(df['category'].values)
        
        # Mamba/GPT-NeoX separator token
        self.sep_token = tokenizer.eos_token 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        t_title = clean_text(row['question_title'])
        t_body = clean_text(row['question_body'])
        t_answer = clean_text(row['answer'])
        
        # [Strategy Change] Explicitly add separator for Mamba
        # Mamba reads left-to-right. We want: Title + Body + [SEP] + Answer
        text_a = t_title + " " + t_body
        text_b = t_answer
        
        # Manually constructing prompt to ensure SEP is present
        full_text = text_a + " " + self.sep_token + " " + text_b

        inputs = self.tokenizer(
            full_text,
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
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# ==========================================
# 3. Model (Verified 780M Config)
# ==========================================
class QuestMamba(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16, tokenizer_len=None):
        super(QuestMamba, self).__init__()
        
        # 1. Setup Config (Force 48 layers & 50288 vocab)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if '780m' in model_name:
            self.config.hidden_size = 1536
            self.config.d_model = 1536
            self.config.num_heads = 48
            self.config.head_dim = 64
            self.config.num_hidden_layers = 48
            self.config.n_layer = 48
            self.config.state_size = 128
            self.config.n_groups = 1
            self.config.vocab_size = 50288  # MATCHES FILE EXACTLY
            
        # 2. Instantiate Empty Skeleton (Random Weights)
        # We don't use from_pretrained() to load weights yet, just the structure.
        self.backbone = AutoModel.from_config(self.config, trust_remote_code=True)
        
        # 3. Manual Weight Loading (The "Surgical" Fix)
        # Download the bin file manually
        from transformers.utils import cached_file
        print(f"Loading weights manually from {model_name}...")
        bin_path = cached_file(model_name, "pytorch_model.bin")
        state_dict = torch.load(bin_path, map_location="cpu")
        
        # Remap keys: strip "backbone." and fix "embedding" -> "embeddings"
        new_state_dict = {}
        for key, val in state_dict.items():
            new_key = key
            
            # Remove 'backbone.' prefix
            if new_key.startswith("backbone."):
                new_key = new_key.replace("backbone.", "")
                
            # Fix embedding name (embedding -> embeddings)
            if "embedding.weight" in new_key:
                new_key = new_key.replace("embedding.weight", "embeddings.weight")
                
            new_state_dict[new_key] = val

        # Load into model (strict=False allows ignoring head/aux params)
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"Weights Loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # Note: 'Missing' might include 'lm_head' which we don't need. 
        # Crucially, 'embeddings.weight' and 'layers.0...' should NOT be missing.

        # 4. Resize Embeddings (50288 -> 50304 or similar)
        if tokenizer_len is not None:
            current_vocab = self.backbone.embeddings.weight.shape[0]
            if current_vocab != tokenizer_len:
                print(f"Resizing embeddings from {current_vocab} to {tokenizer_len}...")
                self.backbone.resize_token_embeddings(tokenizer_len)
        
        # Enable Gradient Checkpointing
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size = self.config.hidden_size
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.hidden_size + cat_emb_dim
        
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats):
        outputs = self.backbone(input_ids=ids) 
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]
        
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_feature = sum_embeddings / sum_mask 
        
        cat_feature = self.cat_embedding(cats)
        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        
        return torch.cat((out_q, out_a), dim=1)
# ==========================================
# 4. Training Loop
# ==========================================
def compute_spearman(preds, targets):
    score = 0
    cols = preds.shape[1]
    for i in range(cols):
        if np.std(targets[:, i]) < 1e-9 or np.std(preds[:, i]) < 1e-9:
            score += 0.5 
        else:
            score += spearmanr(targets[:, i], preds[:, i]).correlation
    return score / cols

def run_fold(fold, train_idx, val_idx, df, tokenizer, cat_encoder):
    print(f"\n=== Running Fold {fold+1}/{CONFIG['n_folds']} (Mamba2-780M) ===")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_dataset = QuestDataset(train_df, tokenizer, CONFIG['max_len'], cat_encoder)
    val_dataset = QuestDataset(val_df, tokenizer, CONFIG['max_len'], cat_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    num_cats = len(cat_encoder.classes_)
    
    # [FIX] Pass tokenizer_len to trigger the resize logic
    model = QuestMamba(
        CONFIG['model_name'], 
        num_cats=num_cats, 
        cat_emb_dim=CONFIG['cat_emb_dim'],
        tokenizer_len=len(tokenizer)
    )
    model = model.to(CONFIG['device'])
    
    # [OPTIMIZATION] Layer-wise Learning Rate / Differential LR
    # Give the pre-trained backbone a smaller LR, and the new heads a larger LR
    optimizer_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters()],
            "lr": CONFIG['lr'],          # 5e-5
            "weight_decay": CONFIG['weight_decay']
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": CONFIG['head_lr'],     # 1e-4
            "weight_decay": CONFIG['weight_decay']
        }
    ]
    
    optimizer = bnb.optim.AdamW8bit(optimizer_parameters)
    
    num_train_steps = int(len(train_df) / CONFIG['train_batch_size'] / CONFIG['accumulation_steps'] * CONFIG['epochs'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # [FIX] Modern AMP Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    best_score = -1
    save_dir = './models_mamba'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for step, data in enumerate(train_loader):
            ids = data['ids'].to(CONFIG['device'])
            mask = data['mask'].to(CONFIG['device'])
            cats = data['cats'].to(CONFIG['device'])
            targets = data['targets'].to(CONFIG['device'])
            
            # [FIX] Modern Autocast
            with torch.amp.autocast('cuda'):
                outputs = model(ids, mask, cats)
                loss = criterion(outputs, targets)
                loss = loss / CONFIG['accumulation_steps']
            
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * CONFIG['accumulation_steps']
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for data in val_loader:
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                targets = data['targets'].to(CONFIG['device'])
                
                with torch.amp.autocast('cuda'):
                    outputs = model(ids, mask, cats)
                
                outputs = torch.sigmoid(outputs)
                val_preds.append(outputs.cpu().float().numpy())
                val_targets.append(targets.cpu().float().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        score = compute_spearman(val_preds, val_targets)
        
        current_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {current_loss:.4f} | Val Spearman: {score:.4f}")
        
        if score > best_score:
            best_score = score
            save_path = f"{save_dir}/model_fold{fold}.bin"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best Saved: {best_score:.4f}")
            
    return best_score

if __name__ == "__main__":
    if not os.path.exists(r'data\train.csv'):
        print("Please upload train.csv")
        sys.exit(1)

    df = pd.read_csv(r'data\train.csv')
    le = LabelEncoder()
    le.fit(df['category'])
    
    print(f"Loading Tokenizer: {CONFIG['tokenizer_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    groups = df['question_body']
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        score = run_fold(fold, train_idx, val_idx, df, tokenizer, le)
        fold_scores.append(score)
        
        print(f"Cleaning up Fold {fold+1} memory...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("Memory cleared.")
    
    print(f"\n>>> Average Mamba Score: {np.mean(fold_scores):.4f}")