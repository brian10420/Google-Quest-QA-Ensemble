import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# [新增] 混合精度訓練工具 (4090 必備)
from torch.cuda.amp import autocast, GradScaler 
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import os
import sys

# --- Config (針對 4090 與 DeBERTa 優化) ---
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base', # [關鍵] 改用 DeBERTa
    'max_len': 512,          
    'train_batch_size': 8,  # [關鍵] 4090 設 16 很穩，想更快可試 24
    'valid_batch_size': 32,  # 驗證可以大一點
    'epochs': 8,             # DeBERTa 建議多跑一點 (原本是 3)
    'lr': 2e-5,
    'n_folds': 5,            # [關鍵] 若要衝高分，建議跑滿 5 folds
    'num_workers': 8,        # 加速資料讀取
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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

# --- Dataset (維持不變) ---
class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, cat_encoder):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = df[CONFIG['target_cols']].values
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
            return_token_type_ids=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'cats': torch.tensor(self.categories[idx], dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# --- Model (Method D: DeBERTa + Category) ---
class QuestDeberta(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestDeberta, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)
        
        # 初始化權重，幫助收斂
        torch.nn.init.xavier_uniform_(self.head_q.weight)
        torch.nn.init.xavier_uniform_(self.head_a.weight)

    def forward(self, ids, mask, token_type_ids, cats):
        # DeBERTa forward
        outputs = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_feature = sum_embeddings / sum_mask 
        
        # Concat Category
        cat_feature = self.cat_embedding(cats)
        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        
        return torch.cat((out_q, out_a), dim=1)

# --- Metric ---
def compute_spearman(preds, targets):
    score = 0
    cols = preds.shape[1]
    for i in range(cols):
        if np.std(targets[:, i]) < 1e-9 or np.std(preds[:, i]) < 1e-9:
            score += 0 
        else:
            score += spearmanr(targets[:, i], preds[:, i]).correlation
    return score / cols

# --- Training Loop (加入 AMP 與 Scheduler) ---
def run_fold(fold, train_idx, val_idx, df, tokenizer, cat_encoder):
    print(f"\n=== Running Fold {fold+1}/{CONFIG['n_folds']} (DeBERTa-v3) ===")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_dataset = QuestDataset(train_df, tokenizer, CONFIG['max_len'], cat_encoder)
    val_dataset = QuestDataset(val_df, tokenizer, CONFIG['max_len'], cat_encoder)
    
    # 開啟 pin_memory 加速 GPU 傳輸
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    num_cats = len(cat_encoder.classes_)
    model = QuestDeberta(CONFIG['model_name'], num_cats=num_cats, cat_emb_dim=CONFIG['cat_emb_dim'])
    model = model.to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    
    # [新增] Scheduler
    num_train_steps = int(len(train_df) / CONFIG['train_batch_size'] * CONFIG['epochs'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    # [新增] GradScaler for AMP
    scaler = GradScaler()
    
    best_score = -1
    save_dir = './models_deberta' # 存到新資料夾，避免混淆
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        for data in train_loader:
            ids = data['ids'].to(CONFIG['device'])
            mask = data['mask'].to(CONFIG['device'])
            token_type_ids = data['token_type_ids'].to(CONFIG['device'])
            cats = data['cats'].to(CONFIG['device'])
            targets = data['targets'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # [關鍵] 使用 autocast 進行混合精度訓練
            with autocast():
                outputs = model(ids, mask, token_type_ids, cats)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for data in val_loader:
                ids = data['ids'].to(CONFIG['device'])
                mask = data['mask'].to(CONFIG['device'])
                token_type_ids = data['token_type_ids'].to(CONFIG['device'])
                cats = data['cats'].to(CONFIG['device'])
                targets = data['targets'].to(CONFIG['device'])
                
                outputs = model(ids, mask, token_type_ids, cats)
                outputs = torch.sigmoid(outputs)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        score = compute_spearman(val_preds, val_targets)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Spearman: {score:.4f}")
        
        if score > best_score:
            best_score = score
            save_path = f"{save_dir}/model_fold{fold}.bin"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best Model Saved: {best_score:.4f}")
            
    return best_score

if __name__ == "__main__":
    if not os.path.exists('train.csv'):
        print("請確保 train.csv 位於同一目錄下")
        sys.exit(1)
        
    df = pd.read_csv('train.csv')
    
    le = LabelEncoder()
    le.fit(df['category'])
    print(f"Categories: {le.classes_}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    groups = df['question_body']
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        score = run_fold(fold, train_idx, val_idx, df, tokenizer, le)
        fold_scores.append(score)
            
    print(f"\n>>> Average DeBERTa Score: {np.mean(fold_scores):.4f}")