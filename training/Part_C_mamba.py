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
import transformers
import bitsandbytes as bnb
import gc
# ==========================================
# 0. 環境設定 (解決顯存破碎化問題)
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==========================================
# 1. Config (針對 24GB 顯存優化)
# ==========================================
CONFIG = {
    'model_name': 'state-spaces/mamba-1.4b-hf',
    'max_len': 2048,         
    'train_batch_size':8,   # [關鍵修改] 降到 1，靠累積來模擬大 Batch
    'valid_batch_size':32,   # 驗證時不存梯度，可以大一點
    'accumulation_steps': 8, # [關鍵修改] 累積 8 次 = 等效 Batch Size 8
    'epochs': 5,             # 1.4B 收斂快，3 epoch 通常夠了
    'lr': 5e-5,              # 微調大模型稍微降一點 LR
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
# 2. Dataset
# ==========================================
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
            return_token_type_ids=False 
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(self.categories[idx], dtype=torch.long),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# ==========================================
# 3. Model
# ==========================================
class QuestMamba(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestMamba, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # [關鍵修改] 啟用 Gradient Checkpointing (省顯存神器)
        self.backbone.gradient_checkpointing_enable()

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
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        text_feature = sum_embeddings / sum_mask 
        
        cat_feature = self.cat_embedding(cats)

        combined_feature = torch.cat((text_feature, cat_feature), dim=1)
        
        out_q = self.head_q(combined_feature)
        out_a = self.head_a(combined_feature)
        
        return torch.cat((out_q, out_a), dim=1)

# ==========================================
# 4. Training Loop (含梯度累積)
# ==========================================
def compute_spearman(preds, targets):
    score = 0
    cols = preds.shape[1]
    for i in range(cols):
        if np.std(targets[:, i]) < 1e-9 or np.std(preds[:, i]) < 1e-9:
            score += 0 
        else:
            score += spearmanr(targets[:, i], preds[:, i]).correlation
    return score / cols

def run_fold(fold, train_idx, val_idx, df, tokenizer, cat_encoder):
    print(f"\n=== Running Fold {fold+1}/{CONFIG['n_folds']} (Mamba 1.4B) ===")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # 訓練集不需要 token_type_ids
    train_dataset = QuestDataset(train_df, tokenizer, CONFIG['max_len'], cat_encoder)
    val_dataset = QuestDataset(val_df, tokenizer, CONFIG['max_len'], cat_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    num_cats = len(cat_encoder.classes_)
    model = QuestMamba(CONFIG['model_name'], num_cats=num_cats, cat_emb_dim=CONFIG['cat_emb_dim'])
    model = model.to(CONFIG['device'])
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG['lr'])
    num_train_steps = int(len(train_df) / CONFIG['train_batch_size'] / CONFIG['accumulation_steps'] * CONFIG['epochs'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_train_steps)
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    best_score = -1
    save_dir = './models_mamba'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        optimizer.zero_grad() # 初始化梯度
        
        for step, data in enumerate(train_loader):
            ids = data['ids'].to(CONFIG['device'])
            mask = data['mask'].to(CONFIG['device'])
            cats = data['cats'].to(CONFIG['device'])
            targets = data['targets'].to(CONFIG['device'])
            
            with autocast():
                outputs = model(ids, mask, cats)
                loss = criterion(outputs, targets)
                # [關鍵] Loss 也要除以累積步數
                loss = loss / CONFIG['accumulation_steps']
            
            scaler.scale(loss).backward()
            
            # [關鍵] 累積 N 步後才更新一次權重
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
                
                outputs = model(ids, mask, cats)
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
            print(f"  >>> Best Saved: {best_score:.4f}")
            
    return best_score

if __name__ == "__main__":
    if not os.path.exists('train.csv'):
        print("Please upload train.csv")
        sys.exit(1)

    df = pd.read_csv('train.csv')
    le = LabelEncoder()
    le.fit(df['category'])
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    groups = df['question_body']
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        
        # 執行訓練
        score = run_fold(fold, train_idx, val_idx, df, tokenizer, le)
        fold_scores.append(score)
        
        # --- 🔥 顯存清理大法 🔥 ---
        print(f"Cleaning up Fold {fold+1} memory...")
        
        # 1. 強制回收 Python 物件 (雖然 run_fold 結束後局部變數會釋放，但保險起見)
        gc.collect()
        
        # 2. 強制清空 CUDA 快取
        torch.cuda.empty_cache()
        
        # 3. 重置峰值記憶體統計 (這步是選用的，方便觀察)
        torch.cuda.reset_peak_memory_stats()
        
        print("Memory cleared. Starting next fold...")
        # ---------------------------
    
    print(f"\n>>> Average Mamba Score: {np.mean(fold_scores):.4f}")