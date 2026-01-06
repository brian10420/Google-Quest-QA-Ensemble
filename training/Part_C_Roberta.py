import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import os

# --- Config ---
CONFIG = {
    'model_name': 'roberta-base',
    'max_len': 512,          # 配合 1060 顯存調整
    'train_batch_size': 8,   
    'valid_batch_size': 64,
    'epochs': 8,
    'lr': 2e-5,
    'n_folds': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cat_emb_dim': 16,       # Category Embedding 維度
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

# --- Dataset (Method D: 支援 Category) ---
class QuestDataset(Dataset):
    # 修正：接收 cat_encoder
    def __init__(self, df, tokenizer, max_len, cat_encoder):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = df[CONFIG['target_cols']].values
        # 這裡會使用傳進來的 encoder 將字串轉為數字
        # 所以外面的 df 必須保持字串格式，不能先轉成數字
        self.categories = cat_encoder.transform(df['category'].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text_a = str(row['question_title']) + " " + str(row['question_body'])
        text_b = str(row['answer'])

        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(self.categories[idx], dtype=torch.long), # 回傳 Category ID
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# --- Model (Method D: Multi-Head + Category Emb) ---
class QuestMethodD(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestMethodD, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 1. Category Embedding
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        
        # 2. Combined Dimension
        combined_dim = self.config.hidden_size + cat_emb_dim
        
        # 3. Multi-Head Output
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats):
        outputs = self.backbone(ids, attention_mask=mask)
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
        
        # Split Heads
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

# --- Training Loop ---
def run_fold(fold, train_idx, val_idx, df, tokenizer, cat_encoder):
    print(f"\n=== Running Fold {fold+1}/{CONFIG['n_folds']} (Method D) ===")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # 這裡傳入 cat_encoder，它會自動把 train_df 裡的字串轉數字
    train_dataset = QuestDataset(train_df, tokenizer, CONFIG['max_len'], cat_encoder)
    val_dataset = QuestDataset(val_df, tokenizer, CONFIG['max_len'], cat_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], shuffle=True, num_workers=0) # Windows上設0比較穩
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, num_workers=0)
    
    num_cats = len(cat_encoder.classes_)
    model = QuestMethodD(CONFIG['model_name'], num_cats=num_cats, cat_emb_dim=CONFIG['cat_emb_dim'])
    model = model.to(CONFIG['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss()
    
    best_score = -1
    save_dir = './models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for data in train_loader:
            ids = data['ids'].to(CONFIG['device'])
            mask = data['mask'].to(CONFIG['device'])
            cats = data['cats'].to(CONFIG['device'])
            targets = data['targets'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(ids, mask, cats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Spearman: {score:.4f}")
        
        if score > best_score:
            best_score = score
            save_path = f"{save_dir}/model_fold{fold}.bin"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Model Saved to {save_path} (Score: {best_score:.4f})")
            
    return best_score

# --- Main ---
if __name__ == "__main__":
    df = pd.read_csv(r'data\train.csv')
    
    # --- 關鍵修正處 ---
    le = LabelEncoder()
    le.fit(df['category']) # 只做 fit，不要 transform df 裡的資料！
    # df['category'] 保持字串狀態，交給 QuestDataset 內部去 transform
    
    print(f"Categories encoded: {le.classes_}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    gkf = GroupKFold(n_splits=CONFIG['n_folds'])
    groups = df['question_body']
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[CONFIG['target_cols']], groups)):
        score = run_fold(fold, train_idx, val_idx, df, tokenizer, le)
        fold_scores.append(score)
        
        # 測試用：只跑完 Fold 0 就停，確認流程沒問題
        # 如果你要跑完全部，請把下面這行註解掉
        # break 
            
    print(f"\n>>> Average Spearman: {np.mean(fold_scores):.4f}")