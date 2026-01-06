import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import sys
import glob

# ==========================================
# 1. 精確路徑設定 (根據你提供的列表修正)
# ==========================================
# [修正] DeBERTa Config 路徑 (加上了子資料夾)
DEBERTA_CONFIG_PATH = '/kaggle/input/deberta-v3-base-config/deberta_v3_base_config/'
# RoBERTa Config 路徑
ROBERTA_CONFIG_PATH = '/kaggle/input/roberta-base/roberta_base_config/'

# 模型權重路徑
ROBERTA_WEIGHTS_DIR = '/kaggle/input/taica-512-q-and-a-v4/' 
DEBERTA_WEIGHTS_DIR = '/kaggle/input/quest-deberta-models/' 

# 比賽資料
TEST_DATA_PATH = '/kaggle/input/google-quest-challenge/test.csv'
TRAIN_DATA_PATH = '/kaggle/input/google-quest-challenge/train.csv' 

# ==========================================
# 2. Config & 通用類別
# ==========================================
CONFIG = {
    'max_len': 512,
    'batch_size': 32,
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

class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, cat_encoder_dict):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cat_encoder_dict = cat_encoder_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_a = str(row['question_title']) + " " + str(row['question_body'])
        text_b = str(row['answer'])
        inputs = self.tokenizer.encode_plus(
            text_a, text_b, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_token_type_ids=True
        )
        cat_val = self.cat_encoder_dict.get(row['category'], 0)
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'cats': torch.tensor(cat_val, dtype=torch.long)
        }

class QuestModel(nn.Module):
    def __init__(self, config_path, num_cats, cat_emb_dim=16):
        super(QuestModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config_path)
        self.backbone = AutoModel.from_config(self.config)
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, token_type_ids, cats):
        outputs = self.backbone(ids, attention_mask=mask, token_type_ids=token_type_ids)
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
# 3. 後處理函式
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
# 4. 主程式
# ==========================================
def predict():
    print(f"Device: {CONFIG['device']}")
    
    # 讀取資料
    if not os.path.exists(TEST_DATA_PATH): return
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    if os.path.exists(TRAIN_DATA_PATH):
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        unique_cats = sorted(train_df['category'].unique()) 
        cat_map = {c: i for i, c in enumerate(unique_cats)}
    else:
        cat_map = {'CULTURE': 0, 'LIFE_ARTS': 1, 'SCIENCE': 2, 'STACKOVERFLOW': 3, 'TECHNOLOGY': 4}
        train_df = None

    final_preds = np.zeros((len(test_df), 30))
    total_models = 0

    # --- Ensemble Step 1: RoBERTa Models ---
    if os.path.exists(ROBERTA_WEIGHTS_DIR):
        roberta_files = sorted(glob.glob(os.path.join(ROBERTA_WEIGHTS_DIR, '*.bin')))
        if len(roberta_files) > 0:
            print(f"\n>>> Found {len(roberta_files)} RoBERTa models.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CONFIG_PATH)
                dataset = QuestDataset(test_df, tokenizer, CONFIG['max_len'], cat_map)
                loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
                
                for f in roberta_files:
                    print(f"Predicting: {os.path.basename(f)}")
                    model = QuestModel(ROBERTA_CONFIG_PATH, len(cat_map))
                    model.load_state_dict(torch.load(f, map_location=CONFIG['device']))
                    model.to(CONFIG['device'])
                    model.eval()
                    fold_preds = []
                    with torch.no_grad():
                        for data in loader:
                            ids, mask = data['ids'].to(CONFIG['device']), data['mask'].to(CONFIG['device'])
                            tt_ids, cats = data['token_type_ids'].to(CONFIG['device']), data['cats'].to(CONFIG['device'])
                            out = model(ids, mask, tt_ids, cats)
                            fold_preds.append(torch.sigmoid(out).cpu().numpy())
                    final_preds += np.concatenate(fold_preds)
                    total_models += 1
                    del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in RoBERTa pipeline: {e}")

    # --- Ensemble Step 2: DeBERTa Models ---
    if os.path.exists(DEBERTA_WEIGHTS_DIR):
        deberta_files = sorted(glob.glob(os.path.join(DEBERTA_WEIGHTS_DIR, '*.bin')))
        if len(deberta_files) > 0:
            print(f"\n>>> Found {len(deberta_files)} DeBERTa models.")
            
            # 檢查檔案是否存在
            if not os.path.exists(os.path.join(DEBERTA_CONFIG_PATH, 'spm.model')):
                print(f"Warning: spm.model not found in {DEBERTA_CONFIG_PATH}")

            try:
                # [關鍵] 使用 use_fast=False，因為我們有 spm.model
                tokenizer = AutoTokenizer.from_pretrained(DEBERTA_CONFIG_PATH, use_fast=False)
                dataset = QuestDataset(test_df, tokenizer, CONFIG['max_len'], cat_map)
                loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
                
                for f in deberta_files:
                    print(f"Predicting: {os.path.basename(f)}")
                    model = QuestModel(DEBERTA_CONFIG_PATH, len(cat_map))
                    model.load_state_dict(torch.load(f, map_location=CONFIG['device']))
                    model.to(CONFIG['device'])
                    model.eval()
                    fold_preds = []
                    with torch.no_grad():
                        for data in loader:
                            ids, mask = data['ids'].to(CONFIG['device']), data['mask'].to(CONFIG['device'])
                            tt_ids, cats = data['token_type_ids'].to(CONFIG['device']), data['cats'].to(CONFIG['device'])
                            out = model(ids, mask, tt_ids, cats)
                            fold_preds.append(torch.sigmoid(out).cpu().numpy())
                    final_preds += np.concatenate(fold_preds)
                    total_models += 1
                    del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ Warning: DeBERTa failed to run. Error: {e}")

    # --- Ensemble Step 3: Averaging ---
    if total_models > 0:
        final_preds /= total_models
        print(f"\n>>> Ensemble complete. Averaged over {total_models} models.")
        
        # Post-Processing
        if train_df is not None:
            print("\n>>> Applying Post-Processing (Rank-based Quantization)...")
            optimized_preds = final_preds.copy()
            for i, col in enumerate(CONFIG['target_cols']):
                mapping = get_best_buckets(train_df, col)
                optimized_preds[:, i] = apply_buckets(final_preds[:, i], mapping)
            
            sub = pd.DataFrame(optimized_preds, columns=CONFIG['target_cols'])
            sub.insert(0, 'qa_id', test_df['qa_id'])
            sub.to_csv('submission.csv', index=False)
            print(">>> submission.csv (Optimized) saved successfully!")
        else:
            raw_sub = pd.DataFrame(final_preds, columns=CONFIG['target_cols'])
            raw_sub.insert(0, 'qa_id', test_df['qa_id'])
            raw_sub.to_csv('submission.csv', index=False)
            
    else:
        print("Error: No models were found/run at all!")

if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print(f"Critical Error: {e}")
    sys.exit(0)