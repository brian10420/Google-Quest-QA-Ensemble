import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
import os

# ==========================================
# 0. 防禦性設定 (防止環境衝突報錯)
# ==========================================
# 強制關閉 TensorFlow 的 Log，避免它干擾 PyTorch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 避免 Tokenizers 平行運算死鎖
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 設定路徑
# ==========================================
MODEL_CONFIG_PATH = '/kaggle/input/roberta-base/roberta_base_config/' 
WEIGHTS_DIR = '/kaggle/input/taica-512-q-and-a-v4/'
TEST_DATA_PATH = '/kaggle/input/google-quest-challenge/test.csv'
TRAIN_DATA_PATH = '/kaggle/input/google-quest-challenge/train.csv' 
SAMPLE_SUB_PATH = '/kaggle/input/google-quest-challenge/sample_submission.csv'
MODEL_WEIGHTS_FILES = [f'model_fold{i}.bin' for i in range(5)]

# ==========================================
# 2. Config & 類別定義
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
            text_a, text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        cat_val = self.cat_encoder_dict.get(row['category'], 0)

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'cats': torch.tensor(cat_val, dtype=torch.long)
        }

class QuestMethodD(nn.Module):
    def __init__(self, model_path, num_cats, cat_emb_dim=16):
        super(QuestMethodD, self).__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModel.from_config(self.config) # 使用 from_config 避免讀取權重錯誤
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
# 3. 推論主程式 (加入 Try-Except 保護)
# ==========================================
def predict():
    try:
        print(f"Device: {CONFIG['device']}")
        
        if not os.path.exists(TEST_DATA_PATH):
            print(f"Error: 找不到 {TEST_DATA_PATH}")
            return
            
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        if os.path.exists(TRAIN_DATA_PATH):
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            unique_cats = sorted(train_df['category'].unique()) 
            cat_map = {c: i for i, c in enumerate(unique_cats)}
            print(f"Category Map Rebuilt: {len(cat_map)} categories.")
        else:
            cat_map = {'CULTURE': 0, 'LIFE_ARTS': 1, 'SCIENCE': 2, 'STACKOVERFLOW': 3, 'TECHNOLOGY': 4}
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG_PATH)
        test_dataset = QuestDataset(test_df, tokenizer, CONFIG['max_len'], cat_map)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
        
        final_preds = np.zeros((len(test_df), 30))
        
        used_models = 0
        for weight_file in MODEL_WEIGHTS_FILES:
            weight_path = os.path.join(WEIGHTS_DIR, weight_file)
            if not os.path.exists(weight_path):
                continue
                
            print(f"\n--- Predicting with {weight_file} ---")
            model = QuestMethodD(MODEL_CONFIG_PATH, num_cats=len(cat_map), cat_emb_dim=CONFIG['cat_emb_dim'])
            state_dict = torch.load(weight_path, map_location=CONFIG['device'])
            model.load_state_dict(state_dict)
            model.to(CONFIG['device'])
            model.eval()
            
            fold_preds = []
            with torch.no_grad():
                for data in test_loader:
                    ids = data['ids'].to(CONFIG['device'])
                    mask = data['mask'].to(CONFIG['device'])
                    token_type_ids = data['token_type_ids'].to(CONFIG['device'])
                    cats = data['cats'].to(CONFIG['device'])
                    
                    outputs = model(ids, mask, token_type_ids, cats)
                    outputs = torch.sigmoid(outputs)
                    fold_preds.append(outputs.cpu().numpy())
            
            fold_preds = np.concatenate(fold_preds)
            final_preds += fold_preds
            used_models += 1
        
        if used_models > 0:
            final_preds /= used_models
            print(f"\nPrediction complete. Averaged over {used_models} models.")
            submission = pd.DataFrame(final_preds, columns=CONFIG['target_cols'])
            submission.insert(0, 'qa_id', test_df['qa_id'])
            submission.to_csv('submission.csv', index=False)
            print(">>> submission.csv saved successfully!")
            
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # 如果檔案已經生成，就不管錯誤了
        if os.path.exists('submission.csv'):
            print("submission.csv exists. Ignoring error.")
            
if __name__ == "__main__":
    predict()
    # 【關鍵修改】強制正常退出，截斷後面的錯誤
    print("Force exiting to prevent environment crash...")
    sys.exit(0)