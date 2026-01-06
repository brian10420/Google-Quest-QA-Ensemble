import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

# --- Config (必須與訓練一致) ---
class CONFIG:
    model_name = 'roberta-base'
    max_len = 384
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cat_emb_dim = 16
    num_cats = 5
    model_dir = './models'  # 讀取您本機訓練好的模型路徑
    
    target_cols = [
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

# --- Model (Method D) ---
# 必須重新定義一次，Inference 才知道模型長怎樣
class QuestMethodD(nn.Module):
    def __init__(self, model_name, num_cats, cat_emb_dim=16):
        super(QuestMethodD, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.cat_embedding = nn.Embedding(num_cats, cat_emb_dim)
        combined_dim = self.config.hidden_size + cat_emb_dim
        self.head_q = nn.Linear(combined_dim, 21)
        self.head_a = nn.Linear(combined_dim, 9)

    def forward(self, ids, mask, cats):
        outputs = self.backbone(ids, attention_mask=mask)
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

# --- Test Dataset ---
class QuestTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 硬編碼 Category Mapping (確保跟 Training 的 LabelEncoder 順序一致)
        # Alphabetical order: CULTURE, LIFE_ARTS, SCIENCE, STACKOVERFLOW, TECHNOLOGY
        self.cat_map = {
            'CULTURE': 0, 'LIFE_ARTS': 1, 'SCIENCE': 2, 
            'STACKOVERFLOW': 3, 'TECHNOLOGY': 4
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_a = str(row['question_title']) + " " + str(row['question_body'])
        text_b = str(row['answer'])
        
        cat_str = row['category']
        cat_idx = self.cat_map.get(cat_str, 0)

        inputs = self.tokenizer.encode_plus(
            text_a, text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'cats': torch.tensor(cat_idx, dtype=torch.long)
        }

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading test.csv...")
    df_test = pd.read_csv('test.csv')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
    test_dataset = QuestTestDataset(df_test, tokenizer, CONFIG.max_len)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=0)
    
    final_preds = np.zeros((len(df_test), 30))
    n_folds = 5
    folds_found = 0
    
    print("Starting Inference...")
    for fold in range(n_folds):
        weight_path = f"{CONFIG.model_dir}/model_fold{fold}.bin"
        if not os.path.exists(weight_path):
            print(f"⚠️ Warning: {weight_path} not found. Skipping.")
            continue
            
        print(f"Predicting with Fold {fold}...")
        model = QuestMethodD(CONFIG.model_name, num_cats=CONFIG.num_cats, cat_emb_dim=CONFIG.cat_emb_dim)
        model.to(CONFIG.device)
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG.device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for data in test_loader:
                ids = data['ids'].to(CONFIG.device)
                mask = data['mask'].to(CONFIG.device)
                cats = data['cats'].to(CONFIG.device)
                
                outputs = model(ids, mask, cats)
                outputs = torch.sigmoid(outputs)
                fold_preds.append(outputs.cpu().numpy())
        
        fold_preds = np.concatenate(fold_preds)
        final_preds += fold_preds
        folds_found += 1
    
    if folds_found > 0:
        final_preds /= folds_found
        print("Saving submission.csv...")
        
        # 讀取 sample 來確保格式
        sub = pd.read_csv('sample_submission.csv')
        # 填入預測值
        sub[CONFIG.target_cols] = final_preds
        sub.to_csv('submission.csv', index=False)
        print("✅ Done! submission.csv created locally.")
        print(sub.head())
    else:
        print("❌ Error: No models found in ./models/ directory. Please run training first.")