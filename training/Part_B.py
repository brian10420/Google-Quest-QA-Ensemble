import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# --- Config ---
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_len': 512,
    'batch_size': 4, # 1060 只有 6GB，設小一點
    'lr': 2e-5,
    'epochs': 1,     # Debug 先跑 1 epoch
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_labels': 30,
    'debug': True    # True: 只跑 100 筆資料測試流程
}

# --- Dataset ---
class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = df.iloc[:, 11:].values # train.csv 前 11 欄是 text/meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 簡單的拼接策略： Title + Body + Answer
        # 改良點預告：之後可以考慮特殊的拼接方式或加入 special tokens
        inputs = self.tokenizer.encode_plus(
            str(row['question_title']) + " " + str(row['question_body']),
            str(row['answer']),
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
            'targets': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# --- Model ---
class QuestModel(nn.Module):
    def __init__(self, model_name):
        super(QuestModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        # 簡單的 Linear Head
        self.fc = nn.Linear(self.config.hidden_size, 30)

    def forward(self, ids, mask, token_type_ids):
        # BERT output
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        # 使用 [CLS] token representation (pooler_output or last_hidden_state[:, 0, :])
        # 有些 model (like DistilBert) 沒有 pooler_output，建議取 [CLS]
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        output = self.fc(cls_output)
        return output

# --- Metric Function ---
def compute_spearman(preds, targets):
    # Column-wise Spearman
    # preds, targets: numpy arrays
    score = 0
    cols = preds.shape[1]
    for i in range(cols):
        # 避免全 0 或常數導致 NaN
        if np.std(targets[:, i]) < 1e-6 or np.std(preds[:, i]) < 1e-6:
            score += 0 # 或者略過
        else:
            score += spearmanr(targets[:, i], preds[:, i]).correlation
    return score / cols

# --- Main Debug Loop ---
if __name__ == "__main__":
    print(f"Running on {CONFIG['device']} with {CONFIG['model_name']}")
    
    # Load Data
    df = pd.read_csv(r'data/train.csv')
    if CONFIG['debug']:
        df = df.head(100) # Debug 只用 100 筆
        print("!! DEBUG MODE: Using only 100 samples !!")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    dataset = QuestDataset(df, tokenizer, CONFIG['max_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    model = QuestModel(CONFIG['model_name']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCEWithLogitsLoss() # 適合 [0,1] 範圍的 regression，比 MSE 穩定

    model.train()
    for batch_idx, data in enumerate(dataloader):
        ids = data['ids'].to(CONFIG['device'])
        mask = data['mask'].to(CONFIG['device'])
        token_type_ids = data['token_type_ids'].to(CONFIG['device'])
        targets = data['targets'].to(CONFIG['device'])

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 簡單驗證一下 metric 能不能跑
    print("Training finished. Testing Metric function...")
    dummy_preds = torch.rand(100, 30).numpy()
    dummy_targets = df.iloc[:100, 11:].values
    score = compute_spearman(dummy_preds, dummy_targets)
    print(f"Dummy Spearman Score: {score:.4f}")
    
    print("\n✅ Phase 1 & 2 Debug Setup Complete.")