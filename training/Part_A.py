import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# Load Data
df = pd.read_csv('train.csv') # 請確保路徑正確

# 1. Label 離散性分析 (以 answer_helpfulness 為例)
plt.figure(figsize=(10, 5))
sns.histplot(df['answer_helpful'], bins=50, kde=False)
plt.title('Distribution of answer_helpfulness (Discrete Nature)')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()
# 截圖這張圖：說明 Target 其實不是連續的，而是幾個離散值的平均

# 2. 文本長度分析
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def get_len(row):
    # 模擬實際輸入模型時的拼接：[CLS] title + body [SEP] answer [SEP]
    text = str(row['question_title']) + " " + str(row['question_body']) + " " + str(row['answer'])
    return len(tokenizer.encode(text, max_length=5000, truncation=True))

# 為了快速演示，只取前 1000 筆算長度
sample_lens = df.head(1000).apply(get_len, axis=1)

plt.figure(figsize=(10, 5))
sns.histplot(sample_lens, bins=30)
plt.axvline(x=512, color='r', linestyle='--', label='BERT Limit (512)')
plt.title('Token Length Distribution (Title + Body + Answer)')
plt.legend()
plt.show()
# 截圖這張圖：說明有多少資料會被截斷