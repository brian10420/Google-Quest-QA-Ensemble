import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取本地 train.csv
df = pd.read_csv('train.csv')
target_cols = [col for col in df.columns if col not in ['qa_id', 'question_title', 'question_body', 'answer_user_name', 'question_user_name', 'question_user_page', 'answer_user_page', 'url', 'category', 'host', 'answer']] # 或是直接複製你的 target_cols 列表

# 找出稀疏欄位
sparse_cols = []
dense_cols = []

print("正在分析欄位分佈...")
for col in target_cols:
    # 計算 0 值所佔的比例
    zero_ratio = (df[col] == 0).mean()
    if zero_ratio > 0.5: # 如果超過 50% 都是 0，視為稀疏
        sparse_cols.append(col)
        print(f"[Sparse] {col}: {zero_ratio:.2%} zeros")
    else:
        dense_cols.append(col)
        print(f"[Dense ] {col}: {zero_ratio:.2%} zeros")

print(f"\n建議策略：\n對 {len(dense_cols)} 個密集欄位使用 Rank Quantization。\n對 {len(sparse_cols)} 個稀疏欄位考慮加入 Threshold (e.g., <0.001 set to 0)。")import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取本地 train.csv
df = pd.read_csv('train.csv')
target_cols = [col for col in df.columns if col not in ['qa_id', 'question_title', 'question_body', 'answer_user_name', 'question_user_name', 'question_user_page', 'answer_user_page', 'url', 'category', 'host', 'answer']] # 或是直接複製你的 target_cols 列表

# 找出稀疏欄位
sparse_cols = []
dense_cols = []

print("正在分析欄位分佈...")
for col in target_cols:
    # 計算 0 值所佔的比例
    zero_ratio = (df[col] == 0).mean()
    if zero_ratio > 0.5: # 如果超過 50% 都是 0，視為稀疏
        sparse_cols.append(col)
        print(f"[Sparse] {col}: {zero_ratio:.2%} zeros")
    else:
        dense_cols.append(col)
        print(f"[Dense ] {col}: {zero_ratio:.2%} zeros")

print(f"\n建議策略：\n對 {len(dense_cols)} 個密集欄位使用 Rank Quantization。\n對 {len(sparse_cols)} 個稀疏欄位考慮加入 Threshold (e.g., <0.001 set to 0)。")