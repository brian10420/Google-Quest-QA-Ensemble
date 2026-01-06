# UV Quick Start Guide

快速上手 UV 包管理工具運行本項目。

## ⚡ 一分鐘快速開始

```bash
# 1. 安裝 UV (首次使用)
# Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆項目
git clone https://github.com/YOUR_USERNAME/Google-Quest-QA-Ensemble.git
cd Google-Quest-QA-Ensemble

# 3. 創建環境並安裝依賴
uv venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
uv sync

# 4. 運行腳本
uv run training/Part_A.py
```

## 📋 常用命令速查

### 環境管理
```bash
uv venv                    # 創建虛擬環境
uv sync                    # 同步依賴（首次或更新後必做）
uv sync --upgrade          # 升級所有依賴
```

### 運行腳本
```bash
# 方式1: 使用 uv run（推薦，自動使用正確環境）
uv run training/Part_C_Roberta.py

# 方式2: 激活環境後直接運行（開發時更快）
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
python training/Part_C_Roberta.py
```

### 依賴管理
```bash
uv add <package>          # 添加新包
uv add --dev <package>    # 添加開發依賴
uv remove <package>       # 移除包
uv pip list               # 查看已安裝的包
```

## 🔄 完整訓練流程（使用 UV）

### Phase 1: 訓練基礎模型（~26小時）

```bash
# RoBERTa (5 folds, ~6 hours)
uv run training/Part_C_Roberta.py

# DeBERTa (5 folds, ~8 hours)  
uv run training/Part_C_Deberta.py

# Mamba (5 folds, ~12 hours)
uv run training/Part_C_mamba.py
```

**輸出**: 
- `./models/` - RoBERTa 模型 (5個 .bin 文件)
- `./models_deberta/` - DeBERTa 模型 (5個 .bin 文件)
- `./models_mamba/` - Mamba 模型 (5個 .bin 文件)

### Phase 2: 生成 OOF 預測（~30分鐘）

```bash
uv run ensemble/step1_generate_oof.py
```

**輸出** (`oof_data/`):
- `oof_roberta.npy` - RoBERTa 預測
- `oof_deberta.npy` - DeBERTa 預測  
- `oof_mamba.npy` - Mamba 預測
- `oof_targets.npy` - 真實標籤
- `meta_features.csv` - 元特徵
- `cat_classes.json` - 類別映射

### Phase 3: 優化集成權重（~10分鐘）

```bash
uv run ensemble/step2_optimize_weights.py
```

**輸出** (`optimized_params/`):
- `best_params.json` - 最優參數
- `weight_visualization.png` - 權重可視化

**預期得分**: 0.48544

### Phase 4: 訓練堆疊模型（~5分鐘）

```bash
uv run ensemble/step3_train_stacker.py
```

**輸出** (`stacker_models/`):
- `ridge_models/models.pkl` - Ridge 回歸模型
- `lgb_models/models.pkl` - LightGBM 模型
- `stacker_config.json` - 配置文件
- `final_oof.npy` - 最終 OOF 預測

**預期得分**: 0.48840 ⭐

### Phase 5: 生成提交文件

```bash
# 本地測試推理
uv run inference/inference_local.py

# Kaggle 提交（需上傳到 Kaggle notebook）
uv run ensemble/step4_inference_kaggle.py
```

## 📂 項目文件輸出總覽

```
項目根目錄/
├── models/                    # Phase 1 輸出
│   ├── model_fold0.bin       # RoBERTa fold 0
│   ├── model_fold1.bin
│   └── ...
├── models_deberta/            # Phase 1 輸出
│   └── model_fold*.bin
├── models_mamba/              # Phase 1 輸出
│   └── model_fold*.bin
├── oof_data/                  # Phase 2 輸出
│   ├── oof_roberta.npy       # 形狀: (6079, 30)
│   ├── oof_deberta.npy
│   ├── oof_mamba.npy
│   ├── oof_targets.npy
│   ├── meta_features.csv
│   └── cat_classes.json
├── optimized_params/          # Phase 3 輸出
│   ├── best_params.json
│   └── weight_visualization.png
└── stacker_models/            # Phase 4 輸出
    ├── ridge_models/
    ├── lgb_models/
    ├── stacker_config.json
    └── final_oof.npy
```

## 🐛 常見問題

### Q: UV 命令找不到？
```bash
# 重啟終端，或手動添加到 PATH
# Windows:
$env:Path += ";$HOME\.cargo\bin"
# macOS/Linux:
export PATH="$HOME/.cargo/bin:$PATH"
```

### Q: CUDA out of memory?
```python
# 在訓練腳本中減少 batch_size
CONFIG['train_batch_size'] = 4  # 原本是 8
CONFIG['accumulation_steps'] = 8  # 保持有效 batch size
```

### Q: uv sync 失敗？
```bash
# 清除緩存重試
uv cache clean
uv sync
```

### Q: 從其他機器遷移項目？
```bash
# 在新機器上：
git clone <your-repo>
cd <project>
uv venv
uv sync  # 自動安裝所有依賴
```

## 🎯 性能對比

| 操作 | pip | uv | 提升 |
|------|-----|-----|------|
| 安裝所有依賴 | ~45秒 | ~4秒 | **11×** |
| 添加單個包 | ~8秒 | ~1秒 | **8×** |
| 鎖定依賴 | 手動 | 自動 | ∞ |

## 💡 最佳實踐

### 1. 開發流程
```bash
# 早上開始工作
cd Google-Quest-QA-Ensemble
source .venv/bin/activate
git pull
uv sync  # 同步最新依賴

# 開發...
python training/Part_C_Roberta.py

# 提交前
git add .
git commit -m "..."
git push
```

### 2. 添加新功能需要新包
```bash
# 例如：添加 wandb 用於實驗追蹤
uv add wandb

# 修改代碼...
import wandb
wandb.init(project="quest-ensemble")

# 提交
git add pyproject.toml uv.lock
git commit -m "Add wandb for experiment tracking"
```

### 3. 團隊協作
```bash
# 隊友 A 添加了新依賴
# 隊友 B 只需要：
git pull
uv sync  # 自動安裝新依賴
```

## 📊 UV vs 傳統工具

### 舊方式 (pip + requirements.txt)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 😴 慢
pip freeze > requirements.txt    # 😓 手動
```

### 新方式 (UV)
```bash
uv venv
source .venv/bin/activate
uv sync  # ⚡ 快，且自動管理 uv.lock
```

## 🚀 進階用法

### 條件安裝
```bash
# 只安裝開發依賴
uv sync --only-dev

# 不安裝開發依賴（生產環境）
uv sync --no-dev
```

### 指定 Python 版本
```bash
# 使用特定 Python 版本創建環境
uv venv --python 3.11
uv venv --python 3.10
```

### 全局工具安裝
```bash
# 安裝全局工具（不在項目環境中）
uv tool install black
uv tool install ruff
```

## 📚 更多資源

- **UV 官方文檔**: https://github.com/astral-sh/uv
- **UV 性能測試**: https://astral.sh/blog/uv
- **Python 打包指南**: https://packaging.python.org/

---

有問題？查看完整的 [SETUP_GUIDE_UV.md](SETUP_GUIDE_UV.md) 或提交 Issue！
