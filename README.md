# Google Quest QA Ensemble

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/google-quest-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Best Score](https://img.shields.io/badge/Best_Score-0.48840-success)](https://www.kaggle.com/competitions/google-quest-challenge)

An adaptive multi-model ensemble system for the Google QUEST Q&A Labeling competition, featuring length-based dynamic weighting and specialized preprocessing strategies for different model architectures.

## 🎯 Competition Overview

The Google QUEST Q&A Labeling challenge involves predicting 30 quality-related scores for question-answer pairs from StackExchange. The task requires understanding both question quality (21 targets) and answer quality (9 targets) with high correlation to human annotations.

**Evaluation Metric**: Mean column-wise Spearman correlation

## 🏆 Performance Benchmarks

### Evolution of Results

| Stage | Configuration | Private Score | Public Score | Key Improvements |
|-------|--------------|---------------|--------------|------------------|
| **Baseline** | 5× RoBERTa-base | 0.35160 | 0.37655 | Basic ensemble |
| **V2** | 5× RoBERTa + 5× DeBERTa + Post-processing | 0.36976 | 0.42359 | +20.3% (rank quantization) |
| **Final** | 5× RoBERTa + 5× DeBERTa + 5× Mamba + Stacking | **0.48840** | - | +38.9% (adaptive ensemble) |

### Model Performance Breakdown

| Method | Spearman Score | Notes |
|--------|----------------|-------|
| RoBERTa Only | 0.48467 | Strong on short contexts (≤512 tokens) |
| DeBERTa Only | 0.37360 | Better semantic understanding |
| Mamba Only | 0.38342 | Excellent on long contexts (>512 tokens) |
| Simple Average | 0.45416 | Baseline ensemble |
| Length-Weighted (Optimized) | 0.48544 | Adaptive weighting by sequence length |
| Stacker (Ridge) | 0.48751 | Meta-learner on base predictions |
| Stacker (LightGBM) | 0.47625 | Gradient boosting approach |
| **Final Ensemble** | **0.48840** | Optimized weights + Ridge stacker |

### Optimal Ensemble Weights (Optuna)

```python
Best Params: {
    'sigmoid_slope': 0.0943,
    'sigmoid_intercept': -2.476,
    'roberta_base': 1.946,
    'deberta_base': 0.101
}
```

### Length-Adaptive Weighting Examples

| Sequence Length | RoBERTa | DeBERTa | Mamba | Strategy |
|-----------------|---------|---------|-------|----------|
| 200 tokens | 88.1% | 4.6% | 7.4% | Short context: RoBERTa dominant |
| 512 tokens | 87.7% | 4.5% | 7.8% | Standard BERT limit |
| 1000 tokens | 87.1% | 4.5% | 8.4% | Transitional zone |
| 2000 tokens | 85.6% | 4.4% | 10.0% | Long context: Mamba gains importance |

## 🏗️ Architecture

### System Overview
![System Architecture](diagrams/mermaid-ai-diagram-2025-12-03-063238.png)

The system employs a dual-preprocessing strategy:
- **Transformer models (RoBERTa/DeBERTa)**: Truncate to head (256) + tail (256) tokens to capture conclusions
- **Mamba models**: Process full context (2048 tokens) without information loss

### Model Architecture (RoBERTa Example)
![Model Architecture](diagrams/Model_2.png)

Features:
- Mean pooling over hidden states
- Category embedding (16-dim) concatenation
- Multi-head output: 21 question targets + 9 answer targets

### Ensemble Pipeline
![Ensemble Pipeline](diagrams/Model_3.png)

Three-stage ensemble:
1. **Base Models**: 5-fold cross-validation for each model type
2. **Average Ensemble**: Simple averaging with rank-based quantization
3. **Stacking**: Meta-learners trained on out-of-fold predictions

### Length-Adaptive System
![Adaptive System](diagrams/Model_3_2.png)

Dynamic weighting function:
```
W_mamba = α + β × σ((L-512)/T)
```
Where L is sequence length, automatically learned via Optuna optimization.

## 🚀 Quick Start

### Prerequisites

```bash
# Hardware Requirements
- GPU: RTX 4090 (24GB VRAM) or equivalent
- CPU: Intel i7-14700 or better
- RAM: 32GB+ recommended

# Software Requirements
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Google-Quest-QA-Ensemble.git
cd Google-Quest-QA-Ensemble

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install pandas numpy scipy scikit-learn
pip install optuna lightgbm
pip install bitsandbytes  # For 8-bit optimization
pip install mamba-ssm  # For Mamba models
```

### Training Pipeline

#### Step 1: Train Base Models

```bash
# RoBERTa (512 max_len, ~6 hours for 5 folds on RTX 4090)
python training/Part_C_Roberta.py

# DeBERTa (512 max_len, ~8 hours for 5 folds)
python training/Part_C_Deberta.py

# Mamba (2048 max_len, ~12 hours for 5 folds)
python training/Part_C_mamba.py
# or for Mamba2
python training/Part_C_mamba2.py
```

Models will be saved to:
- `./models/` (RoBERTa)
- `./models_deberta/` (DeBERTa)
- `./models_mamba/` (Mamba)

#### Step 2: Generate Out-of-Fold Predictions

```bash
python ensemble/step1_generate_oof.py
```

Output: `oof_predictions.npz` containing base model predictions

#### Step 3: Optimize Ensemble Weights

```bash
python ensemble/step2_optimize_weights.py
```

This runs:
- Optuna hyperparameter search (500 trials)
- Scipy differential evolution
- Per-target weight optimization

Output: `optimized_params/best_params.json`

#### Step 4: Train Stacking Models

```bash
python ensemble/step3_train_stacker.py
```

Output: Ridge and LightGBM meta-learners in `stacker_models/`

#### Step 5: Generate Kaggle Submission

```bash
python ensemble/step4_inference_kaggle.py
```

Output: `submission.csv`

### Quick Inference (Pre-trained Models)

```bash
# Using inference V2 (with post-processing)
python inference/inference_V2.py
```

## 📁 Repository Structure

```
Google-Quest-QA-Ensemble/
├── training/
│   ├── Part_A.py                    # Exploratory Data Analysis
│   ├── Part_B.py                    # Baseline BERT model
│   ├── Part_C_Roberta.py           # RoBERTa training (5-fold CV)
│   ├── Part_C_Deberta.py           # DeBERTa training (5-fold CV)
│   ├── Part_C_mamba.py             # Mamba-1.4B training
│   ├── Part_C_mamba2.py            # Mamba2-780M training
│   └── Part_C_mamba2_1_3b.py       # Mamba2-1.3B training
│
├── ensemble/
│   ├── step1_generate_oof.py       # Generate out-of-fold predictions
│   ├── step2_optimize_weights.py   # Optimize ensemble weights
│   ├── step3_train_stacker.py      # Train meta-learners
│   └── step4_inference_kaggle.py   # Final submission generation
│
├── inference/
│   ├── inference_V1.py             # Basic ensemble inference
│   └── inference_V2.py             # With rank-based post-processing
│
├── diagrams/
│   ├── mermaid-ai-diagram-2025-12-03-063238.png
│   ├── Model_2.png
│   ├── Model_3.png
│   └── Model_3_2.png
│
├── utils/
│   └── Pose_processing_check.py    # Analyze label distributions
│
└── README.md
```

## 🔑 Key Innovations

### 1. Length-Adaptive Ensemble Weighting
Automatically adjusts model contributions based on input sequence length:
- **Short contexts (<512)**: RoBERTa dominates (88%)
- **Long contexts (>1500)**: Mamba weight increases to 10%

### 2. Dual Preprocessing Strategy
- **Transformers**: Intelligent truncation (head + tail) to preserve context
- **Mamba**: Full sequence processing with special tokens for code/math

### 3. Text Simplification
```python
text = re.sub(r'```.*?```', '[CODE]', text)      # Code blocks
text = re.sub(r'\$\$.*?\$\$', '[MATH]', text)    # LaTeX math
text = re.sub(r'http\S+', '[URL]', text)         # URLs
```

### 4. Rank-Based Quantization
Aligns prediction distributions with training label distributions:
```python
def apply_buckets(preds, mapping):
    # Maps continuous predictions to discrete buckets
    # based on training data quantiles
```

### 5. Per-Target Stacking
Separate meta-learners for each of 30 targets, allowing specialized optimization.

## 🛠️ Configuration

### Model Hyperparameters

| Model | Max Length | Batch Size | Epochs | Learning Rate | Notes |
|-------|------------|------------|--------|---------------|-------|
| RoBERTa-base | 512 | 8 | 8 | 2e-5 | Standard configuration |
| DeBERTa-v3-base | 512 | 8 | 8 | 2e-5 | Uses mixed precision (AMP) |
| Mamba-1.4B | 2048 | 8 | 5 | 5e-5 | Gradient accumulation: 8 steps |
| Mamba2-780M | 2048 | 8 | 7 | 5e-5 | Differential LR for head |

### Memory Optimization Techniques

1. **Gradient Checkpointing**: Enabled for all models
2. **Mixed Precision (FP16)**: Via PyTorch AMP
3. **8-bit Optimization**: Using bitsandbytes AdamW
4. **Gradient Accumulation**: Effective batch size of 32-64

## 📊 Results Analysis

### What Works Well

✅ **RoBERTa excels at**:
- Short, concise Q&A pairs
- Well-structured questions
- Technical accuracy assessment

✅ **DeBERTa provides**:
- Better semantic understanding
- Improved handling of nuanced language
- Complementary signal to RoBERTa

✅ **Mamba shines on**:
- Long-form answers (>1000 tokens)
- Code-heavy content
- Multi-part questions

### Ablation Study

| Component Removed | Score Drop | Impact |
|-------------------|------------|--------|
| Remove Mamba | -0.0037 | Moderate (mostly long contexts) |
| Remove DeBERTa | -0.0148 | Significant (semantic diversity) |
| Remove post-processing | -0.0229 | Critical (label alignment) |
| Remove stacking | -0.0089 | Moderate (ensemble refinement) |

## 🔮 Future Work

### Proposed: Small Ensemble Manager (SEM)

A lightweight neural network to learn:
1. **When to apply each model** based on input features
2. **Optimal preprocessing strategy** (truncate vs. full context)
3. **Dynamic weight adjustment** beyond simple length-based rules

Architecture concept:
```python
Input → [Length, Category, Code%, Math%, URL%] 
     → Small MLP (3 layers) 
     → [W_roberta, W_deberta, W_mamba, Preprocessing_mode]
```

Benefits:
- End-to-end learnable ensemble
- Adapts to content type, not just length
- Lower inference cost (fewer model calls)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google QUEST Challenge organizers
- Hugging Face Transformers library
- Mamba and Mamba2 authors (State Space Models)
- Optuna hyperparameter optimization framework

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Kaggle discussion forum

---

**Note**: This repository contains research code. For production use, additional validation and error handling is recommended.
