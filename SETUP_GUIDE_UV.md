# Repository Setup Guide (UV Edition)

This guide will help you set up your GitHub repository using UV package manager.

## Prerequisites

### Install UV

UV is a fast Python package manager written in Rust. Install it first:

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify Installation**:
```bash
uv --version
# Should show: uv 0.x.x
```

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right → "New repository"
3. Repository name: `Google-Quest-QA-Ensemble`
4. Description: `Adaptive multi-model ensemble for Google QUEST Q&A Labeling`
5. Choose: ☑️ Public
6. ☐ Do NOT initialize with README (we'll add it manually)
7. Click "Create repository"

## Step 2: Initialize Local Repository

```bash
# Create project directory
mkdir Google-Quest-QA-Ensemble
cd Google-Quest-QA-Ensemble

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
.venv/
venv/
ENV/

# UV specific
.uv/

# PyTorch models
*.pth
*.pt
*.bin
*.ckpt

# Generated data
models/
models_deberta/
models_mamba/
stacker_models/
oof_data/
optimized_params/

# Data files
data/*.csv
!data/sample_submission.csv

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
EOF
```

## Step 3: Setup UV Project

```bash
# Initialize UV project (creates pyproject.toml)
uv init

# Or use the provided pyproject.toml
# Copy the pyproject.toml file to your directory
```

## Step 4: Create Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate it
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

## Step 5: Install Dependencies

```bash
# Sync all dependencies from pyproject.toml
uv sync

# This will:
# 1. Read pyproject.toml
# 2. Resolve all dependencies
# 3. Create/update uv.lock
# 4. Install all packages

# Verify installation
uv pip list
```

## Step 6: Create Directory Structure

```bash
# Create all necessary directories
mkdir -p training ensemble inference diagrams utils notebooks data oof_data stacker_models optimized_params

# Verify structure
tree -L 1
```

## Step 7: Add Your Files

### Copy Training Scripts
```bash
# Move your training files
mv Part_A.py training/
mv Part_B.py training/
mv Part_C_Roberta.py training/
mv Part_C_Deberta.py training/
mv Part_C_mamba*.py training/
```

### Copy Ensemble Scripts
```bash
mv step1_generate_oof.py ensemble/
mv step2_optimize_weights.py ensemble/
mv step3_train_stacker.py ensemble/
mv step4_inference_kaggle.py ensemble/
```

### Copy Inference Scripts
```bash
mv inference*.py inference/
```

### Copy Diagrams
```bash
mv *.png diagrams/
```

### Copy Utilities
```bash
mv Pose_processing_check.py utils/
```

## Step 8: Add README and License

```bash
# Copy the updated README.md
cp README_updated.md README.md

# Copy LICENSE
cp LICENSE .

# Copy data README
cp DATA_README.md data/
```

## Step 9: Configure Git

```bash
# Set your git config (if not already done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/Google-Quest-QA-Ensemble.git
```

## Step 10: Initial Commit

```bash
# Add all files
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: Complete ensemble system with UV package management

- Added RoBERTa, DeBERTa, and Mamba training scripts
- Implemented 4-step ensemble pipeline
- Added length-adaptive weighting optimization
- Included comprehensive documentation
- Configured UV for dependency management"

# Push to GitHub
git push -u origin main
# Or if using master branch:
# git push -u origin master
```

## Step 11: Add Topics and Description on GitHub

1. Go to your repository on GitHub
2. Click "Settings" (or the gear icon near "About")
3. Add topics: `kaggle`, `nlp`, `ensemble-learning`, `transformers`, `mamba`, `pytorch`, `machine-learning`, `uv`
4. Add website: Link to Kaggle competition
5. Under "Social preview": Upload one of your architecture diagrams

## UV-Specific Workflows

### Adding New Dependencies

```bash
# Add a single package
uv add <package-name>

# Add with specific version
uv add "torch>=2.0.0"

# Add to dev dependencies
uv add --dev pytest

# Example: Add wandb for experiment tracking
uv add wandb
```

### Removing Dependencies

```bash
# Remove a package
uv remove <package-name>

# Example
uv remove wandb
```

### Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific package
uv add --upgrade <package-name>

# Update UV itself
uv self update
```

### Running Scripts

```bash
# Run any Python script with UV
uv run training/Part_C_Roberta.py

# Run with specific Python version
uv run --python 3.11 training/Part_C_Roberta.py

# Run in activated venv (faster)
# First activate: source .venv/bin/activate
python training/Part_C_Roberta.py
```

### Freezing Environment

```bash
# The uv.lock file automatically tracks exact versions
# To export for other tools:
uv pip freeze > requirements.txt

# To recreate environment elsewhere:
# 1. Copy pyproject.toml and uv.lock
# 2. Run: uv sync
```

## Working with Jupyter

```bash
# Add Jupyter to dev dependencies (if not already)
uv add --dev jupyter ipykernel ipywidgets

# Install kernel
uv run python -m ipykernel install --user --name quest-ensemble

# Launch Jupyter
uv run jupyter lab
```

## Testing Your Setup

```bash
# Test Python installation
uv run python --version

# Test PyTorch installation
uv run python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Transformers
uv run python -c "import transformers; print(f'Transformers {transformers.__version__}')"

# Run EDA to verify everything works
uv run training/Part_A.py
```

## Reproducing Environment on Another Machine

### On the new machine:

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/Google-Quest-QA-Ensemble.git
cd Google-Quest-QA-Ensemble

# 2. Install UV (if not already installed)
# See installation instructions above

# 3. Create venv and sync
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync

# 4. Done! All dependencies are now installed exactly as specified
```

## UV vs PIP Comparison

| Task | PIP | UV |
|------|-----|-----|
| Create venv | `python -m venv .venv` | `uv venv` |
| Activate | `source .venv/bin/activate` | Same |
| Install | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install package` | `uv add package` |
| Speed | ~30 seconds | **~3 seconds** (10× faster) |
| Lock file | requirements.txt (manual) | uv.lock (automatic) |

## Troubleshooting

### Issue: UV command not found
```bash
# Restart your terminal after installation
# Or add to PATH manually (Windows):
$env:Path += ";$HOME\.cargo\bin"

# macOS/Linux:
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA support
uv add --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: UV sync fails
```bash
# Clear UV cache
uv cache clean

# Delete lock file and resync
rm uv.lock
uv sync
```

### Issue: Import errors in scripts
```bash
# Make sure you're using uv run or activated venv
uv run python script.py

# Or activate first:
source .venv/bin/activate
python script.py
```

## Best Practices with UV

1. **Always use `uv sync`** after pulling from Git
2. **Commit both `pyproject.toml` and `uv.lock`** for reproducibility
3. **Use `uv run`** for one-off scripts
4. **Use activated venv** for development sessions
5. **Regularly update**: `uv sync --upgrade` monthly
6. **Document custom dependencies** in README if using special indices

## Additional Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs PIP Performance](https://astral.sh/blog/uv)
- [Python Packaging Guide](https://packaging.python.org/)

---

Your repository is now set up with modern Python tooling! 🚀

Next steps:
1. Download competition data: `kaggle competitions download -c google-quest-challenge`
2. Start training: `uv run training/Part_C_Roberta.py`
3. Build ensemble: Follow the 4-step pipeline in README
