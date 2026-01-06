# GPU Environment Setup Guide

Complete guide for setting up GPU training environment with PyTorch CUDA support and Mamba dependencies.

## 📋 Hardware Requirements

### Recommended Configuration (Tested)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: Intel i7-14700 or better
- **RAM**: 32GB+
- **Storage**: 100GB+ available space

### Minimum Configuration
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4080, A5000)
- **CPU**: 8+ cores
- **RAM**: 16GB+

### CUDA Version
- **Recommended**: CUDA 13.0 (default in pyproject.toml)
- **Compatible**: CUDA 11.8, 12.1

## 🔧 Installation Steps

### Step 1: Check NVIDIA Driver

```bash
# Check GPU and driver version
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 545.xx.xx    Driver Version: 545.xx.xx    CUDA Version: 13.0   |
# +-----------------------------------------------------------------------------+
```

**Driver Version Requirements**:
- CUDA 13.0: Driver >= 545.xx
- CUDA 11.8: Driver >= 450.80
- CUDA 12.1: Driver >= 525.60

**If driver is outdated**:
- Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- Download and install latest driver
- Restart computer

### Step 2: Install UV Package Manager

```bash
# Windows (PowerShell - Run as Administrator)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation (restart terminal first)
uv --version
```

### Step 3: Clone Project and Create Environment

```bash
git clone https://github.com/YOUR_USERNAME/Google-Quest-QA-Ensemble.git
cd Google-Quest-QA-Ensemble

# Create virtual environment
uv venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 4: Sync Dependencies (Including PyTorch CUDA)

```bash
# This automatically installs from PyTorch CUDA 13.0 index
uv sync

# Estimated time: 2-5 minutes
# Download size: ~3-4 GB (includes CUDA-enabled PyTorch)
```

**What `uv sync` does**:
```
1. Downloads PyTorch from https://download.pytorch.org/whl/cu130
2. Installs torchvision, torchaudio (with CUDA)
3. Installs transformers, mamba-ssm, causal-conv1d
4. Installs all other dependencies
```

### Step 5: Verify Installation

```bash
# Verify PyTorch CUDA support
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

# Expected output:
# PyTorch: 2.x.x+cu130
# CUDA Available: True
# CUDA Version: 13.0
# GPU: NVIDIA GeForce RTX 4090
```

```bash
# Verify Transformers
uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Expected output:
# Transformers: 4.x.x
```

```bash
# Verify Mamba dependencies
uv run python -c "import mamba_ssm; print('Mamba SSM: OK')"
uv run python -c "import causal_conv1d; print('Causal Conv1D: OK')"
uv run python -c "import triton; print('Triton: OK')"

# Expected output:
# Mamba SSM: OK
# Causal Conv1D: OK
# Triton: OK
```

```bash
# Verify Bitsandbytes (8-bit optimization)
uv run python -c "import bitsandbytes as bnb; print(f'Bitsandbytes: {bnb.__version__}')"

# Expected output:
# Bitsandbytes: 0.x.x
```

### Step 6: Complete Environment Test

```bash
# Run comprehensive environment check
uv run python -c "
import torch
import transformers
import mamba_ssm
import bitsandbytes
import numpy as np
import pandas as pd
import sklearn
import optuna
import lightgbm

print('=== Environment Check ===')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA Capability: {torch.cuda.get_device_capability(0)}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Mamba SSM: OK')
print(f'✓ Bitsandbytes: {bitsandbytes.__version__}')
print(f'✓ NumPy: {np.__version__}')
print(f'✓ Pandas: {pd.__version__}')
print(f'✓ Scikit-learn: {sklearn.__version__}')
print(f'✓ Optuna: {optuna.__version__}')
print(f'✓ LightGBM: {lightgbm.__version__}')
print('\\n=== All dependencies installed successfully! ===')
"
```

## 🐛 Troubleshooting

### Problem 1: `torch.cuda.is_available()` Returns False

**Diagnosis**:
```bash
# Check PyTorch version
uv run python -c "import torch; print(torch.__version__)"

# If shows 2.x.x+cpu, CPU version was installed
```

**Solution**:
```bash
# Method 1: Clear cache and reinstall
uv cache clean
rm -rf .venv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv sync

# Method 2: Manual PyTorch installation
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Problem 2: Mamba SSM Installation Failed

**Common Error**:
```
ERROR: Failed building wheel for mamba-ssm
```

**Solution**:
```bash
# Mamba requires compilation, ensure CUDA toolkit is installed
# Windows: Install Visual Studio Build Tools
# Linux: Install build-essential and CUDA toolkit

# Windows:
# 1. Download Visual Studio Build Tools
# 2. Install "Desktop development with C++"

# Linux:
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install nvidia-cuda-toolkit

# Then reinstall
uv sync
```

### Problem 3: Triton Fails on Windows

**Note**: Triton primarily supports Linux, Windows support is limited

**Solution**:
```bash
# Option 1: Use WSL2 (Recommended)
# Run in WSL2 Ubuntu environment

# Option 2: Remove triton dependency (may affect Mamba performance)
uv remove triton

# Option 3: Use precompiled version (if available)
uv add triton --prerelease allow
```

### Problem 4: Bitsandbytes CUDA Mismatch

**Error**:
```
CUDA Setup failed despite GPU being available
```

**Solution**:
```bash
# Check CUDA version
nvidia-smi  # Look for CUDA Version

# Reinstall matching version of bitsandbytes
uv pip install bitsandbytes --force-reinstall
```

### Problem 5: Out of Memory (OOM)

**Symptom**: Training shows `CUDA out of memory`

**Solution**:
```python
# Adjust parameters in training script
CONFIG = {
    'train_batch_size': 4,      # Reduce (was 8)
    'accumulation_steps': 8,    # Increase (maintain effective batch size)
    'mamba_batch_size': 2,      # Lower for Mamba models
}
```

```bash
# Or set environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 📊 Different CUDA Version Configurations

If your system uses a different CUDA version, modify `pyproject.toml`:

### CUDA 11.8 (Older but Stable)
```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cu118", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

### CUDA 12.1 (Latest)
```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cu121", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

### CPU Only (No GPU)
```toml
# Remove [[tool.uv.index]] and [tool.uv.sources] sections
# UV will automatically install CPU version
```

## 🧪 Performance Test

Run simple GPU test to ensure everything works:

```bash
# Create test script
cat > test_gpu.py << 'EOF'
import torch
import time

# Test GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Simple matrix multiplication test
if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Create large matrices
    x = torch.randn(10000, 10000, device=device)
    y = torch.randn(10000, 10000, device=device)
    
    # GPU computation
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"\nGPU Matrix Multiplication (10000x10000):")
    print(f"Time: {gpu_time:.4f} seconds")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    print("\n✓ GPU is working correctly!")
else:
    print("\n✗ CUDA not available. Check installation.")
EOF

# Run test
uv run python test_gpu.py
```

**Expected Output**:
```
CUDA Available: True
Device Count: 1
Current Device: 0
Device Name: NVIDIA GeForce RTX 4090

GPU Matrix Multiplication (10000x10000):
Time: 0.0234 seconds
Memory Allocated: 0.76 GB
Memory Cached: 0.78 GB

✓ GPU is working correctly!
```

## 📚 Reference Documentation

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [Bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [UV Documentation](https://github.com/astral-sh/uv)

## ✅ Installation Checklist

After installation, verify:
- [ ] `nvidia-smi` shows GPU and driver
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] PyTorch version includes `+cu130` (or your CUDA version)
- [ ] Mamba SSM imports successfully
- [ ] Bitsandbytes imports successfully
- [ ] GPU test script runs successfully
- [ ] At least 20GB VRAM available (for Mamba training)

Once all checks pass, you're ready to start training! 🚀

## 💡 Next Steps

After environment setup:
1. Download competition data: `kaggle competitions download -c google-quest-challenge`
2. Run EDA: `uv run training/Part_A.py`
3. Start training: `uv run training/Part_C_Roberta.py`

## 🔍 Quick Verification Script

Save this as `verify_environment.py` for easy checking:

```python
#!/usr/bin/env python3
"""
Quick environment verification script
Run: uv run python verify_environment.py
"""

import sys

def check_package(package_name, import_name=None):
    """Check if package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT FOUND")
        return False

def main():
    print("=== GPU Environment Verification ===\n")
    
    all_ok = True
    
    # Check PyTorch and CUDA
    print("Core Deep Learning:")
    if check_package('torch'):
        import torch
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
        else:
            print("  ✗ WARNING: CUDA not available!")
            all_ok = False
    else:
        all_ok = False
    
    check_package('torchvision')
    check_package('torchaudio')
    print()
    
    # Check Mamba ecosystem
    print("Mamba Ecosystem:")
    all_ok &= check_package('mamba-ssm', 'mamba_ssm')
    all_ok &= check_package('causal-conv1d', 'causal_conv1d')
    all_ok &= check_package('triton')
    print()
    
    # Check optimization
    print("Optimization:")
    all_ok &= check_package('bitsandbytes')
    all_ok &= check_package('accelerate')
    print()
    
    # Check NLP
    print("NLP & ML:")
    all_ok &= check_package('transformers')
    all_ok &= check_package('tokenizers')
    check_package('datasets')
    check_package('sentencepiece')
    print()
    
    # Check ML libraries
    print("Machine Learning:")
    check_package('numpy')
    check_package('pandas')
    check_package('scipy')
    check_package('scikit-learn', 'sklearn')
    check_package('lightgbm')
    check_package('optuna')
    print()
    
    # Final verdict
    if all_ok:
        print("🎉 All critical dependencies verified!")
        print("✓ Environment is ready for training")
        return 0
    else:
        print("⚠️  Some dependencies missing or CUDA not available")
        print("→ See GPU_SETUP.md for troubleshooting")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run verification:
```bash
uv run python verify_environment.py
```

## 🎯 Common Installation Patterns

### Fresh Installation
```bash
# Complete setup from scratch
git clone <repo>
cd Google-Quest-QA-Ensemble
uv venv
source .venv/bin/activate
uv sync
uv run python verify_environment.py
```

### Update After Git Pull
```bash
# After pulling new changes
git pull
uv sync  # Updates dependencies if pyproject.toml changed
```

### Reset Environment
```bash
# If environment is corrupted
rm -rf .venv
uv cache clean
uv venv
source .venv/bin/activate
uv sync
```

### Switch CUDA Version
```bash
# After editing pyproject.toml
uv cache clean
uv sync --reinstall
```

## 📝 Notes

- **Windows Users**: WSL2 is recommended for Triton and better Mamba support
- **macOS Users**: CUDA is not available; use CPU version or cloud GPU
- **Linux Users**: Best experience with native CUDA support
- **Memory**: Mamba training requires minimum 16GB VRAM, 24GB recommended
- **Storage**: Reserve at least 10GB for models and data

## 🆘 Getting Help

If you encounter issues:
1. Check this troubleshooting section first
2. Run `verify_environment.py` to identify missing components
3. Review error messages carefully
4. Search for similar issues on GitHub
5. Check PyTorch/Mamba official documentation

## 🔄 Update Log

This setup guide is maintained alongside the project. Last updated for:
- PyTorch: 2.x+
- CUDA: 13.0 (default)
- UV: Latest version
- Python: 3.10+
