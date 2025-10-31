# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for running LLMs)
- 16GB+ RAM

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/semweight-equity.git
cd semweight-equity
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**On Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Package

```bash
pip install -e .
```

## GPU Setup

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (e.g., `cu117`, `cu121`).

### Verify GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## Download Datasets

```bash
bash scripts/download_data.sh
```

Note: Some datasets may require manual download or API keys.

## Troubleshooting

### Common Issues

**Issue: Module not found**
- Solution: Ensure you've activated the virtual environment and installed the package with `pip install -e .`

**Issue: CUDA out of memory**
- Solution: Reduce batch size in config files or use 8-bit quantization

**Issue: Transformers version mismatch**
- Solution: `pip install --upgrade transformers`

## Development Setup

Install additional development dependencies:

```bash
pip install pytest black ruff mypy jupyter
```

## Verification

Run tests to verify installation:

```bash
pytest tests/
```
