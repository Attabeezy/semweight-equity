# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**SWELM: Semantic Weighting for Equitable Language Modeling**

This is a Python-based research project focused on semantic weighting techniques for equitable language modeling.

## Project Structure

```
semweight-equity/
├── src/           # Main source code for the SWELM implementation
├── configs/       # Configuration files for experiments and model parameters
├── data/          # Dataset storage and preprocessing scripts
├── experiments/   # Experiment scripts and training pipelines
├── notebooks/     # Jupyter notebooks for analysis and visualization
├── requirements.txt
└── setup.py
```

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies (once requirements.txt is populated)
pip install -r requirements.txt

# Install package in editable mode (once setup.py is configured)
pip install -e .
```

### Testing
```bash
# Run all tests (standard pytest convention)
pytest

# Run specific test file
pytest tests/test_<module>.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test function
pytest tests/test_<module>.py::test_function_name
```

### Code Quality
```bash
# Format code with black (if added to dependencies)
black src/ tests/

# Lint with ruff (if added to dependencies)
ruff check src/ tests/

# Type checking with mypy (if added to dependencies)
mypy src/
```

### Running Experiments
```bash
# Run experiment scripts (typical pattern)
python experiments/<experiment_name>.py --config configs/<config_file>.yaml
```

### Jupyter Notebooks
```bash
# Start Jupyter server for exploratory analysis
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

## Architecture Notes

### Expected Code Organization

- **src/**: Core library code including:
  - Model implementations
  - Semantic weighting algorithms
  - Data processing utilities
  - Training and evaluation utilities

- **configs/**: YAML or JSON configuration files for:
  - Model hyperparameters
  - Dataset specifications
  - Experiment settings

- **data/**: Data pipeline components:
  - Raw data storage (typically gitignored)
  - Data preprocessing scripts
  - Dataset loaders

- **experiments/**: Executable experiment scripts:
  - Training scripts
  - Evaluation scripts
  - Ablation studies

- **notebooks/**: Jupyter notebooks for:
  - Exploratory data analysis
  - Results visualization
  - Model behavior investigation

### Development Workflow

1. **Add dependencies** to `requirements.txt` as you install packages
2. **Configure `setup.py`** with project metadata when packaging
3. **Create test files** in a `tests/` directory mirroring the `src/` structure
4. **Use notebooks/** for experimentation, then move stable code to `src/`
5. **Store configurations** in `configs/` rather than hardcoding parameters

## Windows-Specific Notes

- Use PowerShell (pwsh) as the primary shell
- Virtual environment activation: `.\venv\Scripts\Activate.ps1`
- If execution policy blocks scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Use forward slashes or raw strings in Python path handling for cross-platform compatibility
