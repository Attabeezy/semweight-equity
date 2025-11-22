# SWELM: Semantic Weighting for Equitable Language Modeling

A research implementation for improving multilingual language model performance through semantic weighting and adaptive sampling strategies.

## Overview

SWELM addresses the challenge of equitable performance across languages in multilingual NLP by:

1. **Semantic Weighting**: Computing language-specific weights based on representation quality
2. **Adaptive Sampling**: Dynamically adjusting training data distribution
3. **Cross-lingual Evaluation**: Testing on translation, QA, and reasoning tasks

## Features

- 🌍 Support for 200+ languages via FLORES-200
- 🤖 Compatible with Llama 3, GPT, and other LLMs
- 📊 Comprehensive evaluation metrics (BLEU, F1, Exact Match)
- 🔬 Ablation studies and baseline comparisons
- ⚡ GPU-optimized with 8-bit quantization support

## Quick Start

### Installation

```bash
git clone https://github.com/Attabeezy/semweight-equity.git
cd semweight-equity
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from src import SWELM

# Initialize SWELM
swelm = SWELM(
    encoder_model="xlm-roberta-base",
    alpha=0.5,
    beta=1.0
)

# Compute semantic weights
weights = swelm.compute_semantic_weights(texts, languages)
```

### Run Experiments

```bash
# Translation experiments (FLORES-200)
python experiments/translation.py --config configs/translation.yaml

# Question answering (TyDiQA)
python experiments/qa.py --config configs/qa.yaml

# Reasoning (StrategyQA)
python experiments/reasoning.py --config configs/reasoning.yaml
```

## Project Structure

```
semweight-equity/
├── src/                 # Core SWELM implementation
├── experiments/         # Experiment scripts
├── data/               # Data loaders and preprocessing
├── models/             # LLM interfaces and wrappers
├── configs/            # YAML configuration files
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
└── docs/               # Documentation
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Experiments Guide](docs/experiments.md)

## Datasets

- **FLORES-200**: Multilingual translation (200 languages)
- **TyDiQA**: Question answering (11 languages)
- **StrategyQA**: Multi-hop reasoning (translated to 7+ languages)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{swelm2024,
  title={SWELM: Semantic Weighting for Equitable Language Modeling},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue or contact [Attabeezy](mailto:attabeezy@gmail.com).
