# Experiments Guide

## Overview

SWELM includes three main experimental tasks:

1. **Translation** - FLORES-200 multilingual translation
2. **Question Answering** - TyDiQA multilingual QA
3. **Reasoning** - StrategyQA multilingual reasoning

## Translation Experiments

### Dataset: FLORES-200

FLORES-200 is a many-to-many multilingual translation benchmark covering 200 languages.

### Configuration

```yaml
# configs/translation.yaml
data:
  flores_dir: "data/raw/flores200"
  languages:
    - eng_Latn  # English
    - spa_Latn  # Spanish
    - fra_Latn  # French
    # ... more languages
  source_lang: "eng_Latn"

swelm:
  alpha: 0.6
  beta: 1.2

generation:
  max_new_tokens: 256
  temperature: 0.3
  num_beams: 4
```

### Running

```bash
python experiments/translation.py \
    --config configs/translation.yaml \
    --output results/translation.json
```

### Expected Results

Results include BLEU and chrF scores for each language pair.

---

## Question Answering Experiments

### Dataset: TyDiQA

TyDiQA is a question answering dataset covering 11 typologically diverse languages.

### Configuration

```yaml
# configs/qa.yaml
data:
  tydiqa_dir: "data/raw/tydiqa"
  languages:
    - en  # English
    - ar  # Arabic
    - bn  # Bengali
    # ... more languages
  include_context: true

swelm:
  alpha: 0.55
  beta: 1.1

generation:
  max_new_tokens: 128
  temperature: 0.1
  do_sample: false
```

### Running

```bash
python experiments/qa.py \
    --config configs/qa.yaml \
    --output results/qa.json
```

### Expected Results

Results include exact match and F1 scores per language.

---

## Reasoning Experiments

### Dataset: StrategyQA

StrategyQA requires multi-hop reasoning for yes/no questions.

### Configuration

```yaml
# configs/reasoning.yaml
data:
  strategyqa_dir: "data/raw/strategyqa"
  languages:
    - en  # English
    - es  # Spanish
    # ... more languages
  use_chain_of_thought: true

swelm:
  alpha: 0.5
  beta: 1.0

generation:
  max_new_tokens: 512
  temperature: 0.7
  do_sample: true
```

### Running

```bash
python experiments/reasoning.py \
    --config configs/reasoning.yaml \
    --output results/reasoning.json
```

### Expected Results

Results include accuracy and exact match for yes/no predictions.

---

## Baseline Comparisons

Compare SWELM against baseline methods:

### Available Baselines

1. **Uniform Sampling** - Equal weight for all languages
2. **Temperature Sampling** - Simple temperature-based sampling
3. **Diversity Sampling** - Maximize language diversity

### Running Baselines

```bash
python experiments/baselines.py \
    --config configs/default.yaml \
    --output results/baselines.json
```

---

## Ablation Studies

Study the impact of SWELM components.

### Alpha Ablation

Test different values of the semantic distance weight (α):

```bash
python experiments/ablations.py \
    --config configs/default.yaml \
    --study alpha \
    --output results/alpha_ablation.json
```

Tests α ∈ [0.0, 0.1, 0.2, ..., 1.0]

### Beta Ablation

Test different values of the representation quality weight (β):

```bash
python experiments/ablations.py \
    --config configs/default.yaml \
    --study beta \
    --output results/beta_ablation.json
```

Tests β ∈ [0.5, 1.0, 1.5, 2.0]

### Encoder Comparison

Compare different multilingual encoders:

```bash
python experiments/ablations.py \
    --config configs/default.yaml \
    --study encoder \
    --output results/encoder_ablation.json
```

Tests:
- XLM-RoBERTa
- mBERT
- LaBSE

---

## Running All Experiments

Execute all experiments sequentially:

```bash
bash scripts/run_experiments.sh
```

This runs:
1. Translation experiments
2. QA experiments
3. Reasoning experiments
4. Baseline comparisons
5. Ablation studies

Results are saved in `results/` directory.

---

## Analyzing Results

### Loading Results

```python
import json

with open('results/translation.json', 'r') as f:
    results = json.load(f)

print(results['metrics'])
```

### Visualization

Use the analysis notebook:

```bash
jupyter notebook notebooks/05_analysis.ipynb
```

This provides:
- Per-language performance plots
- Baseline comparisons
- Ablation study visualizations
- Statistical significance tests

---

## Reproducing Results

To reproduce results from the paper:

1. Download all datasets:
   ```bash
   bash scripts/download_data.sh
   ```

2. Run all experiments:
   ```bash
   bash scripts/run_experiments.sh
   ```

3. Analyze results:
   ```bash
   jupyter notebook notebooks/05_analysis.ipynb
   ```

---

## Custom Experiments

### Creating Custom Configs

```yaml
# configs/my_experiment.yaml
defaults:
  - default

data:
  languages:
    - eng_Latn
    - zho_Hans
    - ara_Arab

swelm:
  alpha: 0.65
  beta: 1.15
```

### Running Custom Experiments

```bash
python experiments/translation.py \
    --config configs/my_experiment.yaml \
    --output results/my_experiment.json \
    --seed 42
```
