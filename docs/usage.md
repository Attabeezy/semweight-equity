# Usage Guide

## Quick Start

### Basic SWELM Usage

```python
from src import SWELM, MultilingualEncoder

# Initialize SWELM
swelm = SWELM(
    encoder_model="xlm-roberta-base",
    alpha=0.5,
    beta=1.0
)

# Compute semantic weights
texts = ["Hello world", "Bonjour monde", "Hola mundo"]
languages = ["en", "fr", "es"]
weights = swelm.compute_semantic_weights(texts, languages)
```

### Running Experiments

#### Translation Experiments

```bash
python experiments/translation.py \
    --config configs/translation.yaml \
    --output results/translation.json \
    --seed 42
```

#### QA Experiments

```bash
python experiments/qa.py \
    --config configs/qa.yaml \
    --output results/qa.json
```

#### Reasoning Experiments

```bash
python experiments/reasoning.py \
    --config configs/reasoning.yaml \
    --output results/reasoning.json
```

### Run All Experiments

```bash
bash scripts/run_experiments.sh
```

## Configuration

### Basic Configuration

Edit configuration files in `configs/` to customize experiments:

```yaml
# configs/default.yaml
swelm:
  alpha: 0.5      # Semantic distance weight
  beta: 1.0       # Representation quality weight
  sampling_strategy: "proportional"
  
model:
  encoder: "xlm-roberta-base"
  llm: "meta-llama/Meta-Llama-3-8B"
```

### Advanced Configuration

Create custom config files for specific experiments:

```yaml
# configs/my_experiment.yaml
defaults:
  - default

swelm:
  alpha: 0.7
  beta: 1.5

data:
  languages:
    - en
    - zh
    - ar
```

## Using the LLM Interface

```python
from models.llama_wrapper import Llama3Wrapper

# Initialize model
llm = Llama3Wrapper(
    model_name="meta-llama/Meta-Llama-3-8B",
    load_in_8bit=True  # Use 8-bit quantization
)

# Generate responses
prompts = ["Translate to Spanish: Hello"]
responses = llm.generate(prompts, max_length=128)
```

## Adaptive Sampling

```python
from src.adaptive import AdaptiveSampler
import numpy as np

# Initialize sampler
sampler = AdaptiveSampler(
    sampling_strategy="proportional",
    temperature=1.0
)

# Sample data based on weights
data = [{"text": f"sample{i}"} for i in range(100)]
weights = np.random.rand(100)

batch = sampler.sample_batch(data, weights, batch_size=32)
```

## Evaluation

```python
from src.metrics import evaluate_performance

predictions = ["answer1", "answer2", "answer3"]
references = ["answer1", "answer2", "different"]
languages = ["en", "es", "fr"]

results = evaluate_performance(
    predictions,
    references,
    languages=languages,
    metrics=["exact_match", "f1"]
)

print(results)
# Output: {'exact_match': 0.667, 'f1': 0.889, 'en_exact_match': 1.0, ...}
```

## Jupyter Notebooks

Start Jupyter for interactive exploration:

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `01_demo.ipynb` - Quick start demonstration
- `02_translation.ipynb` - Translation experiments
- `03_qa.ipynb` - QA experiments
- `04_reasoning.ipynb` - Reasoning experiments
- `05_analysis.ipynb` - Results analysis

## Ablation Studies

Run ablation studies on specific parameters:

```bash
# Alpha ablation
python experiments/ablations.py \
    --config configs/default.yaml \
    --study alpha \
    --output results/alpha_ablation.json

# Beta ablation
python experiments/ablations.py \
    --config configs/default.yaml \
    --study beta \
    --output results/beta_ablation.json

# Encoder comparison
python experiments/ablations.py \
    --config configs/default.yaml \
    --study encoder \
    --output results/encoder_ablation.json
```

## Baseline Comparisons

Compare SWELM against baseline methods:

```bash
python experiments/baselines.py \
    --config configs/default.yaml \
    --output results/baselines.json
```

## Working with Results

Results are saved as JSON files in the `results/` directory:

```python
import json

# Load results
with open('results/translation.json', 'r') as f:
    results = json.load(f)

# Analyze metrics
print(f"Overall BLEU: {results['metrics']['bleu']}")
print(f"Per-language BLEU: {results['metrics']['per_language_bleu']}")
```
