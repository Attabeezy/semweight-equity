#!/bin/bash
# Run all SWELM experiments

set -e

echo "Running SWELM experiments..."

# Create results directory
mkdir -p results

# Run translation experiments
echo "Running translation experiments..."
python experiments/translation.py \
    --config configs/translation.yaml \
    --output results/translation.json

# Run QA experiments
echo "Running QA experiments..."
python experiments/qa.py \
    --config configs/qa.yaml \
    --output results/qa.json

# Run reasoning experiments
echo "Running reasoning experiments..."
python experiments/reasoning.py \
    --config configs/reasoning.yaml \
    --output results/reasoning.json

# Run baseline comparisons
echo "Running baseline comparisons..."
python experiments/baselines.py \
    --config configs/default.yaml \
    --output results/baselines.json

# Run ablation studies
echo "Running ablation studies..."
python experiments/ablations.py \
    --config configs/default.yaml \
    --output results/ablations.json

echo "All experiments complete!"
echo "Results saved in results/"
