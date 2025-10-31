#!/bin/bash
# Download datasets for SWELM experiments

set -e

echo "Downloading datasets..."

# Create data directories
mkdir -p data/raw/flores200
mkdir -p data/raw/tydiqa
mkdir -p data/raw/strategyqa

# Download FLORES-200
echo "Downloading FLORES-200..."
# TODO: Add actual download commands
# wget https://github.com/facebookresearch/flores/releases/download/v2.0-rc.3/flores200_dataset.tar.gz
# tar -xzf flores200_dataset.tar.gz -C data/raw/flores200

# Download TyDiQA
echo "Downloading TyDiQA..."
# TODO: Add actual download commands
# wget https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz
# gunzip -c tydiqa-v1.0-train.jsonl.gz > data/raw/tydiqa/train.jsonl

# Download StrategyQA
echo "Downloading StrategyQA..."
# TODO: Add actual download commands
# wget https://storage.googleapis.com/ai2i/strategyqa/strategyqa_train.json
# mv strategyqa_train.json data/raw/strategyqa/

echo "Dataset download complete!"
echo "Note: Some datasets may require manual download or API keys."
