#!/bin/bash
# Setup development environment for SWELM

set -e

echo "Setting up SWELM environment..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing SWELM package..."
pip install -e .

# Download spaCy models if needed
# python -m spacy download en_core_web_sm

echo "Environment setup complete!"
echo "Activate the environment with:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  .\\venv\\Scripts\\Activate.ps1  (PowerShell)"
    echo "  .\\venv\\Scripts\\activate.bat  (CMD)"
else
    echo "  source venv/bin/activate"
fi
