#!/bin/bash
# Quick training script for DistilBERT-XGBoost Risk Model

set -e  # Exit on any error

echo "=========================================="
echo "DistilBERT-XGBoost Risk Model Training"
echo "=========================================="
echo ""

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set PYTHONPATH so 'src' module can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Step 1: Check if augmented dataset exists
if [ ! -f "data/neodataset_augmented_3class_high_confidence.csv" ]; then
    echo "⚠ Augmented dataset not found. Running augmentation pipeline..."
    python scripts/augment_neodataset.py
    echo ""
fi

# Step 2: Check dependencies
echo "Checking dependencies..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ spaCy model not found. Downloading..."
    python -m spacy download en_core_web_sm
    echo ""
fi

# Step 3: Train model
echo "Starting model training..."
echo ""
python src/ml/train_risk_model.py

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Model artifacts saved to models/"
echo "=========================================="

# Step 4: Run tests (optional, only if training succeeded)
echo ""
echo "Running tests..."
pytest tests/test_ml_risk_model.py -v
