#!/bin/bash
# Quick training script for DistilBERT-XGBoost Risk Model

echo "=========================================="
echo "DistilBERT-XGBoost Risk Model Training"
echo "=========================================="
echo ""

# Step 1: Check if augmented dataset exists
if [ ! -f "data/neodataset_augmented_high_confidence.csv" ]; then
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

# Step 4: Run tests
echo ""
echo "Running tests..."
pytest tests/test_ml_risk_model.py -v

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Model artifacts saved to models/"
echo "=========================================="

