#!/bin/bash
# Quick training script for DistilBERT-XGBoost Risk Model
# Generates model artifacts + publication-quality visualizations

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

# Disable Python output buffering (needed for tmux/piped output)
export PYTHONUNBUFFERED=1

# Parse arguments
SKIP_TSNE=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tsne)
            SKIP_TSNE="--skip-tsne"
            shift
            ;;
        --fast)
            # Fast mode: skip t-SNE (can add more speed optimizations)
            SKIP_TSNE="--skip-tsne"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

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
python src/ml/train_risk_model.py $SKIP_TSNE $EXTRA_ARGS

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Model artifacts:  models/"
echo "  Visualizations:   visualizations/"
echo ""
echo "Generated visualizations (for paper):"
echo "  - confusion_matrix.pdf      - Prediction accuracy"
echo "  - calibration_plot.pdf      - Model confidence reliability"
echo "  - roc_curves.pdf            - ROC curves with AUC"
echo "  - precision_recall_curves.pdf - PR curves"
echo "  - feature_importance.pdf    - XGBoost feature importance"
echo "  - learning_curves.pdf       - Training convergence"
echo "  - class_distribution.pdf    - Data split balance"
echo "  - shap_summary.pdf          - SHAP beeswarm plot"
echo "  - shap_importance.pdf       - SHAP feature importance"
echo "  - shap_waterfall_high_risk.pdf - Example explanation"
if [ -z "$SKIP_TSNE" ]; then
    echo "  - embeddings_tsne.pdf       - Feature space visualization"
fi
echo "=========================================="

# Step 4: Run tests (optional, only if training succeeded)
echo ""
echo "Running tests..."
pytest tests/test_ml_risk_model.py -v
