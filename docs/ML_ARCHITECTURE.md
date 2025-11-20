# ML Module: DistilBERT-XGBoost Risk Model

This module contains the hybrid neuro-symbolic risk assessment model.

## Module Structure

```
src/ml/
├── __init__.py
├── feature_extractors.py       # Symbolic linguistic features
├── bert_embedder.py            # DistilBERT embedding extraction
├── train_risk_model.py         # Training pipeline
├── calibration.py              # Probability calibration
├── threshold_optimizer.py      # Cost-sensitive classification
├── risk_predictor.py           # End-to-end prediction + SHAP
├── similarity_retriever.py     # Similar story retrieval
├── neodataset_loader.py        # Dataset loading utilities
├── labeling_functions.py       # Weak supervision labeling
├── weak_supervision_pipeline.py # Snorkel pipeline
├── cleanlab_pipeline.py        # Noise remediation
└── README.md                   # This file
```

## Quick Start

### 1. Train the Model

```bash
python src/ml/train_risk_model.py --data data/neodataset_augmented_high_confidence.csv
```

### 2. Use the Model

```python
from src.ml.risk_predictor import RiskPredictor

# Load trained model
predictor = RiskPredictor.load(model_dir="models")

# Predict risk
text = "As a user, I want to implement OAuth authentication..."
risk_level, confidence, explanation = predictor.predict(text, explain=True)

print(f"Risk: {risk_level} ({confidence:.1f}%)")
print(explanation)
```

## Component Details

### feature_extractors.py
**SymbolicFeatureExtractor**: Extracts 15 interpretable features
- Readability metrics (Flesch, Gunning Fog)
- Ambiguity indicators (modals, vague terms)
- Risk lexicons (SATD, security, complexity)

### bert_embedder.py
**BertEmbedder**: Fast, cached DistilBERT embeddings
- INT8 quantization (2-3x speedup)
- LRU caching
- Batch processing

### train_risk_model.py
**RiskModelTrainer**: Complete training pipeline
- Feature extraction (symbolic + neural)
- XGBoost training with hyperparameter tuning
- Model artifact saving

### risk_predictor.py
**RiskPredictor**: Production-ready prediction
- Real-time inference (<100ms)
- SHAP explanations
- Template-based narratives

### similarity_retriever.py
**SimilarityRetriever**: Find similar historical stories
- Dot product similarity on embeddings
- Fast retrieval (<5ms for 50k stories)

## Training Pipeline Flow

```
NeoDataset (augmented)
    ↓
Load & Filter (confidence > 0.75)
    ↓
Stratified Split (by project)
    ↓
Feature Extraction
    ├── Symbolic (15 features)
    └── DistilBERT (768 features)
    ↓
Feature Fusion (783 features)
    ↓
XGBoost Training
    ├── Feature weighting (2x symbolic)
    ├── Sample weighting (class balance)
    └── Early stopping (validation)
    ↓
Model Artifacts
    ├── xgboost_risk_model.json
    ├── feature_scaler.pkl
    ├── feature_names.json
    └── risk_lexicons.json
```

## Inference Pipeline Flow

```
User Story Text
    ↓
Feature Extraction (cached)
    ├── Symbolic (15)
    └── DistilBERT (768)
    ↓
XGBoost Prediction
    ↓
Calibration (Isotonic)
    ↓
Cost-Sensitive Decision
    ↓
TreeSHAP Explanation
    ↓
Risk Level + Confidence + Explanation
```

## Model Artifacts

After training, the `models/` directory contains:

- **xgboost_risk_model.json**: Trained XGBoost model
- **feature_scaler.pkl**: StandardScaler for symbolic features
- **feature_names.json**: Feature name mapping (783 features)
- **risk_lexicons.json**: Keyword lists for symbolic features
- **calibrator.pkl**: (Optional) Probability calibrator
- **cost_matrix.json**: (Optional) Cost-sensitive decision matrix

## Dependencies

Core dependencies:
- `transformers` - DistilBERT
- `torch` - PyTorch backend
- `xgboost` - Gradient boosting
- `shap` - Explainability
- `textstat` - Readability metrics
- `spacy` - NLP pipeline

Install with:
```bash
pip install transformers torch xgboost shap textstat spacy
python -m spacy download en_core_web_sm
```

## Performance Benchmarks

### Latency (CPU)
- Symbolic extraction: ~10ms
- DistilBERT (quantized): ~30-50ms
- XGBoost inference: <1ms
- SHAP computation: ~50ms (cached)
- **Total**: ~100ms

### Memory
- DistilBERT (quantized): ~200MB
- XGBoost model: ~1-5MB
- Historical embeddings (50k stories): ~150MB

### Accuracy
- Expected F1-score: >85%
- Precision for High Risk: >80%
- Recall for High Risk: >90% (cost-sensitive)

## Advanced Usage

### Custom Feature Extraction

```python
from src.ml.feature_extractors import SymbolicFeatureExtractor

extractor = SymbolicFeatureExtractor()
features = extractor.extract_features(text)
feature_names = extractor.get_feature_names()

for name, value in zip(feature_names, features):
    print(f"{name}: {value}")
```

### Batch Prediction

```python
from src.ml.bert_embedder import BertEmbedder

embedder = BertEmbedder(quantize=True)
embeddings = embedder.embed_batch(texts, normalize=True)
```

### Custom Cost Matrix

```python
from src.ml.threshold_optimizer import CostSensitiveClassifier
import numpy as np

# High penalty for missing High Risk stories
cost_matrix = np.array([
    [0, 2, 100],  # Predict Low
    [1, 0, 50],   # Predict Medium
    [3, 2, 0]     # Predict High
])

classifier = CostSensitiveClassifier(cost_matrix=cost_matrix)
predictions = classifier.predict(probabilities)
```

## Troubleshooting

### Issue: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue: Slow inference
- Ensure quantization is enabled: `BertEmbedder(quantize=True)`
- Check CPU threads: `torch.set_num_threads(4)`
- Use caching: embeddings are cached automatically

### Issue: Low accuracy
- Use high-confidence labels: `--confidence 0.80`
- Check class distribution in training data
- Tune hyperparameters with Optuna

## Testing

Run all tests:
```bash
pytest tests/test_ml_risk_model.py -v
```

Run specific test class:
```bash
pytest tests/test_ml_risk_model.py::TestSymbolicFeatureExtractor -v
```

## See Also

- Training guide: `docs/ML_MODEL_GUIDE.md`
- Implementation summary: `IMPLEMENTATION_SUMMARY.md`
- Research reports: `Context_and_info/Markdowns/research_*.md`

