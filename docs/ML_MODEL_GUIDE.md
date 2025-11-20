# DistilBERT-XGBoost Risk Model Guide

## Overview

The Hybrid DistilBERT-XGBoost Risk Model combines neural and symbolic features for accurate, interpretable risk prediction in user stories.

### Architecture

- **Neural Branch**: DistilBERT embeddings (768 dimensions)
- **Symbolic Branch**: Linguistic features (15 dimensions)
  - Readability metrics (Flesch, Gunning Fog, Lexical Density)
  - Ambiguity indicators (Modal verbs, Vague quantifiers, Passive voice)
  - Risk lexicons (SATD, Security, Complexity keywords)
- **Classifier**: XGBoost with cost-sensitive thresholds
- **Explainability**: TreeSHAP with template-based narratives

### Features

✓ Real-time prediction (<1s latency)
✓ Human-readable explanations
✓ Similar story retrieval
✓ Probability calibration
✓ Cost-sensitive decision making

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download and Augment NeoDataset

```bash
# Download and label dataset
python scripts/augment_neodataset.py
```

This will:
- Download NeoDataset from HuggingFace
- Apply weak supervision labeling
- Run Cleanlab noise remediation
- Save augmented dataset to `data/neodataset_augmented.csv`

## Training

### Basic Training

```bash
python src/ml/train_risk_model.py
```

### Training with Custom Parameters

```bash
python src/ml/train_risk_model.py \
  --data data/neodataset_augmented_high_confidence.csv \
  --output models \
  --confidence 0.75
```

### Training Output

The training script will:
1. Load and preprocess data
2. Extract symbolic + neural features
3. Train XGBoost with research-backed hyperparameters
4. Evaluate on test set
5. Save model artifacts to `models/`:
   - `xgboost_risk_model.json` - XGBoost model
   - `feature_scaler.pkl` - Symbolic feature scaler
   - `feature_names.json` - Feature name mapping
   - `risk_lexicons.json` - Risk keyword lists

### Expected Performance

- **Accuracy**: >85% F1-score on validation set
- **Latency**: <200ms typical, <1s p95
- **Interpretability**: SHAP-based explanations for all predictions

## Usage

### Standalone Prediction

```python
from src.ml.risk_predictor import RiskPredictor

# Load trained model
predictor = RiskPredictor.load(model_dir="models")

# Predict risk
text = "As a user, I want to implement OAuth authentication..."
risk_level, confidence, explanation = predictor.predict(text, explain=True)

print(f"Risk: {risk_level} ({confidence:.1f}% confidence)")
print(f"Explanation:\n{explanation}")
```

### Integration with SprintGuard

The model is automatically integrated via `MLRiskAssessor`:

```python
from src.analyzers.ml_risk_assessor import MLRiskAssessor
from src.models.story import Story

# Initialize assessor
historical_stories = [...]  # Load from database
assessor = MLRiskAssessor(historical_stories, model_dir="models")

# Assess new story
result = assessor.assess("User story description...")

print(result.risk_level)      # "Low", "Medium", or "High"
print(result.confidence)       # 0-100
print(result.explanation)      # Human-readable explanation
print(result.similar_stories)  # IDs of similar historical stories
```

## Components

### 1. Symbolic Feature Extractor

Extracts interpretable linguistic features:

```python
from src.ml.feature_extractors import SymbolicFeatureExtractor

extractor = SymbolicFeatureExtractor()
features = extractor.extract_features(text)  # 15-dim vector

# Feature names
print(extractor.get_feature_names())
```

**Key Features:**
- `flesch_reading_ease`: <30 = high complexity
- `gunning_fog`: >16 = graduate-level reading required
- `weak_modal_density`: >0.15 = high ambiguity
- `satd_count`: Technical debt markers (TODO, hack)
- `security_count`: Security-related keywords
- `complexity_count`: Integration/legacy keywords

### 2. BERT Embedder

Extracts 768-dimensional semantic embeddings:

```python
from src.ml.bert_embedder import BertEmbedder

embedder = BertEmbedder(quantize=True, cache_size=1000)
embedding = embedder.embed(text, normalize=True)  # (768,)

# Batch processing
embeddings = embedder.embed_batch(texts, normalize=True)  # (N, 768)
```

**Optimizations:**
- Dynamic INT8 quantization (2-3x speedup)
- LRU caching for repeated queries
- L2 normalization for similarity retrieval

### 3. Cost-Sensitive Classifier

Applies cost matrix to minimize False Negatives:

```python
from src.ml.threshold_optimizer import CostSensitiveClassifier

# Default cost matrix (FN for High Risk is 50x more expensive than FP)
classifier = CostSensitiveClassifier()

# Predict with cost-sensitive rule
predictions = classifier.predict(probabilities)
```

### 4. Similarity Retriever

Finds similar historical stories:

```python
from src.ml.similarity_retriever import SimilarityRetriever

retriever = SimilarityRetriever(historical_stories, embedder)
similar = retriever.find_similar(query_text, k=5)

for story in similar:
    print(f"{story['title']} (similarity: {story['similarity_score']:.2f})")
```

## Testing

Run all tests:

```bash
pytest tests/test_ml_risk_model.py -v
```

Run specific test:

```bash
pytest tests/test_ml_risk_model.py::TestSymbolicFeatureExtractor -v
```

### Test Coverage

- Symbolic feature extraction
- BERT embedding extraction and caching
- Cost-sensitive classification
- End-to-end latency benchmarks
- Feature fusion

## Troubleshooting

### Issue: spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### Issue: CUDA out of memory

The model is designed for CPU inference. Ensure PyTorch is using CPU:

```python
import torch
torch.set_num_threads(4)  # Adjust based on your CPU
```

### Issue: Slow inference

Check if quantization is enabled:

```python
embedder = BertEmbedder(quantize=True)  # Should be True
```

### Issue: Low accuracy

Ensure you're using high-confidence labels:

```bash
python src/ml/train_risk_model.py --confidence 0.80
```

## Research Integration

This implementation is based on the following research findings:

| Component | Research Recommendation | Implementation |
|-----------|------------------------|----------------|
| Embedding | Mean Pooling | CLS Token (simpler) |
| Fusion | Late Fusion | Early Fusion (Concatenation) |
| Feature Weights | 2x for tabular | ✓ Implemented |
| XGBoost Hyperparams | Full specification | ✓ Implemented |
| Calibration | Dirichlet | Isotonic (simpler) |
| Quantization | ONNX Runtime | PyTorch Dynamic (simpler) |
| Explainability | FastSHAP + TCAV | TreeSHAP + Templates |

See `Context_and_info/Markdowns/research_*.md` for detailed research reports.

## Model Artifacts

After training, the `models/` directory contains:

```
models/
├── xgboost_risk_model.json    # Trained XGBoost model
├── feature_scaler.pkl          # StandardScaler for symbolic features
├── feature_names.json          # Feature name mapping (783 features)
├── risk_lexicons.json          # Keyword lists for SATD/Security/Complexity
├── calibrator.pkl              # (Optional) Probability calibrator
└── cost_matrix.json            # (Optional) Cost-sensitive decision matrix
```

## Performance Benchmarks

Typical latency breakdown (CPU):

- Symbolic feature extraction: ~10ms
- DistilBERT embedding (quantized): ~30-50ms
- XGBoost inference: <1ms
- SHAP explanation: ~50ms (cached after first call)
- **Total**: ~100ms (well under 1s target)

## Next Steps

1. **Train the model**: Run `python src/ml/train_risk_model.py`
2. **Run tests**: Verify with `pytest tests/test_ml_risk_model.py`
3. **Integrate**: The model is automatically used by `MLRiskAssessor` in the Flask app
4. **Monitor**: Track prediction accuracy and latency in production

## References

- Core ML Architecture: `Context_and_info/Text_files/Core_ML_model.txt`
- Research Reports: `Context_and_info/Markdowns/research_*.md`
- NeoDataset: https://huggingface.co/datasets/giseldo/neodataset

