# SprintGuard Setup Guide

Complete step-by-step guide to set up SprintGuard from scratch.

---

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4GB+ RAM (for NeoDataset processing)
- Internet connection (for downloading NeoDataset)

---

## Step 1: Install Core Dependencies

Install Flask and basic dependencies:

```bash
cd /home/jovyan/SprintGuard
pip install -r requirements.txt
```

**What gets installed:**
- Flask 3.0.3 (web framework)
- Flask-CORS (API access)
- pandas, numpy (data processing)

---

## Step 2: Install ML Dependencies

Install Snorkel, Cleanlab, and related ML libraries:

```bash
pip install -r requirements-ml.txt
```

**What gets installed:**
- snorkel==0.9.9 (weak supervision)
- cleanlab==2.6.0 (noise detection)
- datasets==2.14.0 (HuggingFace datasets)
- scikit-learn (ML utilities)

---

## Step 3: Run NeoDataset Augmentation

This is a **one-time setup** that generates the labeled dataset:

```bash
python scripts/augment_neodataset.py
```

**What happens:**
1. Downloads NeoDataset (~20K user stories) from HuggingFace
2. Preprocesses text and extracts features
3. Applies 18 research-backed labeling functions
4. Trains Snorkel label model to aggregate votes
5. Uses Cleanlab to detect and remove noisy labels
6. Generates final augmented dataset

**Expected Runtime:** 15-30 minutes (depending on hardware)

**Output Files:**
- `data/neodataset_augmented.csv` - Full labeled dataset
- `data/neodataset_augmented_high_confidence.csv` - High-confidence subset
- `data/neodataset/neodataset_snorkel_labels.csv` - Intermediate results

**Success Indicators:**
- Label Health Score > 0.60
- ~10,000-15,000 high-confidence stories
- Label distribution roughly 30-70% (RISK vs SAFE)

---

## Step 4: (Optional) Train ML Model

**Note:** This step is a placeholder. The ML model training logic needs to be implemented.

Once you have the augmented dataset, you can train any ML classifier:

### Option A: TF-IDF + Logistic Regression (Recommended for PoC)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Load high-confidence subset
df = pd.read_csv('data/neodataset_augmented_high_confidence.csv')

# Prepare data
X = df['description']
y = df['risk_label_binary']  # 1=RISK, 0=SAFE

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_vec, y)

# Save model
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'models/risk_model.pkl')
print("✓ Model saved to models/risk_model.pkl")
```

### Option B: Deep Learning (BERT, etc.)
Train a transformer-based model for better performance. See `Core_ML_model.txt` for details.

### Option C: Ensemble Methods
Use XGBoost, Random Forest, or other ensemble methods.

---

## Step 5: Implement Model Prediction

Update `src/analyzers/ml_risk_assessor.py` to implement the `assess()` method:

```python
def assess(self, description: str) -> RiskResult:
    if self.model is None:
        return self._placeholder_response()
    
    # Extract features
    features = self.vectorizer.transform([description])
    
    # Get prediction
    risk_prob = self.model.predict_proba(features)[0]
    risk_label = self.model.predict(features)[0]
    
    # Map to risk levels
    if risk_label == 1:  # RISK
        risk_level = "High" if risk_prob[1] > 0.8 else "Medium"
        confidence = risk_prob[1] * 100
    else:  # SAFE
        risk_level = "Low"
        confidence = risk_prob[0] * 100
    
    return RiskResult(
        risk_level=risk_level,
        confidence=confidence,
        explanation=f"ML prediction (confidence: {confidence:.1f}%)"
    )
```

---

## Step 6: Start SprintGuard

Run the Flask application:

```bash
python app.py
```

**Expected Output:**
```
======================================================================
SprintGuard - Loading Data...
======================================================================
✓ Loaded 20000 stories from neodataset_augmented.csv
⚠ ML model not yet trained. Using placeholder responses.
  Expected model path: models/risk_model.pkl
======================================================================
SprintGuard PoC Starting...
======================================================================
Risk Assessor: MLRiskAssessor (No model loaded)
Historical Stories: 20000
Server: http://0.0.0.0:5001
======================================================================
```

Open your browser and navigate to: **http://localhost:5001**

---

## Troubleshooting

### Error: "Augmented NeoDataset not found"
**Solution:** Run the augmentation pipeline:
```bash
python scripts/augment_neodataset.py
```

### Error: "ImportError: No module named 'snorkel'"
**Solution:** Install ML dependencies:
```bash
pip install -r requirements-ml.txt
```

### Error: "Low label health score (<0.50)"
**Solution:** 
- Review LF statistics in augmentation output
- Consider refining labeling functions
- Check if NeoDataset matches your domain

### Memory Issues
**Solution:**
- NeoDataset requires ~4GB RAM
- For testing, use a subset: modify `load_neodataset()` to return `df.head(1000)`

### Slow Download
**Info:**
- HuggingFace datasets are cached in `data/neodataset/`
- Subsequent runs will be much faster

---

## Architecture Overview

### Data Flow
```
NeoDataset (HuggingFace)
    ↓
[scripts/augment_neodataset.py]
    ↓
Preprocessing → Labeling Functions → Snorkel → Cleanlab
    ↓
data/neodataset_augmented.csv
    ↓
[Train ML Model]
    ↓
models/risk_model.pkl
    ↓
[app.py → MLRiskAssessor]
    ↓
Risk Predictions via API
```

### Plug-and-Play Risk Assessor

SprintGuard uses the **RiskAssessorInterface** pattern:

1. Any ML implementation must inherit from `RiskAssessorInterface`
2. Implement the `assess(description: str) -> RiskResult` method
3. Drop in to `app.py` without changing other code

This allows you to:
- Swap ML models easily
- Compare different algorithms
- Upgrade to better models without refactoring

---

## Next Steps

1. **Validate Labels** (Optional)
   - Create manual validation set
   - Calculate precision/recall
   - See `README_NEODATASET_AUGMENTATION.md` for details

2. **Train ML Model**
   - Use augmented dataset
   - Experiment with different algorithms
   - Save model to `models/risk_model.pkl`

3. **Integrate Model**
   - Implement `MLRiskAssessor.assess()`
   - Test with real user stories
   - Iterate based on feedback

4. **Deploy**
   - Set up production environment
   - Configure proper logging
   - Add monitoring and alerts

---

## Additional Resources

- **Augmentation Methodology**: `README_NEODATASET_AUGMENTATION.md`
- **ML Model Details**: `Core_ML_model.txt`
- **Augmentation Status**: `docs/AUGMENTATION_STATUS.md`
- **Main README**: `README.md`

---

## Support

For issues or questions:
1. Check this setup guide
2. Review troubleshooting section
3. Check documentation in `docs/`
4. Review augmentation pipeline output for errors

---

**You're all set! Start building risk-aware sprints with SprintGuard.** 🛡️

