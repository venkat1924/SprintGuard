# SprintGuard Project Flow - Complete Sequential Guide

**Last Updated**: November 20, 2025  
**Purpose**: End-to-end flow from raw NeoDataset to production risk assessment API

---

## 🚨 CRITICAL COMPATIBILITY ISSUE 🚨

**BLOCKER FOUND**: There is a mismatch between the augmentation pipeline output and ML training script input:

- **Augmentation Output**: Binary labels (`'SAFE'` or `'RISK'`)
- **Training Script Expects**: 3-class labels (`'Low'`, `'Medium'`, `'High'`)

**Impact**: The training script will FAIL when trying to map labels:
```python
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_class'] = df['risk_label'].map(label_map)  # Will produce NaN values!
```

**Fix Required Before Training**:
1. **Option A**: Modify augmentation to output 3-class labels (requires new labeling functions)
2. **Option B**: Modify training script to work with binary labels ('SAFE'/'RISK' → 0/1)
3. **Option C**: Add a post-augmentation step to map SAFE→Low, RISK→High

**Recommended**: Option B (simplify to binary classification)

---

## Overview

SprintGuard follows a 4-stage pipeline:

```
Stage 1: NeoDataset Download → Preprocessed DataFrame
         ↓
Stage 2: Weak Supervision (Snorkel) → Labeled DataFrame
         ↓
Stage 3: Noise Filtering (Cleanlab) → Clean Augmented Dataset
         ↓
Stage 4: ML Model Training → Trained Model Artifacts
         ↓
Stage 5: Flask API Server → Production Risk Assessment
```

---

## Stage 1: NeoDataset Download & Preprocessing

### Purpose
Download and prepare the NeoDataset from HuggingFace for labeling.

### Commands
```bash
# Ensure dependencies installed
pip install -r requirements-augmentation.txt

# Run augmentation script (includes download)
python scripts/augment_neodataset.py
```

### What Happens
**File**: `src/ml/neodataset_loader.py`

1. **Downloads dataset** from HuggingFace (`giseldo/neodataset`)
2. **Caches locally** to `data/neodataset/` (auto-created)
3. **Filters** invalid stories (missing title, description, or story_points)
4. **Renames** `weight` → `story_points`
5. **Adds computed columns**:
   - `full_text` = title + ' ' + description
   - `word_count`, `title_word_count`, `char_count`
   - `has_list`, `list_item_count` (detects bullet points)
   - `has_code_block` (detects ``` markers)

### Outputs
**In-memory DataFrame** with columns:
- `id`, `title`, `description`
- `story_points` (renamed from weight)
- `project_id`, `state`, `created` (metadata)
- `full_text` (computed)
- `word_count`, `char_count`, `title_word_count` (computed)
- `has_list`, `list_item_count`, `has_code_block` (computed)

**Example Row**:
```python
{
    'id': 12345,
    'title': 'Implement user authentication',
    'description': 'Add login and signup functionality...',
    'story_points': 5,
    'full_text': 'Implement user authentication Add login...',
    'word_count': 42,
    'has_list': True
}
```

### Expected Size
- **~25,000+ stories** from NeoDataset
- **~20,000+ valid stories** after filtering

---

## Stage 2: Weak Supervision with Snorkel

### Purpose
Generate probabilistic risk labels using research-backed labeling functions (LFs).

### What Happens
**File**: `src/ml/weak_supervision_pipeline.py`

1. **Applies labeling functions** (from `src/ml/labeling_functions.py`)
   - Each LF votes: `-1` (ABSTAIN), `0` (SAFE), `1` (RISK)
   - Creates label matrix: `(n_stories × n_LFs)`
   
2. **Trains Snorkel label model**
   - Learns LF accuracies and correlations
   - No ground truth needed (data programming)
   - 500 epochs of training
   
3. **Generates probabilistic labels**
   - Hard labels: 0 (SAFE) or 1 (RISK)
   - Soft labels: P(SAFE) and P(RISK)
   - Confidence: max(P(SAFE), P(RISK))

### Outputs
**DataFrame** (extends Stage 1 output) with **NEW COLUMNS**:
- `risk_label_binary`: `0` (SAFE) or `1` (RISK)
- `risk_label`: `'SAFE'` or `'RISK'` ← **String version**
- `risk_prob_safe`: Probability of SAFE (0.0-1.0)
- `risk_prob_risk`: Probability of RISK (0.0-1.0)
- `risk_confidence`: max(P(SAFE), P(RISK))

**Example Row** (showing only new columns):
```python
{
    'risk_label_binary': 1,
    'risk_label': 'RISK',
    'risk_prob_safe': 0.23,
    'risk_prob_risk': 0.77,
    'risk_confidence': 0.77
}
```

**Intermediate Save**:
```
data/neodataset/neodataset_snorkel_labels.csv
```

---

## Stage 3: Noise Remediation with Cleanlab

### Purpose
Detect and filter label errors from Snorkel output using Confident Learning.

### What Happens
**File**: `src/ml/cleanlab_pipeline.py`

1. **Trains preliminary classifier**
   - TF-IDF + Logistic Regression
   - Gets out-of-sample predictions via 5-fold CV
   - Avoids overfitting
   
2. **Detects label issues**
   - Compares predicted probs vs assigned labels
   - Flags mismatches (e.g., 99% confident SAFE but labeled RISK)
   
3. **Filters dataset**
   - Removes ~5-15% of stories with label errors
   - Keeps clean, high-quality labels
   
4. **Calculates health score**
   - Range: 0.0-1.0 (higher is better)
   - >0.8 = Excellent
   - 0.6-0.8 = Good
   - <0.6 = Consider refining LFs

### Outputs
**Two CSV Files**:

#### 1. Full Augmented Dataset
**File**: `data/neodataset_augmented.csv`

**Columns** (all from Stages 1-2, minus noisy rows):
- Original NeoDataset columns (id, title, description, story_points, etc.)
- Computed columns (full_text, word_count, etc.)
- Risk labels (risk_label_binary, risk_label, risk_prob_safe, risk_prob_risk, risk_confidence)

**Size**: ~17,000-20,000 stories (after filtering ~10-15% noise)

#### 2. High-Confidence Subset
**File**: `data/neodataset_augmented_high_confidence.csv`

**Filter**: `risk_confidence > 0.75`

**Purpose**: For ML model training (higher quality labels)

**Size**: ~10,000-15,000 stories

### Console Output
```
Total stories processed: 20,453
Stories with clean labels: 18,765
High-confidence stories: 12,304
Label health score: 0.823

Final label distribution:
SAFE    11,234 (59.8%)
RISK     7,531 (40.2%)
```

---

## Stage 4: ML Model Training

### Purpose
Train hybrid DistilBERT-XGBoost model for risk prediction.

### Prerequisites
```bash
# Install ML dependencies (includes torch, transformers, etc.)
pip install -r requirements-ml.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Commands
```bash
# Train on high-confidence subset (default)
python src/ml/train_risk_model.py

# OR specify custom data
python src/ml/train_risk_model.py --data data/neodataset_augmented.csv --confidence 0.8

# OR use full dataset
python src/ml/train_risk_model.py --data data/neodataset_augmented.csv --confidence 0.0
```

### What Happens
**File**: `src/ml/train_risk_model.py`

1. **Loads augmented data**
   - Reads CSV (default: `data/neodataset_augmented_high_confidence.csv`)
   - Filters by confidence threshold (default: 0.75)
   - Combines title + description → `full_text`
   
2. **⚠️ Maps labels to integers** (THIS IS THE PROBLEM!)
   ```python
   label_map = {'Low': 0, 'Medium': 1, 'High': 2}  # ❌ WRONG!
   df['risk_class'] = df['risk_label'].map(label_map)
   
   # But risk_label contains 'SAFE' or 'RISK', not 'Low'/'Medium'/'High'
   # This will create NaN values!
   ```
   
3. **Extracts hybrid features** (783 total)
   - **Symbolic features** (15 features): `src/ml/feature_extractors.py`
     - Text statistics (word count, readability, etc.)
     - Risk keywords (SATD, security, complexity)
     - Linguistic patterns (passive voice, modals, vague terms)
   - **DistilBERT embeddings** (768 features): `src/ml/bert_embedder.py`
     - Semantic representation
     - Quantized for speed (8-bit)
     - Cached for efficiency
   
4. **Splits data**
   - Train: 70%
   - Validation: 15%
   - Test: 15%
   
5. **Trains XGBoost**
   - Multi-class classification (3 classes)
   - Cost-sensitive learning (class weights)
   - Early stopping on validation set
   - 500 max rounds, stops if no improvement for 50 rounds
   
6. **Evaluates on test set**
   - Accuracy, precision, recall, F1
   - Confusion matrix
   - Classification report
   
7. **Saves model artifacts**

### Outputs
**Directory**: `models/` (auto-created)

**Files Created**:
1. `xgboost_risk_model.json` - Trained XGBoost model
2. `feature_scaler.pkl` - StandardScaler for symbolic features
3. `feature_names.json` - List of 783 feature names
4. `risk_lexicons.json` - SATD/security/complexity keywords

### Expected Console Output
```
Loading data from data/neodataset_augmented_high_confidence.csv...
  Total stories: 12,304
  High-confidence stories (>0.75): 12,304

Class distribution:
SAFE    7,321 (59.5%)
RISK    4,983 (40.5%)

Extracting features for 12,304 stories...
  [1/2] Extracting symbolic features...
  [2/2] Extracting DistilBERT embeddings...

Training XGBoost model...
[0] train-mlogloss:0.8234 val-mlogloss:0.8456
[50] train-mlogloss:0.3421 val-mlogloss:0.4123
...
✓ Training complete (best iteration: 287)

Evaluating on test set...
  Accuracy: 0.8234

Saving model artifacts to models/...
  ✓ Saved: models/xgboost_risk_model.json
  ✓ Saved: models/feature_scaler.pkl
  ✓ Saved: models/feature_names.json
  ✓ Saved: models/risk_lexicons.json
```

### Training Time
- **~10-30 minutes** (depends on hardware)
- Most time: DistilBERT embeddings (~5-15 min)
- XGBoost training: ~2-5 min

---

## Stage 5: Flask API Server (Production Use)

### Purpose
Serve trained model via REST API for real-time risk assessment.

### Prerequisites
```bash
# Core dependencies only (Flask)
pip install -r requirements.txt

# MUST have completed Stages 1-4:
# ✓ data/neodataset_augmented.csv exists
# ✓ models/xgboost_risk_model.json exists
```

### Commands
```bash
# Start server (default: http://0.0.0.0:5001)
python app.py
```

### What Happens
**File**: `app.py`

1. **Loads augmented dataset**
   - Uses `CSVDataLoader` (src/data_loader.py)
   - Reads `data/neodataset_augmented.csv`
   - Converts rows to `Story` objects
   
2. **Initializes ML Risk Assessor**
   - `MLRiskAssessor` (src/analyzers/ml_risk_assessor.py)
   - Loads model from `models/` directory
   - Uses `RiskPredictor` (src/ml/risk_predictor.py)
   - Sets up `SimilarityRetriever` for similar story search
   
3. **Starts Flask server**
   - Listens on `0.0.0.0:5001`
   - Exposes REST API endpoints

### Expected Startup Output
```
======================================================================
SprintGuard - Loading Data...
======================================================================
✓ Loaded 18,765 stories from neodataset_augmented.csv
Initializing feature extractors...
✓ Model found at models
✓ ML Risk Assessor ready
============================================================
SprintGuard PoC Starting...
============================================================
Risk Assessor: DistilBERT-XGBoost Hybrid Model
Historical Stories: 18765
Server: http://0.0.0.0:5001
============================================================
 * Running on http://0.0.0.0:5001
```

### API Endpoints

#### 1. Health Check
```bash
GET /api/health-check
```

**Response**:
```json
{
  "success": true,
  "data": {
    "health_score": 85.5,
    "total_stories": 18765,
    "recommendations": [...]
  }
}
```

#### 2. Risk Assessment (Main Feature)
```bash
POST /api/assess-risk
Content-Type: application/json

{
  "description": "Implement user authentication with OAuth2"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_level": "High",
    "confidence": 78.3,
    "explanation": "**Risk Assessment**\nHigh risk detected (78% confidence)...",
    "similar_stories": [123, 456, 789]
  }
}
```

#### 3. Scope Simulation
```bash
POST /api/simulate-scope
Content-Type: application/json

{
  "current_end_date": "2025-12-15",
  "current_story_points": 20,
  "new_story_points": 5,
  "team_velocity": 2.5
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "original_end_date": "2025-12-15",
    "new_end_date": "2025-12-17",
    "days_added": 2,
    "impact_summary": "...",
    "recommendations": [...]
  }
}
```

#### 4. Get Stories (Debug)
```bash
GET /api/stories?risk_level=RISK&limit=10
```

**Response**:
```json
{
  "success": true,
  "data": {
    "stories": [...],
    "count": 10
  }
}
```

#### 5. System Info
```bash
GET /api/info
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_assessor": "DistilBERT-XGBoost Hybrid Model",
    "historical_story_count": 18765,
    "version": "1.0.0-PoC"
  }
}
```

---

## Data Flow Diagram

```
HuggingFace Dataset
  │ giseldo/neodataset
  │ ~25,000 stories
  │
  ▼
[Stage 1: Download & Preprocess]
  │ src/ml/neodataset_loader.py
  │ Adds: full_text, word_count, has_list, etc.
  │
  ▼
In-Memory DataFrame
  │ Columns: id, title, description, story_points, full_text, ...
  │ ~20,000 valid stories
  │
  ▼
[Stage 2: Weak Supervision]
  │ src/ml/weak_supervision_pipeline.py
  │ Applies labeling functions via Snorkel
  │ Adds: risk_label_binary, risk_label, risk_confidence
  │
  ▼
data/neodataset/neodataset_snorkel_labels.csv
  │ Columns: [...original], risk_label ('SAFE'/'RISK'), risk_confidence
  │ ~20,000 stories with probabilistic labels
  │
  ▼
[Stage 3: Noise Filtering]
  │ src/ml/cleanlab_pipeline.py
  │ Removes ~10-15% noisy labels
  │ Filters by confidence threshold
  │
  ├──▶ data/neodataset_augmented.csv
  │    ALL clean stories (~18,000)
  │    Columns: [...original], risk_label, risk_confidence
  │
  └──▶ data/neodataset_augmented_high_confidence.csv
       HIGH-CONFIDENCE subset (~12,000)
       Filter: risk_confidence > 0.75
       ↓
[Stage 4: ML Training]
  │ src/ml/train_risk_model.py
  │ ⚠️ EXPECTS: risk_label in {'Low', 'Medium', 'High'}
  │ ❌ RECEIVES: risk_label in {'SAFE', 'RISK'}
  │ 🚨 COMPATIBILITY ISSUE!
  │
  ▼
models/
  ├── xgboost_risk_model.json
  ├── feature_scaler.pkl
  ├── feature_names.json
  └── risk_lexicons.json
  ↓
[Stage 5: Flask API]
  │ app.py
  │ Loads: data/neodataset_augmented.csv
  │ Loads: models/xgboost_risk_model.json
  │
  ▼
REST API Endpoints
  ├── POST /api/assess-risk
  ├── GET /api/health-check
  ├── POST /api/simulate-scope
  ├── GET /api/stories
  └── GET /api/info
```

---

## Input/Output Verification

### Stage 1 → Stage 2
✅ **Compatible**
- Stage 1 outputs: DataFrame with `full_text`, `word_count`, `has_list`, etc.
- Stage 2 expects: DataFrame with text columns
- Stage 2 uses: `full_text` for labeling functions

### Stage 2 → Stage 3
✅ **Compatible**
- Stage 2 outputs: `risk_label_binary` (0/1), `risk_label` ('SAFE'/'RISK'), `risk_confidence`
- Stage 3 expects: `risk_label_binary`, `full_text`
- Stage 3 preserves all columns

### Stage 3 → Stage 4
❌ **INCOMPATIBLE**
- Stage 3 outputs: `risk_label` in `{'SAFE', 'RISK'}`
- Stage 4 expects: `risk_label` in `{'Low', 'Medium', 'High'}`
- **Mapping will fail**:
  ```python
  label_map = {'Low': 0, 'Medium': 1, 'High': 2}
  df['risk_class'] = df['risk_label'].map(label_map)  # Returns NaN!
  ```

### Stage 4 → Stage 5
✅ **Compatible** (IF Stage 4 runs)
- Stage 4 outputs: `models/xgboost_risk_model.json` (3-class model)
- Stage 5 expects: `models/xgboost_risk_model.json`
- Risk predictor expects 3 classes: ['Low', 'Medium', 'High']

### Stage 3 → Stage 5 (Data Path)
⚠️ **PARTIAL COMPATIBILITY**
- Stage 3 outputs: `data/neodataset_augmented.csv` with `risk_label` = 'SAFE'/'RISK'
- Stage 5 `CSVDataLoader` reads `risk_label` and stores it as `Story.risk_level`
- `Story.risk_level` is just stored/displayed, not used for prediction
- **Works for data loading, but semantics are wrong** (says "RISK" not "High")

---

## Fixing the Compatibility Issue

### Option A: Modify Augmentation (Best for 3-Class Problem)

**Change**: Modify labeling functions to output 3 classes

**Files to modify**:
1. `src/ml/labeling_functions.py` - Change LF outputs to -1/0/1/2
2. `src/ml/weak_supervision_pipeline.py` - Update Snorkel cardinality to 3
3. `scripts/augment_neodataset.py` - Update label mapping

**Pros**:
- More granular risk levels
- Better alignment with problem domain

**Cons**:
- Requires redesigning labeling functions
- More complex weak supervision (3-way classification is harder)

### Option B: Simplify Training to Binary (Recommended)

**Change**: Modify training script to use binary labels

**Files to modify**:
1. `src/ml/train_risk_model.py`
   ```python
   # OLD (lines 74-75)
   label_map = {'Low': 0, 'Medium': 1, 'High': 2}
   df['risk_class'] = df['risk_label'].map(label_map)
   
   # NEW
   label_map = {'SAFE': 0, 'RISK': 1}
   df['risk_class'] = df['risk_label'].map(label_map)
   
   # Update XGBoost params
   params = {
       'objective': 'binary:logistic',  # Changed from multi:softprob
       'num_class': 2,  # Changed from 3
       ...
   }
   ```

2. `src/ml/risk_predictor.py`
   ```python
   # OLD (line 30)
   RISK_LABELS = ['Low', 'Medium', 'High']
   
   # NEW
   RISK_LABELS = ['SAFE', 'RISK']
   ```

3. Update classification reports and confusion matrices

**Pros**:
- Matches augmentation output
- Simpler problem (binary classification)
- No changes to labeling functions

**Cons**:
- Less granular (only SAFE vs RISK, not Low/Medium/High)

### Option C: Post-Augmentation Mapping (Quick Fix)

**Change**: Add mapping step after augmentation

**New script**: `scripts/map_to_3class.py`
```python
# Map binary labels to 3-class
# SAFE → Low
# RISK (high confidence) → High
# RISK (low confidence) → Medium

df = pd.read_csv('data/neodataset_augmented.csv')

def map_label(row):
    if row['risk_label'] == 'SAFE':
        return 'Low'
    elif row['risk_confidence'] > 0.85:
        return 'High'
    else:
        return 'Medium'

df['risk_label'] = df.apply(map_label, axis=1)
df.to_csv('data/neodataset_augmented_3class.csv', index=False)
```

**Pros**:
- Minimal code changes
- Preserves existing work

**Cons**:
- Arbitrary mapping (not theoretically grounded)
- "Medium" class is artificially created

---

## Sequential Execution Plan

### Fresh Start (No Prior Work)

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-augmentation.txt

# 2. Download and augment NeoDataset
python scripts/augment_neodataset.py
# Output: data/neodataset_augmented.csv
#         data/neodataset_augmented_high_confidence.csv

# 3. ⚠️ FIX COMPATIBILITY ISSUE (choose one):
# Option B (Recommended): Edit src/ml/train_risk_model.py and risk_predictor.py
# OR
# Option C: Run mapping script (create it first)
# python scripts/map_to_3class.py

# 4. Install ML dependencies
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm

# 5. Train ML model
python src/ml/train_risk_model.py
# Output: models/xgboost_risk_model.json
#         models/feature_scaler.pkl
#         models/feature_names.json
#         models/risk_lexicons.json

# 6. Test model (optional)
python -c "
from src.ml.risk_predictor import RiskPredictor
predictor = RiskPredictor.load('models', quantize_bert=True)
result = predictor.predict('Implement complex authentication system')
print(f'Risk: {result[\"risk_level\"]} ({result[\"confidence\"]:.1f}%)')
"

# 7. Start Flask API (new terminal, same venv)
# Only needs: pip install -r requirements.txt
python app.py
# Server: http://0.0.0.0:5001

# 8. Test API (new terminal)
curl -X POST http://localhost:5001/api/assess-risk \
  -H "Content-Type: application/json" \
  -d '{"description": "Implement OAuth2 authentication"}'
```

### If You Have Augmented Data

```bash
# Skip stage 1-2, start from stage 4
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm

# Fix compatibility issue first!
# Then train
python src/ml/train_risk_model.py
```

### If You Have Trained Model

```bash
# Skip to stage 5
pip install -r requirements.txt
python app.py
```

---

## Troubleshooting

### Issue: "Augmented NeoDataset not found"
**Solution**: Run `python scripts/augment_neodataset.py` first

### Issue: "ML model not found at models"
**Solution**: Run `python src/ml/train_risk_model.py` first

### Issue: "NaN values in risk_class column"
**Solution**: This is the compatibility issue! See "Fixing the Compatibility Issue" section

### Issue: Training script crashes with label mapping error
**Solution**: You hit the compatibility issue. Modify training script to use binary labels (Option B)

### Issue: Out of memory during DistilBERT embedding
**Solution**: Reduce batch size or use quantization (already enabled by default)

### Issue: Augmentation takes too long
**Solution**: Normal! ~10-20 minutes for 20K stories. Snorkel training is the slowest part.

---

## File Dependencies

### Required Before Training
```
data/
└── neodataset_augmented_high_confidence.csv  # From Stage 3
    Columns: id, title, description, story_points, full_text,
             risk_label, risk_confidence, ...
```

### Required Before Flask API
```
data/
└── neodataset_augmented.csv  # From Stage 3

models/
├── xgboost_risk_model.json  # From Stage 4
├── feature_scaler.pkl       # From Stage 4
├── feature_names.json       # From Stage 4
└── risk_lexicons.json       # From Stage 4
```

### Generated During Augmentation
```
data/
└── neodataset/
    ├── downloads/  # HuggingFace cache (auto-generated)
    └── neodataset_snorkel_labels.csv  # Intermediate file
```

---

## Estimated Time Requirements

| Stage | Time | Bottleneck |
|-------|------|------------|
| Stage 1: Download | 2-5 min | Network speed |
| Stage 2: Weak Supervision | 5-10 min | Snorkel training |
| Stage 3: Noise Filtering | 3-5 min | Cross-validation |
| **Total Augmentation** | **10-20 min** | - |
| Stage 4: ML Training | 10-30 min | DistilBERT embeddings |
| **Total First-Time Setup** | **20-50 min** | - |
| Stage 5: Start API | <10 sec | - |
| Stage 5: First Request | 1-2 sec | BERT inference |
| Stage 5: Subsequent Requests | <100ms | Cached |

---

## Key Takeaways

1. **Sequential dependency**: Each stage requires the previous stage's output
2. **Critical blocker**: Binary vs 3-class label mismatch between Stages 3 and 4
3. **One-time cost**: Augmentation and training are done once, then cached
4. **Production use**: Flask API is lightweight (only needs augmented CSV and models/)
5. **Data sizes**:
   - Raw NeoDataset: ~25,000 stories
   - After preprocessing: ~20,000 stories
   - After Cleanlab: ~18,000 stories
   - High-confidence subset: ~12,000 stories

---

## Next Steps

1. **URGENT**: Fix label compatibility issue (recommend Option B)
2. Run augmentation pipeline: `python scripts/augment_neodataset.py`
3. Verify outputs: Check `data/neodataset_augmented.csv` has correct columns
4. Train model: `python src/ml/train_risk_model.py`
5. Test API: `python app.py`
6. (Optional) Create validation set for manual label quality check
7. (Optional) Fine-tune hyperparameters based on test set performance

