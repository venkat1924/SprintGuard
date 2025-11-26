# 🛡️ SprintGuard

**Predictive Sprint Planning & Risk Mitigation Platform**

SprintGuard uses machine learning to predict risk levels of user stories, helping Agile teams avoid estimation failure and scope creep.

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd /home/jovyan/SprintGuard
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies incrementally:**

```bash
# Core web application (required)
pip install -r requirements.txt

# Data augmentation pipeline (required for first-time setup)
pip install -r requirements-augmentation.txt

# ML model training and inference (required for risk prediction)
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm

# Development tools (optional)
pip install -r requirements-dev.txt
```

### First-Time Setup: Data Augmentation

Before running the application, you need to augment the NeoDataset (~20K user stories from HuggingFace) with risk labels:

```bash
# This downloads NeoDataset and applies weak supervision pipeline
# Takes ~15-30 minutes
python scripts/augment_neodataset.py
```

This creates:
- `data/neodataset_augmented.csv` - Full augmented dataset
- `data/neodataset_augmented_high_confidence.csv` - High-confidence subset

### Train the ML Model

After augmentation, train the DistilBERT-XGBoost risk model:

```bash
./scripts/train_ml_model.sh
```

This script will:
- Check for and create augmented dataset if needed
- Download spaCy model if missing
- Train the model with proper PYTHONPATH
- Run validation tests

Model artifacts are saved to the `models/` directory.

### Start the Application

```bash
python app.py
```

Open your browser: `http://localhost:5001`

## 📊 Features

### 1. Data Health Check
Assesses the quality and quantity of your historical data to set realistic expectations about prediction accuracy.

### 2. Probabilistic Story Assessor (PSA)
Analyzes new user stories and assigns risk levels (Low/Medium/High) based on ML models trained on real-world data.

### 3. Scope Impact Simulator (SIS)
Models the timeline impact of adding new work to a sprint, making scope creep costs tangible.

## 🏗️ Architecture

- **Backend**: Python 3.9+ with Flask 3.0
- **Data Source**: Augmented NeoDataset (~20K real user stories)
- **ML Pipeline**: Snorkel (weak supervision) + Cleanlab (noise filtering)
- **Risk Model**: DistilBERT-XGBoost with SHAP explainability

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[SETUP.md](docs/SETUP.md)** - Detailed installation and configuration guide
- **[AUGMENTATION_STATUS.md](docs/AUGMENTATION_STATUS.md)** - NeoDataset augmentation pipeline details
- **[ML_MODEL_GUIDE.md](docs/ML_MODEL_GUIDE.md)** - ML model training and usage
- **[ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md)** - Technical architecture of ML components
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Full implementation overview
- **[research/](docs/research/)** - Research notes on ML techniques

## 📡 API Endpoints

- `GET /api/health-check` - Data quality assessment
- `POST /api/assess-risk` - Story risk prediction
- `POST /api/simulate-scope` - Timeline impact simulation
- `GET /api/stories` - Historical stories retrieval
- `GET /api/info` - System information

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest
pytest --cov=src --cov-report=html  # With coverage
```

## 📁 Project Structure

```
SprintGuard/
├── app.py                          # Flask application
├── config.py                       # Configuration
├── requirements*.txt               # Dependencies (core, augmentation, ML, dev)
├── src/
│   ├── analyzers/                  # Risk assessment, health check, scope simulation
│   ├── ml/                         # ML pipeline (augmentation, training, inference)
│   ├── models/                     # Data models (Story)
│   └── utils/                      # Utilities
├── scripts/
│   ├── augment_neodataset.py       # Main augmentation script
│   ├── train_ml_model.sh           # Model training script
│   └── explore_neodataset.py       # Data exploration tool
├── tests/                          # Unit tests
├── docs/                           # Documentation
└── data/                           # Data files (generated)
```

## 🔮 Future Enhancements

1. **Dynamic Resource Forecaster** - Skill-based bottleneck detection
2. **Jira Cloud Integration** - Real-time API connection
3. **Team Calibration Tool** - Improve estimation consistency
4. **Advanced ML Models** - Deep learning for pattern recognition
5. **Custom Dashboards** - Exportable reports for stakeholders

## 🐛 Troubleshooting

### Augmented dataset not found
```bash
python scripts/augment_neodataset.py
```

### Port 5001 already in use
Edit `config.py` and change `PORT = 5001` to another value.

### ModuleNotFoundError
Use the training script which handles PYTHONPATH automatically:
```bash
./scripts/train_ml_model.sh
```

Or if running Python directly, set PYTHONPATH first:
```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python src/ml/train_risk_model.py
```

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

## 📄 License

This Proof of Concept is provided as-is for educational purposes.

---

**Built with ❤️ to help Agile teams break the cycle of estimation failure and scope creep.**
