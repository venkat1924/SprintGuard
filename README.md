# 🛡️ SprintGuard - Proof of Concept

**Predictive Sprint Planning & Risk Mitigation Platform**

SprintGuard is a data-driven platform designed to mitigate estimation failure and scope creep in Agile software development. This Proof of Concept demonstrates the core value proposition: transforming sprint planning from hopeful forecasting into data-driven risk management.

---

## 🎯 Problem Statement

Agile teams face a vicious cycle:
- **Estimation Failure**: Cognitive biases and unclear requirements lead to inaccurate estimates
- **Scope Creep**: Underestimated work makes adding features seem cheap
- **Sprint Chaos**: Overcommitment and mid-sprint additions cause missed deadlines and burnout

SprintGuard breaks this cycle by injecting predictive analytics directly into the planning process.

---

## ✨ Key Features

### 1. 📊 Data Health Check
Assesses the quality and quantity of your historical data to set realistic expectations about prediction accuracy.

**Metrics:**
- **Volume Score**: Number of historical stories vs minimum threshold
- **Completeness Score**: Percentage of stories with all required data
- **Quality Score**: Description length and specificity
- **Consistency Score**: Story point distribution patterns
- **Overall Grade**: A-F rating with actionable recommendations

### 2. 🎯 Probabilistic Story Assessor (PSA)
Analyzes new user stories and assigns risk levels based on machine learning models trained on real-world data.

**Features:**
- Risk classification: Low, Medium, High
- Confidence scoring (0-100%)
- ML-powered predictions using augmented NeoDataset
- Trained on 20,000+ real user stories
- **Pluggable Architecture**: Swap ML models without changing other code

### 3. ⚡ Scope Impact Simulator (SIS)
Models the timeline impact of adding new work to a sprint, making scope creep costs tangible.

**Outputs:**
- Revised sprint end date
- Business days delay
- Impact summary
- Actionable recommendations

---

## 🏗️ Architecture

### Technology Stack
- **Backend**: Python 3.9+ with Flask 3.0
- **Data Source**: Augmented NeoDataset (~20K real user stories)
- **ML Pipeline**: Snorkel (weak supervision) + Cleanlab (noise filtering)
- **Frontend**: Modern HTML5, CSS3, Vanilla JavaScript
- **Testing**: pytest with comprehensive test coverage

### Project Structure
```
SprintGuard/
├── data/
│   ├── neodataset_augmented.csv            # Augmented dataset (20K+ stories)
│   ├── neodataset_augmented_high_confidence.csv  # High-conf subset
│   └── sprintguard.db                      # Optional SQLite (future use)
├── src/
│   ├── models/
│   │   └── story.py                        # Story data model
│   ├── analyzers/
│   │   ├── risk_assessor_interface.py      # Abstract PSA interface
│   │   ├── ml_risk_assessor.py             # ML implementation
│   │   ├── health_checker.py               # Data quality analyzer
│   │   └── scope_simulator.py              # Timeline simulator
│   ├── ml/                                 # ML pipeline
│   │   ├── neodataset_loader.py            # Dataset loading
│   │   ├── labeling_functions.py           # 18 research-backed LFs
│   │   ├── weak_supervision_pipeline.py    # Snorkel aggregation
│   │   └── cleanlab_pipeline.py            # Noise detection
│   ├── data_loader.py                      # Data abstraction layer
│   ├── database.py                         # Optional DB utils
│   └── utils/
│       └── response_formatter.py           # API response formatting
├── scripts/
│   ├── explore_neodataset.py               # EDA tool
│   └── augment_neodataset.py               # Full augmentation pipeline
├── static/
│   ├── css/style.css                       # Application styles
│   └── js/app.js                           # Frontend logic
├── templates/
│   └── index.html                          # Main UI
├── tests/                                  # Unit tests
├── app.py                                  # Flask application
├── config.py                               # Configuration
├── requirements.txt                        # Core dependencies
└── requirements-ml.txt                     # ML dependencies
```

---

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

3. **Install core dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install ML dependencies:**
```bash
pip install -r requirements-ml.txt
```

5. **Run NeoDataset augmentation** (one-time setup, ~15-30 minutes):
```bash
python scripts/augment_neodataset.py
```
This will:
- Download NeoDataset (~20K user stories) from HuggingFace
- Apply 18 research-backed labeling functions
- Train Snorkel model to aggregate labels
- Use Cleanlab to filter noisy labels
- Generate `data/neodataset_augmented.csv`

6. **Start the application:**
```bash
python app.py
```

7. **Open your browser:**
```
http://localhost:5001
```

---

## 📊 Sample Data

The PoC includes 140 realistic synthetic stories with the following distribution:
- **High Risk** (20%): API integration, database migration, third-party services
- **Medium Risk** (50%): Standard features with moderate complexity
- **Low Risk** (30%): Simple UI updates, copy changes

The data includes realistic patterns:
- ~17% stories caused sprint spillover
- ~13% were significantly underestimated (actual > 1.5× estimate)

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_health_checker.py
```

---

## 🔌 Pluggable Risk Assessor Architecture

### For ML Professors / Data Scientists

The Risk Assessor uses a **pluggable architecture** that allows you to swap the assessment algorithm without modifying any other code.

#### Current Implementation
`KeywordRiskAssessor` (V1) - Simple keyword-based heuristics

#### To Add Your ML Model:

1. **Create your implementation** in `src/analyzers/your_assessor.py`:

```python
from src.analyzers.risk_assessor_interface import RiskAssessorInterface, RiskResult

class YourMLAssessor(RiskAssessorInterface):
    def __init__(self, historical_stories):
        super().__init__(historical_stories)
        # Train your model here
        self.model = self.train_model()
    
    def train_model(self):
        # Your ML training logic
        pass
    
    def assess(self, description: str) -> RiskResult:
        # Your prediction logic
        prediction = self.model.predict(description)
        
        return RiskResult(
            risk_level="High",  # or "Medium", "Low"
            confidence=85.0,
            explanation="Your model's explanation",
            similar_stories=[1, 5, 12]  # Optional
        )
```

2. **Swap the implementation** in `app.py` (line ~24):

```python
# Replace this line:
risk_assessor = KeywordRiskAssessor(historical_stories)

# With:
from src.analyzers.your_assessor import YourMLAssessor
risk_assessor = YourMLAssessor(historical_stories)
```

3. **That's it!** The entire application will now use your algorithm.

#### Interface Contract
Your implementation must:
- Inherit from `RiskAssessorInterface`
- Implement `assess(description: str) -> RiskResult`
- Return risk_level as "Low", "Medium", or "High"
- Provide a confidence score (0-100)
- Include a human-readable explanation

---

## 🎨 UI Features

### Auto-Loading Health Check
The Data Health Check runs automatically on page load, immediately showing data quality.

### Example Stories
The PSA includes a "Try Example" button with pre-filled realistic stories for quick demos.

### Responsive Design
The UI adapts to mobile, tablet, and desktop screens.

### Real-time Feedback
All features provide immediate visual feedback with color-coded results.

---

## 📡 API Endpoints

### GET `/api/health-check`
Returns data quality assessment.

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_grade": "B",
    "overall_score": 82.5,
    "volume_score": 100.0,
    "completeness_score": 95.0,
    "quality_score": 78.0,
    "consistency_score": 75.0,
    "story_count": 140,
    "recommendations": [...]
  }
}
```

### POST `/api/assess-risk`
Assesses risk of a user story.

**Request:**
```json
{
  "description": "Implement OAuth2 authentication for API"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "risk_level": "High",
    "confidence": 85.0,
    "explanation": "Contains high-complexity keywords: oauth2, authentication, api...",
    "similar_stories": [1, 5, 12]
  }
}
```

### POST `/api/simulate-scope`
Simulates timeline impact of scope addition.

**Request:**
```json
{
  "current_end_date": "2025-12-15",
  "current_story_points": 20,
  "new_story_points": 5,
  "team_velocity": 2.0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "original_end_date": "2025-12-15",
    "new_end_date": "2025-12-18",
    "days_added": 3,
    "original_points": 20,
    "new_points": 25,
    "impact_summary": "Adding 5 story points...",
    "recommendations": [...]
  }
}
```

### GET `/api/stories?risk_level=High&limit=10`
Retrieves historical stories (for exploration/debugging).

### GET `/api/info`
Returns system information including current risk assessor name.

---

## 🔮 Future Enhancements

### Post-PoC Roadmap:
1. **Dynamic Resource Forecaster** - Skill-based bottleneck detection
2. **Jira Cloud Integration** - Real-time API connection
3. **Team Calibration Tool** - Improve estimation consistency
4. **Advanced ML Models** - Deep learning for pattern recognition
5. **Custom Dashboards** - Exportable reports for stakeholders
6. **Multi-tenant SaaS** - Support multiple teams with isolated data

---

## 📝 Configuration

Edit `config.py` to customize:
- Database path
- Health check thresholds
- Risk keyword lists
- Flask server settings

---

## 🐛 Troubleshooting

### Database not found
```bash
python3 data/seed_data_generator.py
python3 src/database.py
```

### Port 5001 already in use
Edit `config.py` and change `PORT = 5001` to another value.

### ModuleNotFoundError
Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📚 References

This PoC is based on the comprehensive SprintGuard proposal document that analyzes:
- The anatomy of estimation failure in Agile teams
- Scope creep patterns and their root causes
- The domino effect of flawed sprint planning
- Cultural amplification factors in India's tech ecosystem

---

## 🤝 Contributing

This is a Proof of Concept for educational and demonstration purposes. 

To contribute:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `pytest`
5. Submit a pull request

---

## 📄 License

This Proof of Concept is provided as-is for educational purposes.

---

## 👥 Contact

For questions about integrating ML models or extending functionality, please refer to the pluggable architecture documentation above.

---

**Built with ❤️ to help Agile teams break the cycle of estimation failure and scope creep.**

