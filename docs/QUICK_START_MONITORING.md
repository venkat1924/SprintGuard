# Quick Start: ML Monitoring & Publication Plots

## Installation

```bash
pip install -r requirements-ml.txt
```

## Run Complete Pipeline with Monitoring

### Step 1: Augmentation with Tracking
```bash
python scripts/augment_neodataset.py
```

**Output:**
- ✅ `visualizations/sankey_pipeline_flow.pdf` - Data flow diagram
- ✅ `visualizations/lf_correlation_heatmap.pdf` - LF analysis
- ✅ MLflow logs in `mlruns/`

### Step 2: Model Training with Tracking
```bash
python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv
```

**Output:**
- ✅ `visualizations/calibration_plot.pdf` - Model confidence
- ✅ `visualizations/embeddings_tsne.pdf` - Feature space
- ✅ Training metrics in MLflow

### Step 3: Ablation Study
```bash
python scripts/run_ablation_study.py
```

**Output:**
- ✅ `visualizations/ablation_study.pdf` - Component comparison
- ✅ Statistical results (5 runs each)

### Step 4: View Results
```bash
mlflow ui
```

Open: http://localhost:5000

## What Was Implemented

### ✅ Core Infrastructure
- [x] MLflow experiment tracker (`src/ml/experiment_tracker.py`)
- [x] Publication plot generators (`src/visualization/publication_plots.py`)
- [x] IEEE-compliant styling with SciencePlots

### ✅ Pipeline Integration
- [x] Stage 1: Data loading metrics
- [x] Stage 2: Snorkel diagnostics + LF heatmap generation
- [x] Stage 3: Cleanlab health scores
- [x] Stage 4: Training metrics + calibration + t-SNE

### ✅ Visualizations (5 total)
1. [x] Sankey diagram - Pipeline data flow
2. [x] Ablation study - Baseline comparisons  
3. [x] LF correlation heatmap - Labeling function analysis
4. [x] Calibration plot - Model confidence reliability
5. [x] t-SNE embeddings - Feature space visualization

### ✅ Scripts
- [x] `scripts/augment_neodataset.py` - Updated with tracking
- [x] `scripts/run_ablation_study.py` - NEW: Run baseline comparisons
- [x] `scripts/generate_all_plots.py` - NEW: Regenerate visualizations

### ✅ Documentation
- [x] `MONITORING_IMPLEMENTATION.md` - Full implementation guide
- [x] `QUICK_START_MONITORING.md` - This file

## Files Generated

```
visualizations/
├── sankey_pipeline_flow.pdf        # Pipeline flow
├── sankey_pipeline_flow.svg
├── sankey_pipeline_flow.html       # Interactive version
├── lf_correlation_heatmap.pdf      # LF analysis
├── lf_correlation_heatmap.svg
├── ablation_study.pdf              # Baseline comparison
├── ablation_study.svg
├── calibration_plot.pdf            # Model confidence
├── calibration_plot.svg
├── embeddings_tsne.pdf             # Feature space
└── embeddings_tsne.svg

mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── metrics/                # All logged metrics
        ├── params/                 # All parameters
        └── artifacts/              # Models, plots, data
```

## Next Steps

### For Paper Writing

1. **Include figures in LaTeX:**
   ```latex
   \begin{figure}[t]
     \centering
     \includegraphics[width=\columnwidth]{visualizations/sankey_pipeline_flow.pdf}
     \caption{SprintGuard pipeline data flow.}
     \label{fig:pipeline}
   \end{figure}
   ```

2. **Export metric tables from MLflow UI:**
   - Navigate to experiment
   - Select runs to compare
   - Download as CSV
   - Convert to LaTeX table

3. **Report statistical results:**
   - Use ablation study mean ± std values
   - Include significance tests if needed

### For Reproducibility

1. **Archive experiment data:**
   ```bash
   tar -czf sprintguard_experiments.tar.gz mlruns/ visualizations/
   ```

2. **Document environment:**
   ```bash
   pip freeze > requirements_exact.txt
   ```

3. **Share with collaborators:**
   - Upload MLflow data to shared storage
   - Or use MLflow tracking server

### For Continued Development

1. **Create new experiments:**
   ```python
   tracker = ExperimentTracker(experiment_name="SprintGuard_V2")
   ```

2. **Compare with baselines:**
   - Use MLflow UI comparison view
   - Run `scripts/run_ablation_study.py` with new configs

3. **Monitor training:**
   - Watch MLflow UI during long training runs
   - Check convergence plots in real-time

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: mlflow` | `pip install mlflow>=2.9.0` |
| Kaleido error (Sankey plot) | `pip install kaleido>=0.2.1` |
| SciencePlots not found | `pip install scienceplots>=2.1.0` (optional) |
| MLflow UI port conflict | `mlflow ui --port 5001` |
| Plots not saving | `mkdir -p visualizations` |

## Key Metrics Tracked

### Augmentation Pipeline
- `data_flow/raw_count` - Initial dataset size
- `data_flow/snorkel_count` - After labeling
- `data_flow/cleanlab_count` - After filtering
- `data_flow/final_count` - Final training set
- `snorkel/lf_coverage/*` - Per-LF coverage
- `cleanlab/overall_health_score` - Label quality

### Model Training
- `evaluation/macro_f1` - Primary metric
- `evaluation/accuracy` - Overall accuracy
- `evaluation/f1_*` - Per-class F1 scores
- `evaluation/precision_*` - Per-class precision
- `evaluation/recall_*` - Per-class recall

### Ablation Study
- `ablation/Baseline_mean_f1` - No weak supervision
- `ablation/Snorkel_Only_mean_f1` - No filtering
- `ablation/SprintGuard_Full_mean_f1` - Complete pipeline

## MLflow UI Tips

1. **Compare experiments:** Select multiple runs → "Compare" button
2. **Download plots:** Click any plot → "Download" icon
3. **Export metrics:** Select runs → "Download CSV"
4. **Search runs:** Use query syntax: `metrics.macro_f1 > 0.75`
5. **Tag important runs:** Click run → "Set Tag" → Add "best_model"

## Publication Checklist

- [ ] Run full pipeline with monitoring
- [ ] Generate all 5 visualizations
- [ ] Run ablation study (5 seeds minimum)
- [ ] Export MLflow metrics to tables
- [ ] Convert plots to required format
- [ ] Archive experiment data
- [ ] Document hyperparameters
- [ ] Include reproducibility section in paper
- [ ] Test reproduction on clean environment

## Commands Summary

```bash
# Full pipeline
python scripts/augment_neodataset.py
python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv
python scripts/run_ablation_study.py

# View results
mlflow ui

# Regenerate plots
python scripts/generate_all_plots.py

# Archive for paper
tar -czf experiments.tar.gz mlruns/ visualizations/ *.csv
```

---

**Ready to go!** 🚀

All monitoring infrastructure is implemented. Run the commands above to generate publication-ready results.

