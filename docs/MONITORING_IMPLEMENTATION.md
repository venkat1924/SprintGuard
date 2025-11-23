# SprintGuard ML Monitoring & Visualization Implementation

## Overview

This document describes the comprehensive monitoring and visualization infrastructure implemented for the SprintGuard research project. The implementation enables rigorous experiment tracking and generation of publication-quality figures for academic journals.

## Architecture

### Core Components

1. **Experiment Tracker** (`src/ml/experiment_tracker.py`)
   - MLflow-based tracking system
   - Hierarchical metric logging (stage/metric_name)
   - Automatic visualization export (PDF + SVG)
   - Data flow monitoring for Sankey diagrams

2. **Publication Plots** (`src/visualization/publication_plots.py`)
   - IEEE-compliant visualization styles
   - Vector format exports (PDF/SVG)
   - Colorblind-friendly palettes
   - 5 core visualizations:
     - Sankey diagram (pipeline data flow)
     - Ablation study (baseline comparisons)
     - LF correlation heatmap (labeling function analysis)
     - Calibration plot (confidence reliability)
     - t-SNE embeddings (feature space visualization)

3. **Pipeline Integration**
   - Stage 1 (Data Loading): Basic metrics logging
   - Stage 2 (Snorkel): LF diagnostics + correlation heatmap
   - Stage 3 (Cleanlab): Health scores + retention statistics
   - Stage 4 (Training): XGBoost metrics + calibration + t-SNE

## Installation

### Dependencies

Add to your Python environment:

```bash
pip install mlflow>=2.9.0 \
            scienceplots>=2.1.0 \
            scikit-plot>=0.3.7 \
            plotly>=5.18.0 \
            kaleido>=0.2.1
```

Or install from requirements:

```bash
pip install -r requirements-ml.txt
```

## Usage

### 1. Run Augmentation Pipeline with Tracking

```bash
python scripts/augment_neodataset.py
```

**Generates:**
- MLflow experiment logs in `mlruns/`
- Sankey diagram: `visualizations/sankey_pipeline_flow.pdf`
- LF correlation heatmap: `visualizations/lf_correlation_heatmap.pdf`

**Tracked Metrics:**
- Data counts at each stage (raw, snorkel, cleanlab, final)
- LF coverage and conflicts
- Label health scores
- Stage execution times

### 2. Train Model with Tracking

```bash
python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv
```

**Generates:**
- Calibration plot: `visualizations/calibration_plot.pdf`
- t-SNE embeddings: `visualizations/embeddings_tsne.pdf`

**Tracked Metrics:**
- Training/validation loss per epoch
- Per-class precision, recall, F1
- Overall accuracy and macro-F1
- Feature importance scores

### 3. Run Ablation Study

```bash
python scripts/run_ablation_study.py
```

**Compares:**
1. Baseline: TF-IDF + Logistic Regression
2. Snorkel Only: No Cleanlab filtering
3. SprintGuard Full: Complete pipeline

**Generates:**
- Ablation study plot: `visualizations/ablation_study.pdf`
- Statistical comparison (mean ± std over 5 runs)

### 4. View Results in MLflow UI

```bash
mlflow ui
```

Then open: http://localhost:5000

**Features:**
- Compare experiments side-by-side
- View all logged metrics and plots
- Download artifacts
- Export results for papers

### 5. Regenerate Plots

```bash
python scripts/generate_all_plots.py
```

Regenerates all visualizations from saved data.

## File Structure

```
SprintGuard/
├── src/
│   ├── ml/
│   │   ├── experiment_tracker.py      # MLflow wrapper
│   │   ├── train_risk_model.py        # Updated with tracking
│   │   ├── weak_supervision_pipeline.py  # Updated with logging
│   │   └── cleanlab_pipeline.py       # Updated with logging
│   └── visualization/
│       ├── __init__.py
│       └── publication_plots.py       # All plot generators
├── scripts/
│   ├── augment_neodataset.py          # Updated with tracking
│   ├── run_ablation_study.py          # NEW: Ablation experiments
│   └── generate_all_plots.py          # NEW: Plot regeneration
├── visualizations/                    # Generated plots (PDF + SVG)
│   ├── sankey_pipeline_flow.pdf
│   ├── lf_correlation_heatmap.pdf
│   ├── ablation_study.pdf
│   ├── calibration_plot.pdf
│   └── embeddings_tsne.pdf
└── mlruns/                            # MLflow tracking data
```

## Publication-Ready Outputs

All visualizations are generated in IEEE-compliant format:

### 1. Sankey Diagram
- **Purpose:** Show data flow through pipeline stages
- **Format:** PDF (for paper) + HTML (interactive)
- **Shows:** Raw → Snorkel → Cleanlab → Final Training Set

### 2. Ablation Study
- **Purpose:** Demonstrate impact of each component
- **Format:** Bar chart with error bars
- **Shows:** Macro-F1 scores for 3 configurations

### 3. LF Correlation Heatmap
- **Purpose:** Analyze labeling function relationships
- **Format:** Grayscale heatmap (18×18 matrix)
- **Shows:** Pearson correlation between all LF pairs

### 4. Calibration Plot
- **Purpose:** Assess model confidence reliability
- **Format:** Line plot with perfect calibration reference
- **Shows:** Predicted probability vs. actual frequency

### 5. t-SNE Embeddings
- **Purpose:** Visualize learned feature representations
- **Format:** Scatter plot with class colors
- **Shows:** 2D projection of 768-dim DistilBERT embeddings

## Experiment Tracking Details

### Logged Parameters

**Augmentation Pipeline:**
- Number of labeling functions
- Number of stories at each stage
- Confidence thresholds
- Stage execution times

**Model Training:**
- XGBoost hyperparameters
- Train/val/test split sizes
- Feature dimensions
- Early stopping rounds

### Logged Metrics

**Snorkel (Stage 2):**
- `snorkel/lf_coverage/<lf_name>`: Coverage per LF
- `snorkel/overall_coverage`: % stories with labels
- `snorkel/num_conflicts`: Stories with disagreements
- `snorkel/confidence_mean`: Average confidence
- `snorkel/label_count/<label>`: Per-label counts

**Cleanlab (Stage 3):**
- `cleanlab/overall_health_score`: Label quality (0-1)
- `cleanlab/num_issues_detected`: Noisy labels found
- `cleanlab/pct_data_pruned`: % data removed
- `cleanlab/retention_rate`: % data kept

**Training (Stage 4):**
- `training/num_boost_round`: XGBoost rounds
- `evaluation/accuracy`: Overall accuracy
- `evaluation/macro_f1`: Macro-averaged F1
- `evaluation/f1_low`, `f1_medium`, `f1_high`: Per-class F1
- `evaluation/precision_*`, `recall_*`: Per-class metrics

### Logged Artifacts

- All visualization PDFs and SVGs
- Model files (XGBoost, scaler, feature names)
- Stage count JSON (for Sankey diagram)
- Intermediate datasets

## Integration with Existing Code

### Minimal Changes Required

The tracking system was integrated with minimal disruption:

1. **Added tracker parameter** to `RiskModelTrainer.__init__()`
2. **Added logging methods** to existing pipeline classes:
   - `WeakSupervisionPipeline.log_lf_diagnostics()`
   - `CleanlabPipeline.log_cleanlab_diagnostics()`
3. **Wrapped main scripts** with tracker initialization/cleanup

### Backward Compatibility

All tracking is **optional**:
- Scripts work without tracker (tracker=None)
- No breaking changes to existing APIs
- Tracking can be disabled by not initializing ExperimentTracker

## Research Paper Integration

### Recommended Figures for Paper

1. **Pipeline Architecture** (Section: Methodology)
   - Use: Sankey diagram
   - Shows: Data transformation at each stage

2. **Weak Supervision Analysis** (Section: Labeling Functions)
   - Use: LF correlation heatmap
   - Demonstrates: LF diversity and complementarity

3. **Ablation Study** (Section: Evaluation)
   - Use: Ablation study bar chart
   - Proves: Each component contributes to performance

4. **Model Quality** (Section: Results)
   - Use: Calibration plot + Confusion matrix
   - Shows: Model reliability and per-class performance

5. **Feature Learning** (Section: Discussion / Appendix)
   - Use: t-SNE embeddings
   - Illustrates: Learned representations separate classes

### Table Data

Export from MLflow UI:
- Hyperparameter table (all XGBoost params)
- Metric comparison table (train vs val vs test)
- Ablation results table (with statistical significance tests)

## Reproducibility

All experiments are fully reproducible:

1. **Fixed random seeds** in all scripts
2. **Logged parameters** capture all configuration
3. **Versioned data** via stage count tracking
4. **Exported artifacts** enable result verification

To reproduce published results:

```bash
# 1. Run full pipeline
python scripts/augment_neodataset.py

# 2. Train model
python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv

# 3. Run ablation study
python scripts/run_ablation_study.py

# 4. View results
mlflow ui
```

## Best Practices

### For Paper Submission

1. **Include MLflow run ID** in paper footnotes
2. **Export metric tables** to LaTeX format
3. **Use vector formats** (PDF/SVG) for all figures
4. **Report confidence intervals** from ablation study
5. **Archive MLflow data** with paper submission

### For Collaboration

1. **Commit `requirements-ml.txt`** with exact versions
2. **Share MLflow tracking URI** with team
3. **Document experiment naming** conventions
4. **Regular backup** of `mlruns/` directory

### For Continued Development

1. **Create new experiment** for each major change
2. **Tag runs** with descriptive labels
3. **Compare metrics** before/after changes
4. **Archive successful runs** for reference

## Troubleshooting

### "Module not found: mlflow"
```bash
pip install -r requirements-ml.txt
```

### "Kaleido not found" (for Sankey diagram)
```bash
pip install kaleido>=0.2.1
```

### "SciencePlots style not found"
```bash
pip install scienceplots>=2.1.0
```

Falls back to default matplotlib style if unavailable.

### MLflow UI not starting
```bash
# Specify port if 5000 is in use
mlflow ui --port 5001
```

### Plots not generating
Check that `visualizations/` directory exists:
```bash
mkdir -p visualizations
```

## Future Enhancements

Potential additions:

1. **DVC Integration** - Version control for datasets
2. **Weights & Biases** - Cloud-based tracking option
3. **Statistical Tests** - Automatic significance testing
4. **Hyperparameter Tuning** - Optuna integration with MLflow
5. **Model Comparison** - Multi-model benchmarking dashboard

## References

- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **SciencePlots:** https://github.com/garrettj403/SciencePlots
- **IEEE Publication Standards:** https://www.ieee.org/publications/authors/author-resources.html

## Support

For issues or questions:
1. Check MLflow UI for logged data
2. Review `mlruns/` directory structure
3. Verify all dependencies installed
4. Check script output for error messages

---

**Implementation Date:** November 2025  
**SprintGuard Version:** 1.0  
**MLflow Experiment:** SprintGuard_Augmentation, SprintGuard_Training

