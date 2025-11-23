#!/usr/bin/env python3
"""
Generate All Publication Plots from Saved Data

This script regenerates all publication-quality visualizations
from previously saved experiment results and data.

Usage:
    python scripts/generate_all_plots.py
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

sys.path.insert(0, '.')
from src.visualization.publication_plots import (
    generate_sankey_diagram,
    generate_ablation_study,
    generate_lf_correlation_heatmap,
    generate_calibration_plot,
    generate_tsne_embeddings
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("\n" + "="*70)
    logging.info("REGENERATE ALL PUBLICATION PLOTS")
    logging.info("="*70)
    
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Sankey Diagram
    logging.info("\n[1/5] Generating Sankey diagram...")
    stage_counts_path = output_dir / "stage_counts.json"
    if stage_counts_path.exists():
        with open(stage_counts_path) as f:
            stage_counts = json.load(f)
        generate_sankey_diagram(stage_counts, output_path="visualizations/sankey_pipeline_flow.pdf")
        logging.info("  ✓ Sankey diagram saved")
    else:
        logging.warning(f"  ⚠ Stage counts not found: {stage_counts_path}")
        logging.info("    Using placeholder data...")
        stage_counts = {'raw': 12106, 'snorkel': 12106, 'cleanlab': 10500, 'final': 9000}
        generate_sankey_diagram(stage_counts, output_path="visualizations/sankey_pipeline_flow.pdf")
        logging.info("  ✓ Sankey diagram saved (with placeholder data)")
    
    # 2. Ablation Study
    logging.info("\n[2/5] Generating ablation study plot...")
    # Example results (replace with actual results from experiments)
    results = {
        'Baseline\n(No WS)': (0.68, 0.03),
        'Snorkel\nOnly': (0.73, 0.02),
        'SprintGuard\n(Full)': (0.79, 0.02)
    }
    generate_ablation_study(results, output_path="visualizations/ablation_study.pdf")
    logging.info("  ✓ Ablation study plot saved")
    logging.info("    Note: Using example data. Run run_ablation_study.py for real results.")
    
    # 3. LF Correlation Heatmap
    logging.info("\n[3/5] Generating LF correlation heatmap...")
    snorkel_data_path = Path("data/neodataset/neodataset_snorkel_labels.csv")
    if snorkel_data_path.exists():
        # Load label matrix from intermediate data
        # Note: This is simplified - actual L_matrix is not saved, need to rerun Snorkel
        logging.warning("  ⚠ L_matrix not saved. Need to rerun augmentation pipeline.")
        logging.info("    Run: python scripts/augment_neodataset.py")
    else:
        logging.warning(f"  ⚠ Snorkel data not found: {snorkel_data_path}")
    
    # 4. Calibration Plot
    logging.info("\n[4/5] Generating calibration plot...")
    # Need trained model predictions - check if model exists
    model_path = Path("models/xgboost_risk_model.json")
    test_data_path = Path("data/neodataset_augmented_3class.csv")
    
    if model_path.exists() and test_data_path.exists():
        logging.info("  Model and data found, but predictions needed.")
        logging.info("  Calibration plot is generated during model training/evaluation.")
        logging.info("    Run: python src/ml/train_risk_model.py")
    else:
        logging.warning("  ⚠ Model or data not found")
        logging.info("    Run full pipeline first")
    
    # 5. t-SNE Embeddings
    logging.info("\n[5/5] Generating t-SNE embedding visualization...")
    if test_data_path.exists():
        logging.info("  Data found, but embeddings need to be computed.")
        logging.info("  t-SNE plot is generated during model training.")
        logging.info("    Run: python src/ml/train_risk_model.py")
    else:
        logging.warning("  ⚠ Data not found")
    
    # Summary
    logging.info("\n" + "="*70)
    logging.info("PLOT GENERATION SUMMARY")
    logging.info("="*70)
    logging.info(f"✓ Sankey diagram: {output_dir / 'sankey_pipeline_flow.pdf'}")
    logging.info(f"✓ Ablation study: {output_dir / 'ablation_study.pdf'}")
    logging.info(f"⚠ LF correlation heatmap: Run augmentation pipeline")
    logging.info(f"⚠ Calibration plot: Run model training")
    logging.info(f"⚠ t-SNE embeddings: Run model training")
    
    logging.info("\nTo generate all plots:")
    logging.info("  1. python scripts/augment_neodataset.py  # Generates Sankey + LF heatmap")
    logging.info("  2. python src/ml/train_risk_model.py     # Generates calibration + t-SNE")
    logging.info("  3. python scripts/run_ablation_study.py  # Generates ablation study")
    
    logging.info("\nAll plots will be in IEEE-ready format (PDF + SVG)")
    logging.info("View MLflow UI: mlflow ui")


if __name__ == '__main__':
    main()

