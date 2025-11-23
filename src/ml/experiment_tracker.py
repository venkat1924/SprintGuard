"""
MLflow Experiment Tracker for SprintGuard Pipeline
Centralizes logging across all pipeline stages for reproducibility
"""
import os
import json
import logging
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ExperimentTracker:
    """
    Manages MLflow experiment tracking across all pipeline stages.
    
    Features:
    - Hierarchical metric logging (stage/metric_name)
    - Automatic visualization saving (PDF + SVG)
    - Parameter and artifact tracking
    - Stage timing and data flow monitoring
    """
    
    def __init__(self, experiment_name: str = "SprintGuard", output_dir: str = "visualizations"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name for MLflow experiment
            output_dir: Directory to save visualizations
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set MLflow tracking URI to local directory
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        
        self.run = None
        self.run_id = None
        self.stage_counts = {}  # For Sankey diagram
        
        logging.info(f"[TRACKER] Initialized experiment: {experiment_name}")
        logging.info(f"[TRACKER] Visualizations will be saved to: {self.output_dir}")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Descriptive name for this run
            tags: Optional tags for categorization
        """
        if self.run is not None:
            logging.warning("[TRACKER] Previous run still active. Ending it first.")
            self.end_run()
        
        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})
        self.run_id = self.run.info.run_id
        
        logging.info(f"[TRACKER] Started run: {run_name}")
        logging.info(f"[TRACKER] Run ID: {self.run_id}")
        
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        mlflow.log_params(params)
        logging.info(f"[TRACKER] Logged {len(params)} parameters")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow."""
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        mlflow.log_metric(key, value, step=step)
    
    def log_stage_metrics(self, stage: str, metrics: Dict[str, float]):
        """
        Log metrics for a specific pipeline stage.
        
        Args:
            stage: Stage name (e.g., 'snorkel', 'cleanlab', 'training')
            metrics: Dictionary of metric name -> value
        """
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        # Prefix metrics with stage name
        prefixed_metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(prefixed_metrics)
        
        logging.info(f"[TRACKER] Logged {len(metrics)} metrics for stage '{stage}'")
        for k, v in metrics.items():
            logging.info(f"  {stage}/{k}: {v}")
    
    def log_figure(self, fig: plt.Figure, name: str, close: bool = True):
        """
        Save figure as both PDF and SVG, and log to MLflow.
        
        Args:
            fig: Matplotlib figure
            name: Filename (without extension)
            close: Whether to close figure after saving
        """
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        # Save as PDF (for paper)
        pdf_path = self.output_dir / f"{name}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        mlflow.log_artifact(str(pdf_path), artifact_path="visualizations")
        
        # Save as SVG (vector format)
        svg_path = self.output_dir / f"{name}.svg"
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        mlflow.log_artifact(str(svg_path), artifact_path="visualizations")
        
        logging.info(f"[TRACKER] Saved figure: {name}.pdf and {name}.svg")
        
        if close:
            plt.close(fig)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logging.info(f"[TRACKER] Logged artifact: {local_path}")
    
    def log_stage_count(self, stage: str, count: int):
        """
        Track data counts for each stage (for Sankey diagram).
        
        Args:
            stage: Stage name (e.g., 'raw', 'snorkel', 'cleanlab', 'final')
            count: Number of data points at this stage
        """
        self.stage_counts[stage] = count
        self.log_metric(f"data_flow/{stage}_count", count)
        logging.info(f"[TRACKER] Recorded {stage} count: {count}")
    
    def save_stage_counts(self):
        """Save stage counts as JSON artifact for Sankey diagram generation."""
        if self.run is None:
            logging.warning("[TRACKER] No active run. Call start_run() first.")
            return
        
        counts_path = self.output_dir / "stage_counts.json"
        with open(counts_path, 'w') as f:
            json.dump(self.stage_counts, f, indent=2)
        
        mlflow.log_artifact(str(counts_path), artifact_path="data_flow")
        logging.info(f"[TRACKER] Saved stage counts to {counts_path}")
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run is None:
            logging.warning("[TRACKER] No active run to end.")
            return
        
        # Save stage counts before ending
        if self.stage_counts:
            self.save_stage_counts()
        
        mlflow.end_run()
        logging.info(f"[TRACKER] Ended run: {self.run_id}")
        
        self.run = None
        self.run_id = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures run is ended."""
        self.end_run()
        return False

