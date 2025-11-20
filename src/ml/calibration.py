"""
Probability Calibration for XGBoost Risk Model
Uses Isotonic Regression to improve probability estimates.
"""
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from typing import Tuple


class MulticlassCalibrator:
    """
    Calibrates multi-class XGBoost probabilities using Isotonic Regression.
    
    Research shows raw XGBoost probabilities are not well-calibrated.
    This wrapper applies per-class isotonic calibration.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.calibrators = {}  # One per class
        self.is_fitted = False
    
    def fit(self, y_val: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit isotonic regressors on validation set.
        
        Args:
            y_val: True labels (N,)
            y_pred_proba: Predicted probabilities (N, 3)
        """
        print("Fitting probability calibrators...")
        
        num_classes = y_pred_proba.shape[1]
        
        for class_idx in range(num_classes):
            # Create binary labels (class vs rest)
            y_binary = (y_val == class_idx).astype(int)
            
            # Fit isotonic regressor
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred_proba[:, class_idx], y_binary)
            
            self.calibrators[class_idx] = calibrator
        
        self.is_fitted = True
        print(f"✓ Calibrated {num_classes} classes")
    
    def predict_proba(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            y_pred_proba: Raw probabilities (N, 3)
            
        Returns:
            Calibrated probabilities (N, 3)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        calibrated = np.zeros_like(y_pred_proba)
        
        for class_idx, calibrator in self.calibrators.items():
            calibrated[:, class_idx] = calibrator.predict(y_pred_proba[:, class_idx])
        
        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums
        
        return calibrated
    
    def save(self, path: str):
        """Save calibrator to disk."""
        joblib.dump({
            'calibrators': self.calibrators,
            'is_fitted': self.is_fitted
        }, path)
        print(f"✓ Saved calibrator to {path}")
    
    @staticmethod
    def load(path: str) -> 'MulticlassCalibrator':
        """Load calibrator from disk."""
        data = joblib.load(path)
        calibrator = MulticlassCalibrator()
        calibrator.calibrators = data['calibrators']
        calibrator.is_fitted = data['is_fitted']
        return calibrator


def evaluate_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10):
    """
    Evaluate calibration using Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (N,)
        y_pred_proba: Predicted probabilities (N, num_classes)
        n_bins: Number of bins for calibration curve
        
    Returns:
        ECE score (lower is better)
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_size = np.sum(in_bin)
            
            # Weighted difference
            ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)
    
    return ece

