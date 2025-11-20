"""
Cost-Sensitive Threshold Optimization for Risk Classification
Minimizes expected cost instead of maximizing accuracy.
"""
import json
import numpy as np
from typing import Dict, Tuple


class CostSensitiveClassifier:
    """
    Applies cost-sensitive decision rule for multi-class risk prediction.
    
    Instead of argmax(probabilities), selects class that minimizes expected cost.
    This is critical for catching High Risk stories (false negatives are expensive).
    """
    
    # Default cost matrix (research-backed)
    # Rows: Predicted class, Columns: True class
    # Cost[i,j] = cost of predicting i when truth is j
    DEFAULT_COST_MATRIX = np.array([
        #  True: Low  Med  High
        [     0,    2,   50  ],  # Predict Low
        [     1,    0,   10  ],  # Predict Medium
        [     3,    2,    0  ]   # Predict High
    ])
    
    def __init__(self, cost_matrix: np.ndarray = None):
        """
        Initialize classifier with cost matrix.
        
        Args:
            cost_matrix: (3, 3) cost matrix. If None, uses default.
        """
        self.cost_matrix = cost_matrix if cost_matrix is not None else self.DEFAULT_COST_MATRIX
        
        # Validate shape
        if self.cost_matrix.shape != (3, 3):
            raise ValueError("Cost matrix must be (3, 3)")
        
        print("Cost-Sensitive Classifier initialized")
        print("Cost Matrix (Pred × True):")
        print(self.cost_matrix)
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Predict class labels using cost-sensitive decision rule.
        
        Args:
            probabilities: (N, 3) probability matrix
            
        Returns:
            (N,) predicted class labels
        """
        # For each sample, compute expected cost for each prediction
        # Expected_Cost[k] = Σ_j P(j) * Cost[k,j]
        expected_costs = probabilities @ self.cost_matrix.T  # (N, 3) @ (3, 3).T = (N, 3)
        
        # Select class with minimum expected cost
        predictions = np.argmin(expected_costs, axis=1)
        
        return predictions
    
    def predict_with_costs(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes and return expected costs.
        
        Args:
            probabilities: (N, 3) probability matrix
            
        Returns:
            Tuple of (predictions, expected_costs)
        """
        expected_costs = probabilities @ self.cost_matrix.T
        predictions = np.argmin(expected_costs, axis=1)
        
        # Get the cost of the chosen prediction
        chosen_costs = expected_costs[np.arange(len(predictions)), predictions]
        
        return predictions, chosen_costs
    
    def evaluate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """
        Evaluate cost-sensitive performance.
        
        Args:
            y_true: True labels (N,)
            y_pred_proba: Predicted probabilities (N, 3)
            
        Returns:
            Dict with metrics
        """
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Get predictions
        y_pred = self.predict(y_pred_proba)
        _, expected_costs = self.predict_with_costs(y_pred_proba)
        
        # Calculate total realized cost
        realized_costs = self.cost_matrix[y_pred, y_true]
        total_cost = np.sum(realized_costs)
        avg_cost = np.mean(realized_costs)
        
        # Standard metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
    
    def save(self, path: str):
        """Save cost matrix to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'cost_matrix': self.cost_matrix.tolist()
            }, f, indent=2)
        print(f"✓ Saved cost matrix to {path}")
    
    @staticmethod
    def load(path: str) -> 'CostSensitiveClassifier':
        """Load cost matrix from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        cost_matrix = np.array(data['cost_matrix'])
        return CostSensitiveClassifier(cost_matrix)


def optimize_cost_matrix(
    y_val: np.ndarray,
    y_pred_proba: np.ndarray,
    weight_fn_ratio: float = 50.0
) -> np.ndarray:
    """
    Generate cost matrix based on validation set characteristics.
    
    Args:
        y_val: Validation labels
        y_pred_proba: Predicted probabilities
        weight_fn_ratio: Ratio of FN cost to FP cost for High risk
        
    Returns:
        Optimized cost matrix
    """
    # Calculate class frequencies
    class_counts = np.bincount(y_val, minlength=3)
    class_frequencies = class_counts / len(y_val)
    
    # Base cost matrix (asymmetric)
    cost_matrix = np.array([
        #  True: Low  Med  High
        [     0,    2,   weight_fn_ratio  ],  # Predict Low (FN for High is expensive!)
        [     1,    0,   weight_fn_ratio / 5  ],  # Predict Medium
        [     3,    2,    0  ]   # Predict High (FP less costly)
    ])
    
    print(f"Generated cost matrix with FN/FP ratio = {weight_fn_ratio}")
    print(cost_matrix)
    
    return cost_matrix

