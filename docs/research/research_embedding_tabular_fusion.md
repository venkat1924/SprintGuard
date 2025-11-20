### **1. Feature Extraction Pipeline**

  * **Text Source:** Use **Mean Pooling** from the last hidden layer of BERT-base (768-dim). Avoid \`\` token unless fine-tuned specifically for this task. [2, 3]
  * **Dimensionality Reduction:** Apply **PCA** to reduce embeddings to **32–64 components**.
      * *Why:* Aligns variance with the axis-parallel splits of XGBoost and removes the "noise tail" of the embedding space. [4, 5]
  * **Normalization:**
      * **Embeddings:** L2-normalize *before* PCA.
      * **Tabular:** Standardization (Z-score) is optional for XGBoost but recommended if using PCA on the combined set.

### **2. Fusion Architecture Strategy**

  * **Primary Strategy (Robust):** **Stacked Generalization (Late Fusion)**
    1.  Fine-tune BERT on text only.
    2.  Extract **logits** or **penultimate layer activations**.
    3.  Concatenate these compact signals with tabular features as input to XGBoost. [1, 6]
  * **Alternative Strategy (Fast):** **Concatenation (Early Fusion)**
    1.  Concatenate \`\` + `[PCA_Embeddings (32-64)]`.
    2.  *Do not* feed raw 768-dim vectors directly; it dilutes feature importance.

### **3. XGBoost Configuration (Critical)**

You must alter defaults to handle the density difference between symbolic and embedding features.

| Parameter | Recommended Value | Purpose |
| :--- | :--- | :--- |
| **`colsample_bytree`** | **0.3 – 0.5** | Forces trees to consider tabular features by preventing greedy selection of embedding dimensions at every split. [7] |
| **`reg_alpha` (L1)** | **\> 0 (e.g., 0.1-1.0)** | Activates sparsity; aggressively zeros out irrelevant embedding dimensions. [8] |
| **`feature_weights`** | **Tabular: 1.0, Emb: 0.5** | Explicitly biases the model to trust symbolic features more than dense vectors. |
| **`interaction_constraints`** | **, [Emb]]** | *Optional:* Disallow interactions between modalities if overfitting occurs (forces additive rather than multiplicative logic). [9] |

### **4. Handling Imbalance**

  * **Objective Function:** Use **Focal Loss** (`objective='binary:focal'`) rather than standard `scale_pos_weight`.
      * *Reason:* Embeddings create "easy" examples that dominate gradients; Focal Loss forces focus on "hard" minority examples. [10]

### **5. Code Snippet (Fusion Setup)**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

# 1. Preprocessing
pca = PCA(n_components=50)
X_text_pca = pca.fit_transform(embeddings_768) # L2 normalized input
X_tab_scaled = StandardScaler().fit_transform(X_tabular)

# 2. Fusion
X_combined = np.hstack([X_tab_scaled, X_text_pca])

# 3. Weighting (Give tabular features 2x importance)
f_weights = [2.0] * X_tab_scaled.shape[1] + [1.0] * X_text_pca.shape[1]
dtrain = xgb.DMatrix(X_combined, label=y, feature_weights=f_weights)

# 4. Training with Regularization
params = {
    'colsample_bytree': 0.4,  # Critical for mixed modalities
    'reg_alpha': 0.5,         # L1 to prune noise
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
model = xgb.train(params, dtrain)
```