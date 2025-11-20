Here is the high-signal, actionable summary for your implementation plan.

### **Strategic Framework: Economic Risk Minimization**
Do not optimize for Accuracy or F1-score; these metrics treat all errors equally. In software project management, a **False Negative** (missing a High-Risk defect) is exponentially more costly than a **False Positive** (unnecessary review).
*   **Objective Function:** Minimize **Expected Cost** (Bayes Risk), not error rate.[1, 2]
*   **Target Metric:** **Total Misclassification Cost** or **Recall at Fixed Precision** (e.g., 95% Recall for High Risk).[3, 4]

### **1. Probability Calibration (Mandatory Pre-requisite)**
Raw probabilities from XGBoost or Neural Networks are not trustworthy confidence scores. You must calibrate them before applying thresholds.
*   **Recommended Method:** **Dirichlet Calibration** (Best for multi-class).[5, 6]
*   **Alternative:** **Isotonic Regression** (Only if >1,000 samples per class).[7, 8]
*   **Fallback:** **Temperature Scaling** (For Neural Networks, preserves rank).[9]
*   *Action:* Pipeline raw model outputs $\rightarrow$ Calibrator $\rightarrow$ Probability Vector.

### **2. Optimal Threshold Selection Algorithms**
Abandon fixed thresholds (e.g., >0.5). Use one of these two data-driven approaches:

**A. Cost-Sensitive Minimization (Gold Standard)**
Define a Cost Matrix ($C$) where $C_{ij}$ is the cost of predicting class $i$ when truth is $j$.
*   **Recommended Ratios:**
    *   High Risk False Negative ($C_{Low, High}$): **10 to 100** units.
    *   High Risk False Positive ($C_{High, Low}$): **1 unit**.
*   **Decision Rule:** For input $x$, predict class $k$ that minimizes expected cost:
    $$\hat{y} = \underset{k}{\mathrm{argmin}} \sum_{j \in \{L,M,H\}} P(y=j|x) \cdot C_{kj}$$
    *Note: This naturally shifts the threshold for High Risk to low probabilities (e.g., 0.15-0.25) based on the severity of the error.[10, 11]*

**B. Ordinal Decomposition (Frank & Hall)**
Respects the hierarchy ($Low < Med < High$). Decompose the 3-class problem into 2 binary problems:
1.  **Risk > Low?** (Threshold $t_1$)
2.  **Risk > Medium?** (Threshold $t_2$)
*   *Action:* Optimize $t_1$ for high recall (safety net), optimize $t_2$ for precision (escalation trigger).[12, 13]

### **3. Scope & Adaptation**
*   **Scope:** Use **Project-Specific** thresholds. Global thresholds fail due to data heterogeneity (e.g., different definitions of "High Risk" between teams).[14, 15]
*   **Adaptation:** Implement **Concept Drift Detection** (ADWIN or DDM). Risk definitions shift as projects mature. Re-calibrate thresholds when drift is detected (typically every 2-4 sprints).[16, 17]

### **4. Python Implementation Stack**
*   **Calibration:** `sklearn.calibration.CalibratedClassifierCV` (Isotonic/Sigmoid) or `dirichletcal` library.[8, 18]
*   **Optimization:** `scipy.optimize.minimize` (Nelder-Mead) to find optimal $t_1, t_2$ that minimize the Cost Matrix loss on the validation set.[19, 20]
*   **Drift:** `river.drift.ADWIN` for online threshold adjustment.[21]