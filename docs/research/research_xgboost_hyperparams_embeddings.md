Here is your deployment plan for the Hybrid BERT-XGBoost Risk Classifier.

### 1. Core Configuration Protocol
**Target Architecture:** XGBoost (Histogram-based)
**Feature Space:** 778 Dimensions (768 Dense Embeddings + 10 Symbolic)

**Fixed Parameters (Do Not Tune)**
*   `tree_method`: **`hist`** (Essential for speed/memory)
*   `max_bin`: **64** (Reduces noise in embeddings; improves L2 cache hits for inference)
*   `booster`: `gbtree`
*   `objective`: **`multi:softprob`** (Required for probability calibration)
*   `num_class`: 3
*   `eval_metric`: `mlogloss` (Optimizes probability density, not just accuracy)

**Hyperparameter Search Space (Bayesian/Optuna)**
*   **`colsample_bytree`**: **0.3 – 0.5** (Critical: Forces model to not over-rely on specific embedding dimensions)
*   **`colsample_bynode`**: **0.5 – 0.7** (High randomization needed for dense vectors)
*   **`max_depth`**: **4 – 6** (Shallow trees prevent memorizing embedding noise)
*   **`min_child_weight`**: **5 – 10** (High value required to stop isolating sparse outliers in vector space)
*   **`subsample`**: 0.6 – 0.8
*   **`alpha` (L1)**: **0.1 – 1.0** (Induces sparsity within the dense embedding vectors)
*   **`eta`**: 0.05 – 0.1

### 2. Handling Class Imbalance (Low/Med/High)
**Do not use `scale_pos_weight`** (It is binary-only and ignored in multi-class).

*   **Strategy A (Primary):** Compute **Sample Weights** based on inverse class frequency ($N_{total} / (3 * N_{class})$) and pass `sample_weight` vector to the `.fit()` method.
*   **Strategy B (Ordinal Output):** Post-processing. Instead of `argmax`, calculate an **Expected Risk Score**:
    $$Risk = 0 \times P(Low) + 1 \times P(Med) + 2 \times P(High)$$
    Threshold this continuous score (e.g., $>1.6$ = High) for better operational sensitivity.

### 3. Feature Engineering & Constraints
*   **Symbolic Features:** Apply **Monotonic Constraints** to the 10 symbolic features where directionality is known (e.g., `constraints=(1, 0, 1...)`).
*   **Embeddings:** Apply **Zero** constraints. Do not scale/normalize embeddings; XGBoost handles raw values.
*   **Interaction Constraints:** If symbolic features are being ignored, strictly constrain the first 1-2 splits of every tree to be Symbolic features (if using `exact` method) or rely on the low `colsample_bytree` (0.3) to statistically force their inclusion.

### 4. Inference Optimization (<1s Latency)
*   **Bottleneck:** Feature extraction (BERT) will take ~50-200ms. XGBoost must be negligible (<10ms).
*   **Deployment:** Compile the trained model using **Treelite**.[1, 2]
    *   *Why:* Converts tree ensemble into C code, optimizing branch prediction and eliminating Python overhead.
    *   *Speedup:* ~2-5x vs native XGBoost.
*   **Fallback:** If Treelite is unavailable, use **ONNX Runtime** with `graph_optimization_level='ORT_ENABLE_ALL'`.

### 5. Execution Roadmap
1.  **Baseline:** Train with `tree_method='hist'`, `max_depth=4`, `colsample_bytree=0.4`, `sample_weight=balanced`.
2.  **Tuning:** Run Optuna for 50 trials maximizing negative `mlogloss`.
3.  **Compilation:** Export best model to `.so` (Shared Object) via Treelite.
4.  **Integration:** Pipeline: Input Text $\to$ BERT (ONNX/TensorRT) $\to$ Concat(Sym, Emb) $\to$ Treelite Predictor.