The following is a high-signal executive summary and technical implementation plan based on the research analysis.

### **Executive Summary: Hybrid Explainable Risk Pipeline**

  * **Architecture:** Late Fusion (DistilBERT embeddings + Tabular features $\rightarrow$ XGBoost).
  * **Core Recommendation:** Unified **SHAP** framework. Use **TreeSHAP** for XGBoost and **PartitionExplainer** for DistilBERT.
  * **Latency Strategy:** **FastSHAP** (amortized estimation) + **Caching** to meet \<1s targets.
  * **Stakeholder Output:** **LLM-generated narratives** grounded in SHAP values (Feature $\rightarrow$ Text).

-----

### **1. Methodology Selection (Accuracy vs. Speed)**

| Feature | Recommendation | Justification |
| :--- | :--- | :--- |
| **Global Framework** | **SHAP** | Axiomatic consistency is required for risk audits. LIME is unstable (different results on re-runs) and fails to capture feature interactions in hybrid models.[1, 2] |
| **Tabular XAI** | **TreeSHAP** | Exact, fast, and natively supported by XGBoost. Handles mixed feature types efficiently.[3, 4] |
| **Text XAI** | **PartitionExplainer** | Superior to KernelSHAP for text. It respects token correlations (grouping "not" + "good") rather than treating words independently, which is crucial for sentiment/risk detection.[5] |
| **Deep Dive** | **Integrated Gradients** | Use **only** if token-level debugging is needed. **Avoid Attention Weights**; they indicate information flow, not causal importance, and often highlight irrelevant tokens (e.g., separators).[6, 7, 8] |
| **Concept XAI** | **TCAV** | Use **Concept Activation Vectors (TCAV)** to detect abstract risks (e.g., "Vagueness," "Technical Debt") directly in embedding space, even if specific keywords are missing.[9, 10] |

### **2. Latency Optimization (\<1s Requirement)**

Standard SHAP calculation on BERT models takes 2–10+ seconds. To achieve sub-second latency:

1.  **FastSHAP (Amortized Inference):** Train a lightweight neural network (explainer model) to *predict* SHAP values in a single forward pass (\~20ms) instead of running the iterative SHAP algorithm at inference time.[11]
2.  **Quantization:** Deploy DistilBERT in **ONNX Runtime** with INT8 quantization. Reduces memory footprint by 4x and inference time by \~50%.[12, 13]
3.  **Semantic Caching:** Hash input text. If the requirements text hasn't changed, retrieve pre-computed risk explanations from Redis. Most project data is read-heavy, not write-heavy.[14]
4.  **Async Architecture:** Display the Risk Score immediately (\<100ms). Load the Explanation asynchronously.

### **3. Stakeholder Communication Strategy**

Do not show raw SHAP plots to Scrum Masters. Use **Generative Translation**.

  * **Technique:** Feed the top 3 positive and negative SHAP features into a template-constrained LLM (e.g., GPT-3.5/4 or Llama 3).
  * **Prompt Pattern:** "You are a Risk Analyst. Based on the following drivers: [Feature A: High impact],, summarize why this story is high risk in one sentence."
  * **Example Output:** *"This story is high risk primarily because the acceptance criteria are vague (Text Risk) and the team velocity has dropped by 15% (Tabular Risk)."*.[15, 16]

### **4. Implementation Plan & Libraries**

**Phase 1: Core Pipeline**

  * **Model:** `transformers` (DistilBERT), `xgboost`
  * **Wrapper:** Use `shap.PartitionExplainer` with a custom masker that handles the DistilBERT tokenizer.[17]
    ```python
    import shap
    # Wrap pipeline for SHAP
    explainer = shap.PartitionExplainer(model_predict, shap.maskers.Text(tokenizer))
    shap_values = explainer(text_data)
    ```

**Phase 2: Concept Detection (TCAV)**

  * **Library:** `captum` or `tcav` (Google).
  * **Action:** Define concept sets (e.g., 50 sentences representing "Security Risks"). Train a linear classifier on embeddings to find the "Security Direction." Check if new stories align with this vector.[18]

**Phase 3: Dashboard**

  * **Library:** `OmniXAI` (provides a unified view for hybrid models) or `Streamlit`.
  * **Visuals:**
      * **Scrum Master:** Traffic light risk score + LLM text summary.
      * **Developer:** Text highlighting (red/blue) on specific words using SHAP values.

### **5. Decision Matrix**

| Requirement | Solution |
| :--- | :--- |
| **Accuracy/Trust** | **SHAP (Partition + Tree)** |
| **Speed (\<1s)** | **FastSHAP + ONNX Quantization** |
| **Concepts** | **TCAV (Captum)** |
| **User Interface** | **LLM Narratives (OpenAI/Llama)** |