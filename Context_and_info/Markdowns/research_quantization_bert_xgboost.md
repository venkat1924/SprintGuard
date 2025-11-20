Here is a high-signal summary of the findings to support your implementation plan.

### 1\. Recommended Architecture

  * **Embedding Engine:** DistilBERT (converted to ONNX, Dynamic INT8 Quantization).
  * **Classification Engine:** XGBoost (compiled via Treelite).
  * **Inference Flow:** Tokenization (Rust-based) $\rightarrow$ ONNX Runtime (CPU) $\rightarrow$ CLS Token Extraction $\rightarrow$ Treelite Prediction.

### 2\. Performance Expectations (CPU)

  * **Total Latency:** **\< 50ms** per single user story (Target: \< 1000ms).
      * DistilBERT Inference: \~15–30ms.[1]
      * XGBoost Inference: \< 0.1ms (via Treelite).[2]
      * Tokenization/Overhead: \~5–10ms.
  * **Throughput Gain:** 3x–4x speedup over PyTorch FP32.[1]

### 3\. Accuracy & Data Integrity

  * **Accuracy Degradation:** Negligible (\< 1% drop on GLUE benchmarks).[3]
  * **Adversarial Robustness:** **+18.8% improvement**. Quantization acts as a regularizer, filtering out high-frequency noise often found in text artifacts.[4]
  * **Embedding Quality:** Retains \>0.80 cosine similarity with FP32 vectors, preserving the geometric clusters required for XGBoost splits.[5]

### 4\. Implementation Specifications

  * **Quantization Type:** Dynamic Quantization (Weights = INT8, Activations = FP32 $\rightarrow$ INT8 at runtime). This avoids the need for calibration data and handles the high variance of Transformer activations better than static quantization.[6]
  * **Optimization Flags:** Enable `avx512_vnni` in ONNX Runtime to leverage vector instructions on modern Intel CPUs.[7]
  * **XGBoost Compilation:** Compile the trained XGBoost model to a shared C library (`.so`) using Treelite. This removes Python interpreter overhead during the decision tree traversal.[8]
  * **Tokenizer Config:** Set `max_length=128` (or 64). User stories are short; processing 512 tokens is computationally wasteful ($O(L^2)$ complexity).[9]

### 5\. Code Snippets for Engineering

**ONNX Dynamic Quantization:**

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Use AVX-512 optimization for Intel CPUs
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained("distilbert-base-uncased", feature="feature-extraction")
quantizer.quantize(save_dir="onnx_int8", quantization_config=qconfig)
```

[7]

**Treelite Compilation:**

```python
import treelite
import treelite_runtime

# 1. Load trained XGBoost model and compile to C code
model = treelite.Model.load("risk_model.json", model_format="xgboost")
model.export_lib(toolchain="gcc", libpath="./risk_predictor.so", params={"parallel_comp": 40})

# 2. Production Inference
predictor = treelite_runtime.Predictor("./risk_predictor.so")
# Input must be a numpy array (batch_size, embedding_dim)
scores = predictor.predict(batch_of_embeddings)
```

[10]