Here is the high-signal implementation summary based on the research.

### **1. Core Architecture**
*   **Embedding Model:** Use **`sentence-transformers/all-mpnet-base-v2`**. It offers the best trade-off between speed and semantic quality for technical text.[1, 2]
*   **Vector Normalization:** Normalize all vectors to $L_2$ unit length immediately upon generation.
*   **Distance Metric:** Use **Inner Product (Dot Product)**. On normalized vectors, this is mathematically equivalent to Cosine Similarity but computationally faster.[3]
*   **Index Strategy:** Use **FAISS `IndexFlatIP`**.
    *   **Reasoning:** For <50k vectors (~150MB RAM), "Brute Force" exact search is sub-millisecond (<5ms).
    *   **Avoid:** Approximate methods (HNSW/IVF) and `sklearn.neighbors` (orders of magnitude slower).[4, 5]

### **2. Retrieval Pipeline**
1.  **Recall (Stage 1):** Retrieve top 100 candidates using semantic similarity.
2.  **Filter (Stage 2):** Apply strict hard filters for non-negotiable context (e.g., "Must be `Mobile App` project") if required, but prefer soft-weighting to maximize cross-pollination of ideas.[6]
3.  **Re-Rank (Stage 3):** Apply a **Risk-Aware Scoring** formula to bubble up high-risk benchmarks:
    $$Score_{final} = Similarity_{vector} \times (1 + w_{risk} \times RiskScore_{historical})$$
    *   *RiskScore* should derive from historical defect density and revert rates.[7]

### **3. Data & Training**
*   **Fine-Tuning:** Essential for software jargon (e.g., distinguishing "Java" language from "Java" coffee). Use **Triplet Loss** (Anchor, Positive, Negative).[8, 9]
*   **Negative Mining:** Use "Dissimilar" stories (e.g., UI vs. Backend tasks) as negative examples during training to sharpen decision boundaries.

### **4. User Interface & Experience**
*   **Cognitive Load:** Display exactly **3-5 recommendations**. More options increase cognitive load and reduce decision quality.[10, 11]
*   **Visualization:** Use **UMAP** to project the retrieval set (top 50) into a 2D "Risk Landscape" to visually place the new story relative to historical clusters.[12, 13]
*   **Explainability:** Highlight keywords in the retrieved text that drove the similarity match (e.g., "Matched on: *OAuth protocol*") to build trust.[14]

### **5. Performance Budget**
*   **Total Latency Target:** < 200ms
    *   Embedding Generation: ~50ms (CPU)
    *   FAISS Search: < 5ms
    *   Re-ranking & Metadata Fetch: ~20ms
    *   Network Overhead: ~50ms