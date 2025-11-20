Here is the high-signal technical summary for your DistilBERT-XGBoost implementation plan.

### **1. Architecture Blueprint (Neuro-Symbolic)**

  * **Model Type:** Hybrid Early Fusion.
  * **Inputs:**
    1.  **Neural Branch:** Raw User Story Text $\rightarrow$ DistilBERT Tokenizer $\rightarrow$ DistilBERT Model $\rightarrow$ \`\` Embedding Vector (768 dimensions).
    2.  **Symbolic Branch:** Raw User Story Text $\rightarrow$ `spaCy`/`textstat` Pipelines $\rightarrow$ Normalized Feature Vector (approx. 10-15 dimensions).
  * **Fusion:** Concatenate \`\` vector + Symbolic Vector.
  * **Classifier:** XGBoost (handles non-linear interactions between embeddings and symbolic "hard" rules).
  * **Target Output:** Risk Class (High/Medium/Low) or Risk Score (0.0 - 1.0).

### **2. Symbolic Feature Extraction Specifications**

Concatenate these calculated values to your neural embeddings.

#### **A. Readability & Complexity (Cognitive Load)**

  * **Library:** `textstat`
  * **Metric 1: Flesch Reading Ease (FRE)**
      * *Signal:* Scores **\< 30** indicate "Very Difficult" (High Risk).
      * *Feature:* `fre_score` (float).
  * **Metric 2: Gunning Fog Index**
      * *Signal:* Scores **\> 16** indicate structural complexity requiring graduate-level reading (High Risk).
      * *Feature:* `fog_index` (float).
  * **Metric 3: Lexical Density**
      * *Signal:* Ratio of content words (nouns/verbs) to total words. High density = information overload.
      * *Feature:* `lexical_density` (float).

#### **B. Ambiguity Indicators (Verification Risk)**

  * **Library:** `spaCy` (Matcher & POS Tagging)
  * **Metric 1: Weak Modal Density**
      * *Keywords:* *might, could, should, may, ought*.
      * *Threshold:* Density **\> 0.15** (approx 1 weak modal per 6 sentences) correlates with verification failure.
      * *Feature:* `weak_modal_density` (count / sentence\_count).
  * **Metric 2: Vague Quantifiers**
      * *Keywords:* *fast, user-friendly, robust, easy, efficient, seamless, many, few, several, TBD, etc.*
      * *Signal:* Any presence implies untestability.
      * *Feature:* `vagueness_count` (int) or `has_vagueness` (bool).
  * **Metric 3: Passive Voice**
      * *Signal:* Hides the "actor" (who performs the action?).
      * *Threshold:* **\> 20%** of sentences in passive voice.
      * *Feature:* `passive_voice_ratio` (passive\_sentences / total\_sentences).

#### **C. Domain Risk Lexicons (Technical Debt & Security)**

  * **Library:** `spaCy` (PhraseMatcher) or RegEx
  * **Lexicon 1: Self-Admitted Technical Debt (SATD)**
      * *Keywords:* *hack, fixme, todo, workaround, temporary, ugly, hardcoded, spaghetti, quick fix*.
      * *Feature:* `satd_flag` (bool).
  * **Lexicon 2: Security Risk**
      * *Keywords:* *auth, token, jwt, encrypt, pii, gdpr, role, permission, injection, xss, secret*.
      * *Feature:* `security_term_count` (int).
  * **Lexicon 3: Integration/Legacy Complexity**
      * *Keywords:* *legacy, mainframe, wrapper, migration, api, oauth, synchronization, handshake, middleware*.
      * *Feature:* `complexity_term_count` (int).

#### **D. Sentiment (Developer Frustration)**

  * **Library:** Use a model trained on SE data (e.g., Senti4SD), **not** VADER/TextBlob.
  * **Signal:** High negative sentiment in comments/stories correlates with frustration and higher defect density.
  * **Feature:** `sentiment_polarity` (categorical: -1, 0, 1).

### **3. Implementation Recommendations**

**Python Stack:**

```python
import spacy
import textstat
import xgboost as xgb
from transformers import DistilBertModel, DistilBertTokenizer

# 1. Load Resources
nlp = spacy.load("en_core_web_sm")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 2. Feature Extraction Function
def get_hybrid_vector(text):
    # A. Neural Embedding
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    neural_feat = outputs.last_hidden_state[:, 0, :].detach().numpy() # token

    # B. Symbolic Features
    doc = nlp(text)
    fre = textstat.flesch_reading_ease(text)
    fog = textstat.gunning_fog(text)
    #... calculate densities for modals, passive voice, etc....
    symbolic_feat = [fre, fog, modal_density, vagueness_flag, satd_flag]

    # C. Concatenate
    return np.concatenate([neural_feat, symbolic_feat])
```

### **4. Expected Performance & Feature Importance**

  * **Target Baseline (Pure DistilBERT):** \~0.80 - 0.85 F1-Score.
  * **Target Hybrid (DistilBERT + XGBoost):** **\> 0.88 F1-Score**.
  * **Primary Gain:** Significant increase in **Recall** for the "High Risk" class (catching vague/risky requirements that look semantically "normal" to BERT).
  * **Top Features (SHAP Value Ranking):**
    1.  SATD Keywords (e.g., "TODO", "Hack") — *Highest Signal*
    2.  Vague Quantifier Count
    3.  Neural Embedding Dimensions (Context)
    4.  Gunning Fog Index
    5.  Weak Modal Density