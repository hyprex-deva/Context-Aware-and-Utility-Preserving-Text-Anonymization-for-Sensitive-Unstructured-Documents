# Context-Aware & Utility-Preserving Text Anonymization for Sensitive Unstructured Documents

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app.py)
[![Evaluation Benchmark](https://img.shields.io/badge/Benchmark-100--Doc%20AI4Privacy-purple.svg)](data/evaluation_100_results.csv)

A modular, research-grade NLP framework for sensitive unstructured document anonymization that maximizes **privacy defense** against both direct identifiers and contextual quasi-identifiers while preserving **downstream semantic utility**.

---

## 🏛️ System Architecture

```text
                               Raw Unstructured Document
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Token-Level Direct PII Detection & Synthetic Surrogate Replacement            │
│  ├─ High-Precision Regex Detectors (Email, Phone, SSN, IP, Credit Cards, Passwords)   │
│  ├─ Transformer Token Classifier NER (PER, ORG, LOC, MISC)                             │
│  ├─ Collision Resolver (Prioritizes exact boundaries & higher confidence spans)       │
│  └─ Deterministic Faker Generator (Preserves document-level consistency for entities) │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼ [Stage 1 Output: Grammatical Surrogate Text]
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Semantic Reasoning & Quasi-Identifier (QI) Generalization                    │
│  ├─ Local Quantized SLM: Ollama (qwen2.5:1.5b)                                        │
│  │   └─ Hierarchical coarsening of dates, monetary amounts, rare roles, demographics   │
│  │                                                                                     │
│  ├─ SLM-Independent Deterministic QI Detection Layer (QuasiIdentifierDetector)         │
│  │   ├─ Measurable residual risk scoring r_i ∈ [0.0, 1.0]                              │
│  │   └─ Conservative Convex Compound Aggregation:                                     │
│  │         r_compound = α * mean(r_k) + (1 - α) * prod(r_k)   (α = 0.40)               │
│  │                                                                                     │
│  └─ Three-Way Multi-Criteria Composite Guardrail:                                      │
│        ├─ Semantic Similarity (S_semantic): SentenceTransformers (all-MiniLM-L6-v2)    │
│        ├─ Privacy Abstraction (S_qi_abstraction): Independent QI residual risk score   │
│        ├─ Readability & Integrity (S_readability): Length-ratio & syntactic penalty   │
│        ├─ S_composite = (0.50 * S_sem) + (0.30 * S_qi) + (0.20 * S_read)               │
│        │                                                                               │
│        ├─ ACCEPT Candidate Rewriting IF:                                               │
│        │     S_composite >= tau (0.72) AND                                             │
│        │     S_semantic  >= floor_sim (0.60) AND                                       │
│        │     S_qi_abstraction >= qi_floor (0.50)                                       │
│        │                                                                               │
│        └─ REJECT ➔ Automatic Deterministic Fallback to Stage 1 Surrogate Text         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                   Final Privacy-Safe & Utility-Preserved Document
```

---

## 📊 Research Benchmark Evaluation (100 AI4Privacy Documents)

Evaluated on a fixed, reproducible manifest of 100 stratified English documents from `ai4privacy/pii-masking-300k` ([`data/evaluation_100_manifest.csv`](data/evaluation_100_manifest.csv)) across 4 comparative systems:

### 1. System Comparison Matrix

| System / Method | PII Prec (Rel.) $\uparrow$ | PII Rec (Rel.) $\uparrow$ | PII F1 (Rel.) $\uparrow$ | PII F1 (Exact) $\uparrow$ | PII Leak Rate $\downarrow$ | PII Rem. Rate $\uparrow$ | ROUGE-L F1 $\uparrow$ | BLEU-4 $\uparrow$ | Cosine Sim $\uparrow$ | BERTScore F1 $\uparrow$ | Readability $\uparrow$ | QI Abst. Score $\uparrow$ | QI Mit. Rate $\uparrow$ | Fallback Rate $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rigid Redaction** (`[REDACTED]`) | 0.8764 | 0.5270 | 0.6582 | 0.4375 | 0.4671 | 0.5329 | **0.6858** | **0.4706** | 0.6928 | 0.8601 | **0.9456** | N/A | N/A | N/A |
| **Microsoft Presidio** (`<TAGS>`) | 0.6744 | 0.5031 | 0.5763 | **0.4573** | 0.4987 | 0.5013 | 0.6607 | 0.4341 | **0.8507** | 0.8807 | 0.9422 | N/A | N/A | N/A |
| **Proposed: Stage 1** (Surrogates) | **0.8891** | **0.5346** | **0.6677** | 0.4375 | 0.4759 | 0.5241 | 0.6390 | 0.4512 | 0.7863 | **0.9179** | 0.9007 | N/A | N/A | N/A |
| **Proposed: Two-Stage** (Stage 1 + Stage 2) | **0.8891** | **0.5346** | **0.6677** | 0.4375 | **0.4516** | **0.5484** | 0.5899 | 0.3978 | 0.7560 | 0.9026 | 0.8685 | **0.4872** | **54.93%** | 0.7800 |

*Data source: [`data/evaluation_100_results.csv`](data/evaluation_100_results.csv)*

### 2. Detected Quasi-Identifier (QI) Abstraction Breakdown

Analyzed using the SLM-independent deterministic QI detector across 142 detected contextual attributes:

| Detected QI Category | Count | Initial Risk | Residual Risk | Mitigated Count | Partial Count | Exposed Count | Mitigation Rate | Exposure Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DATE`** | 109 | 1.0000 | 0.5797 | 35 | 21 | 53 | **51.38%** | 48.62% |
| **`DEMOGRAPHIC`** | 31 | 1.0000 | 0.4181 | 10 | 12 | 9 | **70.97%** | 29.03% |
| **`MONEY`** | 2 | 1.0000 | 1.0000 | 0 | 0 | 2 | **0.00%** | 100.00% |
| **Total / Overall** | **142** | **1.0000** | **0.5504** | **45** | **33** | **64** | **54.93%** | **45.07%** |

*Data source: [`data/evaluation_100_qi_breakdown.csv`](data/evaluation_100_qi_breakdown.csv)*

### 3. Guardrail Ablation Study

Evaluated across cached candidate rewrites to demonstrate the necessity of multi-criteria gating:

- **Approach A (Single Cosine Threshold $S_{\text{sim}} \ge 0.80$):** Acceptance Rate = **41.0%** | Fallback Rate = **59.0%**
- **Approach B (Proposed Three-Way Composite Guardrail):** Acceptance Rate = **22.0%** | Fallback Rate = **78.0%**

*Key Finding:* A single similarity threshold mistakenly accepted 19 candidate rewrites that had high cosine similarity but failed to abstract sensitive quasi-identifiers. The 3-way guardrail successfully blocked these privacy leaks.

---

## 📂 Project Structure

```text
Text-Anonymization/
├── data/
│   ├── evaluation_100_manifest.csv        # 100-document stratified manifest
│   ├── evaluation_100_dataset.json        # Snapshot of 100 benchmark documents
│   ├── evaluation_100_results.csv         # Main comparative benchmark matrix
│   ├── evaluation_100_entity_breakdown.csv# Direct PII metrics by entity type
│   ├── evaluation_100_qi_breakdown.csv    # QI abstraction analysis by category
│   ├── evaluation_100_error_analysis.json # Qualitative exemplars & failure cases
│   ├── evaluation_100_summary.json        # Machine-readable summary dictionary
│   ├── evaluation_100_report.txt          # Human-readable benchmark report
│   └── evaluation_100_stage2_results.json # Full Stage 2 inference cache
├── src/
│   ├── __init__.py                        # Package exports
│   ├── stage1_direct_pii.py               # Token-Level Direct PII Detection & Surrogates
│   ├── stage2_semantic_defense.py         # SLM Defense & Three-Way Composite Guardrail
│   ├── quasi_identifier_detector.py       # Independent Deterministic QI Detection Layer
│   ├── baselines.py                       # Microsoft Presidio & Rigid Redaction baselines
│   └── evaluate_metrics.py                # Privacy P/R/F1, Leakage, BLEU, ROUGE & BERTScore
├── tests/
│   ├── test_anonymization.py              # End-to-end integration & regression tests
│   └── test_stage2_guardrail.py           # Guardrail & QI detector unit tests
├── app.py                                 # Interactive Streamlit Web UI Dashboard
├── run_evaluation_100.py                  # Reproducible 100-document benchmarking script
├── run_evaluation.py                      # Configurable evaluation CLI
├── requirements.txt                       # Project dependencies
└── README.md                              # Documentation
```

---

## 🚀 Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
The dashboard allows real-time interactive testing with configurable thresholds (`tau`, `floor_sim`, `qi_floor`), live metric gauges, detected QI tables, and diff visualizations.

### 3. Run the Final 100-Document Benchmark
```bash
python run_evaluation_100.py --max_samples 100 --output_dir data
```
*(Supports disk-caching in `data/evaluation_100_stage2_results.json` so completed SLM inferences are never rerun).*

### 4. Run Unit Test Suite
```bash
python -m unittest tests/test_stage2_guardrail.py
python -m unittest tests/test_anonymization.py
```

---

## 📐 Methodological & Research Notes

- **Direct PII Ground Truth:** Evaluated against actual annotations from the `ai4privacy/pii-masking-300k` dataset.
- **Quasi-Identifier Evaluation:** Evaluated via the SLM-independent deterministic QI detection layer (AI4Privacy does not provide QI labels).
- **Provisional Development Defaults:** Thresholds ($\tau = 0.72$, $\text{floor\_sim} = 0.60$, $\text{qi\_floor} = 0.50$) and criteria weights ($w_{\text{sim}}=0.50, w_{\text{qi}}=0.30, w_{\text{read}}=0.20$) are provisional development defaults and are not claimed as scientifically optimal.
- **Privacy Assurance:** The QI abstraction score quantifies coarsening quality and is not a formal mathematical re-identification proof.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
