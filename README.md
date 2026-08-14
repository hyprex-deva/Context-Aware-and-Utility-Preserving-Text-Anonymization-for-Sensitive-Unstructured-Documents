# Context-Aware & Utility-Preserving Text Anonymization for Unstructured Documents

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app.py)

A modular, production-ready NLP architecture for unstructured document anonymization that maximizes both **data privacy** and **downstream semantic utility**.

---

## 🏛️ System Architecture

```
Raw Unstructured Text
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Token-Level Direct PII Detection & Synthetic Surrogates       │
│  ├─ Multi-Pattern Regex (Email, Phone, SSN, IP, Credit Cards)          │
│  ├─ Transformer Token Classifier NER (PER, ORG, LOC, MISC)             │
│  ├─ Span Collision Resolver & Offset Sorter                            │
│  └─ Deterministic Faker Mapper (Document-Level Consistent Cache)       │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼ [Stage 1 Output: Realistic Surrogates]
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Semantic Reasoning & Quasi-Identifier Generalization          │
│  ├─ Local Quantized SLM: Ollama (qwen2.5:1.5b)                        │
│  ├─ Hierarchical Coarsening of Rare Roles, Dates, & Event Combinations │
│  └─ Semantic Drift Guardrail (SentenceTransformers all-MiniLM-L6-v2)  │
│        ├─ Cosine Sim >= 0.80 ➔ ACCEPT Generalized Text                 │
│        └─ Cosine Sim <  0.80 ➔ REJECT & Fallback to Stage 1            │
└────────────────────────────────────────────────────────────────────────┘

       │
       ▼
Final Privacy-Safe & Utility-Preserved Text Output
```

---

## 📂 Project Structure

```text
context-aware-pii-masking/
├── data/
│   ├── benchmark_samples.json      # 10 multi-domain real-world benchmark samples
│   ├── evaluation_results.csv      # Exported benchmark metric comparison table
│   └── evaluation_results.json     # Machine-readable evaluation logs
├── src/
│   ├── __init__.py                 # Core package exports
│   ├── stage1_direct_pii.py        # Stage 1: Regex + Transformer NER + Faker
│   ├── stage2_semantic_defense.py  # Stage 2: SLM Reasoning + MiniLM Drift Guardrail
│   ├── baselines.py                # Microsoft Presidio & Rigid Redaction baselines
│   └── evaluate_metrics.py         # Privacy F1, ROUGE-L, BERTScore & Cosine Sim
├── tests/
│   └── test_anonymization.py       # Unit and integration test suite
├── app.py                          # Modern Streamlit interactive UI dashboard
├── run_evaluation.py               # Benchmark evaluation CLI script
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web App
```bash
streamlit run app.py
```

### 3. Run the Automated Benchmark Suite
```bash
python run_evaluation.py
```

Options for evaluation:
- `--max_samples 10`: Number of benchmark samples to evaluate.
- `--stage2_backend [heuristic|ollama|hf_pipeline]`: Stage 2 SLM inference backend.
- `--skip_heavy_metrics`: Skip neural BERTScore for lightning-fast testing.

---

## 🧪 Unit & Integration Tests

Run the test suite:
```bash
python -m unittest tests/test_anonymization.py
```

---

## 📊 Comparison Matrix

| System / Architecture | Privacy F1 | Semantic Utility | Document Flow | Downstream Task Compatibility |
| :--- | :---: | :---: | :---: | :---: |
| **Rigid Redaction (`[REDACTED]`)** | Medium | Low | Broken Syntax | Poor (Tokens destroyed) |
| **Microsoft Presidio (`<PERSON>`)** | High | Low-Medium | Degraded | Medium (Static tags disrupt language models) |
| **Proposed Stage 1 (Surrogates)** | **High** | **High** | Natural | Excellent |
| **Proposed Two-Stage Architecture** | **Highest (Direct + Quasi)** | **High (Guaranteed $\ge 0.80$)** | **Natural & Coarsened** | **Superior** |
