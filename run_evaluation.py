"""
Benchmark Evaluation Runner for Context-Aware Text Anonymization.

Evaluates and compares 4 approaches:
1. Baseline A: Microsoft Presidio (<PERSON>, <EMAIL_ADDRESS>, etc.)
2. Baseline B: Rigid Redaction ([REDACTED])
3. Proposed Stage 1: Direct PII + Synthetic Surrogates
4. Proposed Two-Stage: Stage 1 Surrogates + Stage 2 Semantic Quasi-Identifier Generalization

Outputs a formatted Pandas comparison table and exports CSV/JSON evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage1_direct_pii import DirectPIIAnonymizer
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense
from src.baselines import BaselinePresidio, BaselineRedacted
from src.evaluate_metrics import evaluate_anonymization_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunEvaluation")


def load_benchmark_data(data_path: str, max_samples: int = 10) -> List[Dict[str, Any]]:
    """Load benchmark dataset from local JSON or Hugging Face dataset fallback."""
    if os.path.exists(data_path):
        logger.info(f"Loading local benchmark samples from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return samples[:max_samples]
    
    logger.info(f"Local file {data_path} not found. Attempting to load Hugging Face dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "id": f"hf_sample_{i}",
                "domain": "General Web",
                "text": item.get("unmasked_text", item.get("text", "")),
                "direct_pii": [],
                "quasi_identifiers": [],
            })
        return samples
    except Exception as exc:
        logger.warning(f"Could not load Hugging Face dataset ({exc}). Generating default dummy test samples.")
        return [
            {
                "id": "dummy_01",
                "domain": "Clinical",
                "text": "Dr. Sarah Connor treated patient John Doe at General Hospital on 2023-01-10.",
                "direct_pii": [
                    {"entity": "Sarah Connor", "type": "PER", "start": 4, "end": 16},
                    {"entity": "John Doe", "type": "PER", "start": 33, "end": 41},
                    {"entity": "General Hospital", "type": "ORG", "start": 45, "end": 61},
                ],
                "quasi_identifiers": ["treated patient on 2023-01-10"]
            }
        ]


def run_benchmark(
    samples: List[Dict[str, Any]],
    output_dir: str = "data",
    backend_stage2: str = "heuristic",
    compute_heavy_metrics: bool = True,
) -> pd.DataFrame:
    """
    Execute end-to-end evaluation across all models and baseline systems.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Benchmark Evaluation on {len(samples)} multi-domain test samples...")

    # Initialize models
    logger.info("Initializing models...")
    stage1 = DirectPIIAnonymizer(use_ner=True)
    stage2 = SemanticQuasiIdentifierDefense(backend=backend_stage2)
    presidio_base = BaselinePresidio()
    redacted_base = BaselineRedacted(stage1_anonymizer=stage1)

    original_texts = [s["text"] for s in samples]
    ground_truth_entities = [s.get("direct_pii", []) for s in samples]

    # Model inference outputs
    outputs = {
        "Baseline: Rigid Redaction ([REDACTED])": {"sanitized": [], "entities": [], "time": 0.0},
        "Baseline: Microsoft Presidio (<TAGS>)": {"sanitized": [], "entities": [], "time": 0.0},
        "Proposed: Stage 1 (Direct PII + Surrogates)": {"sanitized": [], "entities": [], "time": 0.0},
        "Proposed: Two-Stage Architecture (Surrogates + Semantic SLM)": {
            "sanitized": [], "entities": [], "drift_passes": 0, "time": 0.0
        },
    }

    # 1. Baseline: Rigid Redaction
    t0 = time.time()
    for text in tqdm(original_texts, desc="Evaluating Rigid Redaction"):
        res = redacted_base.anonymize(text)
        outputs["Baseline: Rigid Redaction ([REDACTED])"]["sanitized"].append(res["sanitized_text"])
        outputs["Baseline: Rigid Redaction ([REDACTED])"]["entities"].append(res["detected_entities"])
    outputs["Baseline: Rigid Redaction ([REDACTED])"]["time"] = time.time() - t0

    # 2. Baseline: Microsoft Presidio
    t0 = time.time()
    for text in tqdm(original_texts, desc="Evaluating Microsoft Presidio"):
        res = presidio_base.anonymize(text)
        outputs["Baseline: Microsoft Presidio (<TAGS>)"]["sanitized"].append(res["sanitized_text"])
        outputs["Baseline: Microsoft Presidio (<TAGS>)"]["entities"].append(res["detected_entities"])
    outputs["Baseline: Microsoft Presidio (<TAGS>)"]["time"] = time.time() - t0

    # 3. Proposed Stage 1
    t0 = time.time()
    s1_results = []
    for text in tqdm(original_texts, desc="Evaluating Proposed Stage 1"):
        res = stage1.anonymize(text)
        s1_results.append(res)
        outputs["Proposed: Stage 1 (Direct PII + Surrogates)"]["sanitized"].append(res["sanitized_text"])
        outputs["Proposed: Stage 1 (Direct PII + Surrogates)"]["entities"].append(res["detected_entities"])
    outputs["Proposed: Stage 1 (Direct PII + Surrogates)"]["time"] = time.time() - t0

    # 4. Proposed Two-Stage
    t0 = time.time()
    drift_pass_count = 0
    for idx, s1_res in enumerate(tqdm(s1_results, desc="Evaluating Proposed Two-Stage")):
        res = stage2.generalize_text(
            stage1_text=s1_res["sanitized_text"], original_text=original_texts[idx]
        )
        if res["drift_passed"]:
            drift_pass_count += 1
        outputs["Proposed: Two-Stage Architecture (Surrogates + Semantic SLM)"]["sanitized"].append(
            res["final_text"]
        )
        outputs["Proposed: Two-Stage Architecture (Surrogates + Semantic SLM)"]["entities"].append(
            s1_res["detected_entities"]
        )
    outputs["Proposed: Two-Stage Architecture (Surrogates + Semantic SLM)"]["time"] = time.time() - t0
    outputs["Proposed: Two-Stage Architecture (Surrogates + Semantic SLM)"]["drift_passes"] = drift_pass_count

    # Compute Evaluation Metrics
    logger.info("Computing Privacy, Utility (ROUGE-L, Cosine Sim, BERTScore) metrics...")
    summary_rows = []

    for system_name, data in outputs.items():
        metrics = evaluate_anonymization_system(
            original_texts=original_texts,
            sanitized_texts=data["sanitized"],
            predicted_entities=data["entities"],
            ground_truth_entities=ground_truth_entities,
            compute_heavy_metrics=compute_heavy_metrics,
        )

        row = {
            "Method / Architecture": system_name,
            "Privacy Precision": metrics.get("privacy_precision", 1.0),
            "Privacy Recall": metrics.get("privacy_recall", 1.0),
            "Privacy F1": metrics.get("privacy_f1", 1.0),
            "ROUGE-L F1": metrics.get("rougeL_f1", 0.0),
            "Semantic Cosine Sim": metrics.get("cosine_similarity", 0.0),
            "BERTScore F1": metrics.get("bertscore_f1", metrics.get("cosine_similarity", 0.0)),
            "Latency (sec)": round(data["time"], 3),
        }
        summary_rows.append(row)

    df_results = pd.DataFrame(summary_rows)

    # Save to disk
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    json_path = os.path.join(output_dir, "evaluation_results.json")
    df_results.to_csv(csv_path, index=False)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    logger.info(f"Evaluation results successfully saved to {csv_path} and {json_path}")
    return df_results


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print formatted markdown comparison table to terminal."""
    header = "\n" + "=" * 105 + "\n"
    header += "  CONTEXT-AWARE & UTILITY-PRESERVING TEXT ANONYMIZATION: BENCHMARK RESULTS\n"
    header += "=" * 105 + "\n"
    print(header)
    print(df.to_string(index=False))
    print("\n" + "=" * 105 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Text Anonymization Benchmark Evaluation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/benchmark_samples.json",
        help="Path to benchmark JSON samples",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--stage2_backend",
        type=str,
        default="heuristic",
        choices=["heuristic", "ollama", "hf_pipeline"],
        help="Stage 2 SLM backend to use",
    )
    parser.add_argument(
        "--skip_heavy_metrics",
        action="store_true",
        help="Skip heavy neural BERTScore downloads for fast local testing",
    )

    args = parser.parse_args()
    samples = load_benchmark_data(args.data_path, max_samples=args.max_samples)
    df_results = run_benchmark(
        samples=samples,
        output_dir=args.output_dir,
        backend_stage2=args.stage2_backend,
        compute_heavy_metrics=not args.skip_heavy_metrics,
    )
    print_comparison_table(df_results)


if __name__ == "__main__":
    main()
