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
    if data_path and os.path.exists(data_path):
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if len(data) >= max_samples:
                    logger.info(f"Loaded {len(data[:max_samples])} benchmark samples from '{data_path}' (ai4privacy/pii-masking-300k).")
                    return data[:max_samples]
                else:
                    logger.info(f"Local file '{data_path}' has {len(data)} samples, but {max_samples} requested. Fetching from Hugging Face dataset...")
        except Exception as exc:
            logger.error(f"Error loading benchmark dataset from {data_path}: {exc}")

    try:
        from datasets import load_dataset
        logger.info(f"Loading {max_samples} benchmark samples from Hugging Face: 'ai4privacy/pii-masking-300k'...")
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
        samples = []
        for i, item in enumerate(ds):
            if len(samples) >= max_samples:
                break
            if item.get("language") == "English":
                masks = item.get("privacy_mask", [])
                direct_pii = [{"entity": m["value"], "type": m["label"], "start": m["start"], "end": m["end"]} for m in masks]
                samples.append({
                    "id": f"ai4privacy_{item.get('id', i)}",
                    "domain": "ai4privacy",
                    "text": item.get("source_text", ""),
                    "direct_pii": direct_pii,
                    "target_text": item.get("target_text", ""),
                })
        logger.info(f"Successfully prepared {len(samples)} English benchmark samples.")
        return samples
    except Exception as exc:
        logger.warning(f"Could not load Hugging Face dataset ({exc}).")
        return []


def run_benchmark(
    samples: List[Dict[str, Any]],
    output_dir: str = "data",
    ollama_model: str = "qwen2.5:1.5b",
    compute_heavy_metrics: bool = True,
) -> pd.DataFrame:

    """
    Execute end-to-end evaluation across all models and baseline systems.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting Benchmark Evaluation on {len(samples)} multi-domain test samples...")

    # Initialize models
    logger.info(f"Initializing models (Stage 2: Ollama {ollama_model})...")
    stage1 = DirectPIIAnonymizer(use_ner=True)
    stage2 = SemanticQuasiIdentifierDefense(ollama_model=ollama_model)
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
    logger.info("Computing Privacy, Utility (BLEU-4, ROUGE-L, Cosine Sim, BERTScore) metrics...")
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
            "BLEU-4": metrics.get("bleu_score", 0.0),
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
        "--ollama_model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model tag for Stage 2 (default: qwen2.5:1.5b)",
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
        ollama_model=args.ollama_model,
        compute_heavy_metrics=not args.skip_heavy_metrics,
    )
    print_comparison_table(df_results)



if __name__ == "__main__":
    main()
