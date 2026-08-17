"""
Final Research-Grade Evaluation & Benchmarking Pipeline (100 AI4Privacy PII-Masking-300K Documents).

Evaluates 4 comparative systems:
1. Baseline A: Rigid Redaction ([REDACTED])
2. Baseline B: Microsoft Presidio (<TAGS>)
3. Proposed Stage 1: Direct PII + Deterministic Synthetic Surrogates
4. Proposed Two-Stage: Stage 1 Surrogates + Stage 2 Semantic SLM with Three-Way Composite Guardrail

Outputs generated:
- data/evaluation_100_manifest.csv
- data/evaluation_100_dataset.json
- data/evaluation_100_stage2_results.json (Cache & Resumability)
- data/evaluation_100_results.csv
- data/evaluation_100_entity_breakdown.csv
- data/evaluation_100_qi_breakdown.csv
- data/evaluation_100_error_analysis.json
- data/evaluation_100_summary.json
- data/evaluation_100_report.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.stage1_direct_pii import DirectPIIAnonymizer
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense
from src.quasi_identifier_detector import QuasiIdentifierDetector
from src.baselines import BaselinePresidio, BaselineRedacted
from src.evaluate_metrics import (
    compute_privacy_metrics,
    compute_ground_truth_pii_leakage,
    compute_per_entity_pii_metrics,
    compute_rouge_l_scores,
    compute_bleu_scores,
    compute_semantic_similarities,
    compute_bert_scores,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunEvaluation100")


# -----------------------------------------------------------------------------
# 1. Fixed 100-Document Sampling & Manifest Generator
# -----------------------------------------------------------------------------

def select_and_cache_dataset(
    output_dir: str,
    target_count: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select exactly 100 diverse, high-coverage English documents from AI4Privacy PII-Masking-300K
    and persist the fixed dataset snapshot and manifest to disk.
    """
    dataset_file = os.path.join(output_dir, "evaluation_100_dataset.json")
    manifest_file = os.path.join(output_dir, "evaluation_100_manifest.csv")

    # If already cached, load and return
    if os.path.exists(dataset_file) and os.path.exists(manifest_file):
        logger.info(f"Loading cached 100-document dataset from '{dataset_file}'...")
        with open(dataset_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if len(samples) >= target_count:
            return samples[:target_count]

    logger.info("Loading 'ai4privacy/pii-masking-300k' from Hugging Face for stratified selection...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train")
    except Exception as exc:
        raise RuntimeError(f"Failed to load Hugging Face dataset: {exc}")

    random.seed(seed)
    np.random.seed(seed)

    detector = QuasiIdentifierDetector()
    selected_samples: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    # Iterate deterministically through the dataset
    candidates = []
    for idx, row in enumerate(ds):
        if row.get("language") != "English":
            continue

        text = row.get("source_text", "").strip()
        if not text or len(text) < 60:
            continue

        masks = row.get("privacy_mask", [])
        if not masks:
            continue

        direct_pii = [
            {
                "entity": m["value"],
                "type": m["label"],
                "start": m["start"],
                "end": m["end"],
            }
            for m in masks
        ]
        pii_types = sorted(list(set(m["label"] for m in masks)))
        qis_detected = detector.detect_quasi_identifiers(text)
        qi_types = sorted(list(set(q["type"] for q in qis_detected)))

        # Scoring heuristic for diversity: text length + PII variety + detected contextual features
        char_len = len(text)
        word_count = len(text.split())
        score = char_len + (len(pii_types) * 150) + (len(qi_types) * 200)

        candidates.append({
            "id": f"ai4privacy_{row.get('id', idx)}",
            "dataset_index": idx,
            "text": text,
            "target_text": row.get("target_text", ""),
            "direct_pii": direct_pii,
            "pii_count": len(direct_pii),
            "pii_categories": pii_types,
            "qi_categories_detected": qi_types,
            "char_length": char_len,
            "word_count": word_count,
            "diversity_score": score,
        })

    # Sort deterministically by diversity score and take top stratified distribution
    candidates.sort(key=lambda x: x["diversity_score"], reverse=True)
    
    # Select target_count samples
    selected_samples = candidates[:target_count]

    # Build manifest
    for rank, s in enumerate(selected_samples, start=1):
        manifest_rows.append({
            "selection_rank": rank,
            "sample_id": s["id"],
            "dataset_index": s["dataset_index"],
            "char_length": s["char_length"],
            "word_count": s["word_count"],
            "pii_entity_count": s["pii_count"],
            "pii_categories": "; ".join(s["pii_categories"]),
            "detected_qi_categories": "; ".join(s["qi_categories_detected"]) if s["qi_categories_detected"] else "None",
            "selection_rationale": "High-coverage multi-PII contextual document",
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=2)

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_file, index=False)
    logger.info(f"Saved {len(selected_samples)} selected samples to '{dataset_file}' and manifest to '{manifest_file}'.")

    return selected_samples


# -----------------------------------------------------------------------------
# 2. Stage 2 Inferences with Caching & Resumability
# -----------------------------------------------------------------------------

def run_or_load_stage2_inferences(
    samples: List[Dict[str, Any]],
    stage1_results: List[Dict[str, Any]],
    stage2_defense: SemanticQuasiIdentifierDefense,
    cache_path: str,
    tau: float = 0.72,
    floor_sim: float = 0.60,
    qi_floor: float = 0.50,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Stage 2 SLM inferences with incremental disk caching and resumability.
    """
    cached_results: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_results = json.load(f)
            logger.info(f"Loaded {len(cached_results)} existing Stage 2 results from cache: '{cache_path}'.")
        except Exception as exc:
            logger.warning(f"Could not read cache file ({exc}). Starting fresh.")
            cached_results = {}

    for idx, sample in enumerate(tqdm(samples, desc="Executing Stage 2 SLM Inference")):
        sid = sample["id"]
        s1_res = stage1_results[idx]
        s1_text = s1_res["sanitized_text"]
        orig_text = sample["text"]

        # Check if already cached
        if sid in cached_results:
            continue

        t0 = time.time()
        s2_res = stage2_defense.generalize_text(
            stage1_text=s1_text,
            original_text=orig_text,
            tau=tau,
            floor_sim=floor_sim,
            qi_floor=qi_floor,
        )
        elapsed = round(time.time() - t0, 3)

        # Store rich telemetry
        cached_results[sid] = {
            "sample_id": sid,
            "original_text": orig_text,
            "stage1_text": s1_text,
            "candidate_text": s2_res.get("candidate_text", s1_text),
            "final_text": s2_res.get("final_text", s1_text),
            "is_accepted": s2_res.get("is_accepted", True),
            "fallback_triggered": s2_res.get("fallback_triggered", False),
            "composite_score": s2_res.get("composite_score", 0.0),
            "metrics_breakdown": s2_res.get("metrics_breakdown", {}),
            "thresholds": s2_res.get("thresholds", {}),
            "qi_analysis": s2_res.get("qi_analysis", []),
            "modifications": s2_res.get("modifications", []),
            "inference_time_sec": elapsed,
            "backend_used": s2_res.get("backend_used", "Ollama qwen2.5:1.5b"),
        }

        # Incremental write to disk for immediate crash resilience
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached_results, f, indent=2)

    return cached_results


# -----------------------------------------------------------------------------
# 3. Main End-to-End Evaluation Engine
# -----------------------------------------------------------------------------

def run_evaluation_pipeline(
    samples: List[Dict[str, Any]],
    output_dir: str = "data",
    ollama_model: str = "qwen2.5:1.5b",
    tau: float = 0.72,
    floor_sim: float = 0.60,
    qi_floor: float = 0.50,
    compute_heavy_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Execute full 4-system comparative evaluation across the 100 fixed documents.
    """
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    logger.info(f"Initializing Evaluation Pipeline on {len(samples)} benchmark documents...")
    stage1 = DirectPIIAnonymizer(use_ner=True)
    stage2 = SemanticQuasiIdentifierDefense(
        ollama_model=ollama_model,
        tau=tau,
        floor_sim=floor_sim,
        qi_floor=qi_floor,
        request_timeout=90,
    )
    presidio_base = BaselinePresidio()
    redacted_base = BaselineRedacted(stage1_anonymizer=stage1)
    qi_detector = QuasiIdentifierDetector()

    original_texts = [s["text"] for s in samples]
    ground_truth_entities = [s.get("direct_pii", []) for s in samples]

    # Execution data dictionaries
    systems_data = {
        "Rigid Redaction": {"sanitized": [], "entities": [], "time": 0.0},
        "Microsoft Presidio": {"sanitized": [], "entities": [], "time": 0.0},
        "Proposed: Stage 1 (Surrogates)": {"sanitized": [], "entities": [], "time": 0.0},
        "Proposed: Two-Stage (Stage 1 + Stage 2)": {
            "sanitized": [], "entities": [], "time": 0.0, "stage2_details": []
        },
    }

    # 1. System A: Rigid Redaction
    t0 = time.time()
    for text in tqdm(original_texts, desc="Evaluating Rigid Redaction"):
        res = redacted_base.anonymize(text)
        systems_data["Rigid Redaction"]["sanitized"].append(res["sanitized_text"])
        systems_data["Rigid Redaction"]["entities"].append(res["detected_entities"])
    systems_data["Rigid Redaction"]["time"] = time.time() - t0

    # 2. System B: Microsoft Presidio
    t0 = time.time()
    for text in tqdm(original_texts, desc="Evaluating Microsoft Presidio"):
        res = presidio_base.anonymize(text)
        systems_data["Microsoft Presidio"]["sanitized"].append(res["sanitized_text"])
        systems_data["Microsoft Presidio"]["entities"].append(res["detected_entities"])
    systems_data["Microsoft Presidio"]["time"] = time.time() - t0

    # 3. System C: Proposed Stage 1
    t0 = time.time()
    s1_results = []
    for text in tqdm(original_texts, desc="Evaluating Proposed Stage 1"):
        res = stage1.anonymize(text)
        s1_results.append(res)
        systems_data["Proposed: Stage 1 (Surrogates)"]["sanitized"].append(res["sanitized_text"])
        systems_data["Proposed: Stage 1 (Surrogates)"]["entities"].append(res["detected_entities"])
    systems_data["Proposed: Stage 1 (Surrogates)"]["time"] = time.time() - t0

    # 4. System D: Proposed Two-Stage (with Caching)
    cache_path = os.path.join(output_dir, "evaluation_100_stage2_results.json")
    stage2_cache = run_or_load_stage2_inferences(
        samples=samples,
        stage1_results=s1_results,
        stage2_defense=stage2,
        cache_path=cache_path,
        tau=tau,
        floor_sim=floor_sim,
        qi_floor=qi_floor,
    )

    t0 = time.time()
    for idx, s in enumerate(samples):
        sid = s["id"]
        cached_entry = stage2_cache[sid]
        systems_data["Proposed: Two-Stage (Stage 1 + Stage 2)"]["sanitized"].append(cached_entry["final_text"])
        systems_data["Proposed: Two-Stage (Stage 1 + Stage 2)"]["entities"].append(s1_results[idx]["detected_entities"])
        systems_data["Proposed: Two-Stage (Stage 1 + Stage 2)"]["stage2_details"].append(cached_entry)
    systems_data["Proposed: Two-Stage (Stage 1 + Stage 2)"]["time"] = time.time() - t0

    # -------------------------------------------------------------------------
    # 4. Metrics Computation across all Systems
    # -------------------------------------------------------------------------
    logger.info("Computing PII, Residual Leakage, QI Abstraction, and Utility metrics...")

    results_table_rows = []
    system_summaries = {}

    for sys_name, data in systems_data.items():
        sanitized_texts = data["sanitized"]
        pred_entities = data["entities"]

        # Direct PII Metrics (Exact & Relaxed)
        p_relaxed = compute_privacy_metrics(pred_entities, ground_truth_entities, match_mode="relaxed")
        p_exact = compute_privacy_metrics(pred_entities, ground_truth_entities, match_mode="exact")
        leakage_res = compute_ground_truth_pii_leakage(sanitized_texts, ground_truth_entities)

        # Utility Metrics
        rouge_res = compute_rouge_l_scores(original_texts, sanitized_texts)
        bleu_res = compute_bleu_scores(original_texts, sanitized_texts)
        sim_res = compute_semantic_similarities(original_texts, sanitized_texts)
        bert_res = compute_bert_scores(original_texts, sanitized_texts) if compute_heavy_metrics else {"bertscore_f1": sim_res["cosine_similarity"]}

        # Readability Metric
        readability_scores = [
            stage2.compute_readability_score(orig, sanit)
            for orig, sanit in zip(original_texts, sanitized_texts)
        ]
        avg_readability = float(np.mean(readability_scores)) if readability_scores else 1.0

        row = {
            "System": sys_name,
            "PII Precision (Relaxed)": p_relaxed["precision"],
            "PII Recall (Relaxed)": p_relaxed["recall"],
            "PII F1 (Relaxed)": p_relaxed["f1"],
            "PII Precision (Exact)": p_exact["precision"],
            "PII Recall (Exact)": p_exact["recall"],
            "PII F1 (Exact)": p_exact["f1"],
            "PII Leakage Rate": leakage_res["pii_leakage_rate"],
            "PII Removal Rate": leakage_res["pii_removal_rate"],
            "ROUGE-L F1": rouge_res["rougeL_f1"],
            "BLEU-4": bleu_res["bleu_score"],
            "Semantic Cosine Sim": sim_res["cosine_similarity"],
            "BERTScore F1": bert_res["bertscore_f1"],
            "Readability Score": round(avg_readability, 4),
            "Avg Latency (s)": round(data["time"] / max(len(samples), 1), 3),
        }

        # Stage 2 specific QI metrics
        if "Two-Stage" in sys_name:
            stage2_details = data["stage2_details"]
            all_qi_evals = []
            for d in stage2_details:
                all_qi_evals.extend(d.get("qi_analysis", []))

            total_qis = len(all_qi_evals)
            mitigated_count = sum(1 for q in all_qi_evals if q.get("status") == "mitigated")
            partial_count = sum(1 for q in all_qi_evals if q.get("status") == "partial")
            exposed_count = sum(1 for q in all_qi_evals if q.get("status") == "exposed")

            mit_rate = (mitigated_count + partial_count) / max(total_qis, 1)
            exp_rate = exposed_count / max(total_qis, 1)

            avg_qi_scores = [d.get("metrics_breakdown", {}).get("qi_abstraction_score", 1.0) for d in stage2_details]
            avg_qi_score = float(np.mean(avg_qi_scores)) if avg_qi_scores else 1.0

            fallback_count = sum(1 for d in stage2_details if d.get("fallback_triggered", False))
            fallback_rate = fallback_count / max(len(samples), 1)

            row.update({
                "QI Abstraction Score": round(avg_qi_score, 4),
                "QI Mitigation Rate": round(mit_rate, 4),
                "QI Exposure Rate": round(exp_rate, 4),
                "Fallback Rate": round(fallback_rate, 4),
            })
        else:
            row.update({
                "QI Abstraction Score": "N/A",
                "QI Mitigation Rate": "N/A",
                "QI Exposure Rate": "N/A",
                "Fallback Rate": "N/A",
            })

        results_table_rows.append(row)
        system_summaries[sys_name] = row

    df_results = pd.DataFrame(results_table_rows)
    df_results.to_csv(os.path.join(output_dir, "evaluation_100_results.csv"), index=False)

    # -------------------------------------------------------------------------
    # 5. Per-Entity PII Breakdown
    # -------------------------------------------------------------------------
    entity_breakdown_rows = compute_per_entity_pii_metrics(
        systems_data["Proposed: Stage 1 (Surrogates)"]["entities"],
        ground_truth_entities,
        match_mode="relaxed",
    )
    df_entity = pd.DataFrame(entity_breakdown_rows)
    df_entity.to_csv(os.path.join(output_dir, "evaluation_100_entity_breakdown.csv"), index=False)

    # -------------------------------------------------------------------------
    # 6. Per-QI-Type Breakdown (Stage 2)
    # -------------------------------------------------------------------------
    stage2_details = systems_data["Proposed: Two-Stage (Stage 1 + Stage 2)"]["stage2_details"]
    qi_type_stats: Dict[str, Dict[str, Any]] = {}

    for d in stage2_details:
        for q in d.get("qi_analysis", []):
            q_type = q.get("type", "UNKNOWN")
            if q_type not in qi_type_stats:
                qi_type_stats[q_type] = {
                    "count": 0,
                    "init_risks": [],
                    "resid_risks": [],
                    "mitigated": 0,
                    "partial": 0,
                    "exposed": 0,
                }
            qi_type_stats[q_type]["count"] += 1
            qi_type_stats[q_type]["init_risks"].append(q.get("initial_risk", 1.0))
            qi_type_stats[q_type]["resid_risks"].append(q.get("residual_risk", 0.0))
            status = q.get("status", "mitigated")
            if status == "mitigated":
                qi_type_stats[q_type]["mitigated"] += 1
            elif status == "partial":
                qi_type_stats[q_type]["partial"] += 1
            else:
                qi_type_stats[q_type]["exposed"] += 1

    qi_breakdown_rows = []
    for q_type, stats in sorted(qi_type_stats.items()):
        cnt = stats["count"]
        mit = stats["mitigated"]
        part = stats["partial"]
        exp = stats["exposed"]
        avg_init = float(np.mean(stats["init_risks"])) if stats["init_risks"] else 1.0
        avg_resid = float(np.mean(stats["resid_risks"])) if stats["resid_risks"] else 0.0
        mit_rate = (mit + part) / max(cnt, 1)
        exp_rate = exp / max(cnt, 1)

        qi_breakdown_rows.append({
            "QI Type": q_type,
            "Count": cnt,
            "Average Initial Risk": round(avg_init, 4),
            "Average Residual Risk": round(avg_resid, 4),
            "Mitigated Count": mit,
            "Partial Count": part,
            "Exposed Count": exp,
            "Mitigation Rate": round(mit_rate, 4),
            "Exposure Rate": round(exp_rate, 4),
        })

    df_qi = pd.DataFrame(qi_breakdown_rows)
    df_qi.to_csv(os.path.join(output_dir, "evaluation_100_qi_breakdown.csv"), index=False)

    # -------------------------------------------------------------------------
    # 7. Guardrail Telemetry & Ablation (Reusing Cached Rewrites)
    # -------------------------------------------------------------------------
    total_candidates = len(stage2_details)
    accepted_count = sum(1 for d in stage2_details if d.get("is_accepted", True))
    rejected_count = total_candidates - accepted_count
    fallback_rate = rejected_count / max(total_candidates, 1)

    # Rejection attribution
    reason_counts = {"tau_failure": 0, "semantic_floor_failure": 0, "qi_floor_failure": 0, "multiple_failures": 0}
    for d in stage2_details:
        if not d.get("is_accepted", True):
            bd = d.get("metrics_breakdown", {})
            s_comp = d.get("composite_score", 0.0)
            s_sim = bd.get("semantic_similarity", 0.0)
            s_qi = bd.get("qi_abstraction_score", 1.0)
            fails = []
            if s_comp < tau:
                fails.append("tau")
            if s_sim < floor_sim:
                fails.append("sim")
            if s_qi < qi_floor:
                fails.append("qi")

            if len(fails) > 1:
                reason_counts["multiple_failures"] += 1
            elif "sim" in fails:
                reason_counts["semantic_floor_failure"] += 1
            elif "qi" in fails:
                reason_counts["qi_floor_failure"] += 1
            elif "tau" in fails:
                reason_counts["tau_failure"] += 1

    # Ablation Evaluation on Cached Candidate Rewrites
    ablation_single_thresh_accepts = 0
    for d in stage2_details:
        sim_val = d.get("metrics_breakdown", {}).get("semantic_similarity", 0.0)
        if sim_val >= 0.80:
            ablation_single_thresh_accepts += 1

    ablation_summary = {
        "Approach A: Single Cosine Threshold (S_sim >= 0.80)": {
            "accepted_count": ablation_single_thresh_accepts,
            "acceptance_rate": round(ablation_single_thresh_accepts / max(total_candidates, 1), 4),
            "fallback_rate": round((total_candidates - ablation_single_thresh_accepts) / max(total_candidates, 1), 4),
        },
        "Approach B: Three-Way Composite Guardrail (Current)": {
            "accepted_count": accepted_count,
            "acceptance_rate": round(accepted_count / max(total_candidates, 1), 4),
            "fallback_rate": round(fallback_rate, 4),
        },
    }

    # -------------------------------------------------------------------------
    # 8. Qualitative Error Analysis Exemplars
    # -------------------------------------------------------------------------
    successful_cases = []
    failure_cases = []
    partial_cases = []
    exposed_cases = []
    low_sim_cases = []
    qi_floor_cases = []

    for d in stage2_details:
        entry = {
            "sample_id": d["sample_id"],
            "original_text": d["original_text"][:200] + "...",
            "stage1_text": d["stage1_text"][:200] + "...",
            "candidate_text": d["candidate_text"][:200] + "...",
            "final_text": d["final_text"][:200] + "...",
            "composite_score": d["composite_score"],
            "metrics": d["metrics_breakdown"],
            "qi_analysis": d.get("qi_analysis", []),
        }
        if d.get("is_accepted", True):
            if len(successful_cases) < 10:
                successful_cases.append(entry)
        else:
            if len(failure_cases) < 10:
                failure_cases.append(entry)

        for q in d.get("qi_analysis", []):
            if q.get("status") == "partial" and len(partial_cases) < 5:
                partial_cases.append({"sample_id": d["sample_id"], "qi": q, "candidate": d["candidate_text"][:150]})
            elif q.get("status") == "exposed" and len(exposed_cases) < 5:
                exposed_cases.append({"sample_id": d["sample_id"], "qi": q, "candidate": d["candidate_text"][:150]})

        if d.get("metrics_breakdown", {}).get("semantic_similarity", 1.0) < floor_sim and len(low_sim_cases) < 5:
            low_sim_cases.append(entry)
        if d.get("metrics_breakdown", {}).get("qi_abstraction_score", 1.0) < qi_floor and len(qi_floor_cases) < 5:
            qi_floor_cases.append(entry)

    error_analysis_data = {
        "successful_stage2_generalizations": successful_cases,
        "rejected_stage2_fallbacks": failure_cases,
        "partial_qi_mitigation_examples": partial_cases,
        "exposed_qi_examples": exposed_cases,
        "low_semantic_similarity_examples": low_sim_cases,
        "qi_floor_rejection_examples": qi_floor_cases,
    }

    with open(os.path.join(output_dir, "evaluation_100_error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)

    # -------------------------------------------------------------------------
    # 9. Summary & Human-Readable Report Export
    # -------------------------------------------------------------------------
    total_elapsed = round(time.time() - t_start, 2)
    summary_data = {
        "metadata": {
            "evaluation_title": "Context-Aware & Utility-Preserving Text Anonymization Benchmark",
            "dataset": "ai4privacy/pii-masking-300k (English subset)",
            "sample_count": len(samples),
            "random_seed": 42,
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "provisional_parameters": {
                "tau": tau,
                "floor_sim": floor_sim,
                "qi_floor": qi_floor,
                "weights": {"sim": 0.50, "qi": 0.30, "read": 0.20},
            },
            "total_elapsed_seconds": total_elapsed,
        },
        "system_results": system_summaries,
        "guardrail_telemetry": {
            "total_candidates": total_candidates,
            "accepted_candidates": accepted_count,
            "rejected_candidates": rejected_count,
            "fallback_rate": round(fallback_rate, 4),
            "rejection_cause_distribution": reason_counts,
            "ablation_comparison": ablation_summary,
        },
    }

    with open(os.path.join(output_dir, "evaluation_100_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Formatted text report
    report_lines = [
        "=" * 105,
        "  RESEARCH BENCHMARK REPORT: CONTEXT-AWARE & UTILITY-PRESERVING TEXT ANONYMIZATION",
        "=" * 105,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Dataset: AI4Privacy PII-Masking-300K | Samples: {len(samples)}",
        f"Stage 2 SLM: {ollama_model} | Embedder: all-MiniLM-L6-v2 | Total Runtime: {total_elapsed:.2f}s",
        "-" * 105,
        "\n1. SYSTEM COMPARISON MATRIX\n",
        df_results.to_string(index=False),
        "\n" + "-" * 105,
        "\n2. GUARDRAIL TELEMETRY & ABLATION STUDY\n",
        f"Total Evaluated Candidates : {total_candidates}",
        f"Accepted Candidates        : {accepted_count} ({accepted_count/max(total_candidates,1)*100:.1f}%)",
        f"Fallback Triggered         : {rejected_count} ({fallback_rate*100:.1f}%)",
        f"Rejection Failure Breakdown: {reason_counts}",
        "\nAblation Comparison on Cached Candidates:",
        f"- Original Single Threshold (S_sim >= 0.80) : Acceptance Rate = {ablation_summary['Approach A: Single Cosine Threshold (S_sim >= 0.80)']['acceptance_rate']*100:.1f}%",
        f"- Proposed Composite Guardrail (3-Way Floor): Acceptance Rate = {ablation_summary['Approach B: Three-Way Composite Guardrail (Current)']['acceptance_rate']*100:.1f}%",
        "\n" + "-" * 105,
        "\n3. DETECTED QUASI-IDENTIFIER (QI) ABSTRACTION BREAKDOWN\n",
        df_qi.to_string(index=False) if not df_qi.empty else "No quasi-identifiers detected.",
        "\n" + "-" * 105,
        "\n4. RESEARCH INTEGRITY & METHODOLOGICAL NOTES\n",
        "- Direct PII Evaluation: Computed against AI4Privacy ground truth annotations.",
        "- QI Evaluation: Computed via SLM-independent deterministic QI detection layer (no QI ground truth in AI4Privacy).",
        "- Privacy Metric Note: QI Abstraction Score reflects candidate coarsening quality and is not a formal re-identification proof.",
        "=" * 105,
    ]

    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, "evaluation_100_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text + "\n")
    return summary_data


# -----------------------------------------------------------------------------
# 4. CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Final 100-Document Evaluation Pipeline")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save evaluation artifacts and reports (default: 'data')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (e.g. 5 for dry-run, 100 for full benchmark)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="qwen2.5:1.5b",
        help="Ollama model tag for Stage 2 (default: 'qwen2.5:1.5b')",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.72,
        help="Provisional composite threshold tau (default: 0.72)",
    )
    parser.add_argument(
        "--floor_sim",
        type=float,
        default=0.60,
        help="Provisional hard semantic similarity floor (default: 0.60)",
    )
    parser.add_argument(
        "--qi_floor",
        type=float,
        default=0.50,
        help="Provisional hard QI abstraction safety floor (default: 0.50)",
    )
    parser.add_argument(
        "--skip_heavy_metrics",
        action="store_true",
        help="Skip heavy BERTScore neural download",
    )

    args = parser.parse_args()

    # Step 1: Select or load dataset snapshot
    all_100_samples = select_and_cache_dataset(
        output_dir=args.output_dir,
        target_count=100,
        seed=args.seed,
    )

    # Take slice for execution
    active_samples = all_100_samples[:args.max_samples]
    logger.info(f"Loaded {len(active_samples)} active samples for evaluation run.")

    # Step 2: Run complete 4-system evaluation
    run_evaluation_pipeline(
        samples=active_samples,
        output_dir=args.output_dir,
        ollama_model=args.ollama_model,
        tau=args.tau,
        floor_sim=args.floor_sim,
        qi_floor=args.qi_floor,
        compute_heavy_metrics=not args.skip_heavy_metrics,
    )


if __name__ == "__main__":
    main()
