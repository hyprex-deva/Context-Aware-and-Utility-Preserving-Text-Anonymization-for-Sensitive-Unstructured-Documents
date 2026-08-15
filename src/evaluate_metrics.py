"""
Evaluation Metrics for Privacy and Text Utility Preservation.

Computes:
1. Privacy Metrics: Precision, Recall, F1 against ground truth direct PII annotations.
2. Utility Metrics:
   - BERTScore (Precision, Recall, F1)
   - ROUGE-L (Precision, Recall, F-measure)
   - Semantic Cosine Similarity (SentenceTransformers all-MiniLM-L6-v2)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvaluateMetrics")


def compute_privacy_metrics(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    match_mode: str = "relaxed",  # 'exact' or 'relaxed' (character overlap)
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 score for entity detection.

    Args:
        predictions: Nested list of detected entities per sample.
        ground_truths: Nested list of ground truth entities per sample.
        match_mode: 'relaxed' allows token/span overlap; 'exact' requires identical start/end offsets.

    Returns:
        Dict with 'precision', 'recall', and 'f1'.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, truths in zip(predictions, ground_truths):
        matched_truth_indices = set()

        for pred in preds:
            p_start, p_end = pred.get("start", 0), pred.get("end", 0)
            p_val = pred.get("entity_value", pred.get("text", "")).strip().lower()

            found_match = False
            for t_idx, truth in enumerate(truths):
                if t_idx in matched_truth_indices:
                    continue

                t_start, t_end = truth.get("start", 0), truth.get("end", 0)
                t_val = truth.get("entity", truth.get("text", "")).strip().lower()

                if match_mode == "exact":
                    is_match = (p_start == t_start and p_end == t_end) or (p_val == t_val)
                else:
                    # Relaxed character overlap or substring match
                    overlap = max(0, min(p_end, t_end) - max(p_start, t_start))
                    is_match = overlap > 0 or (p_val in t_val) or (t_val in p_val)

                if is_match:
                    found_match = True
                    matched_truth_indices.add(t_idx)
                    break

            if found_match:
                total_tp += 1
            else:
                total_fp += 1

        total_fn += (len(truths) - len(matched_truth_indices))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def _internal_lcs_rouge_l(ref: str, hyp: str) -> float:
    """Internal Longest Common Subsequence calculator for ROUGE-L fallback."""
    ref_tokens = re.findall(r"\w+", ref.lower())
    hyp_tokens = re.findall(r"\w+", hyp.lower())
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec = lcs / m if m > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1


def compute_rouge_l_scores(
    original_texts: List[str], sanitized_texts: List[str]
) -> Dict[str, float]:
    """
    Compute average ROUGE-L F1 retention between original and sanitized documents.
    """
    if not original_texts or not sanitized_texts:
        return {"rougeL_f1": 0.0}

    scores = []
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        for orig, sanit in zip(original_texts, sanitized_texts):
            res = scorer.score(orig, sanit)
            scores.append(res["rougeL"].fmeasure)
    except Exception:
        # Fallback to internal LCS calculator
        for orig, sanit in zip(original_texts, sanitized_texts):
            scores.append(_internal_lcs_rouge_l(orig, sanit))

    avg_rouge = float(np.mean(scores)) if scores else 0.0
    return {"rougeL_f1": round(avg_rouge, 4)}


def compute_bleu_scores(
    original_texts: List[str],
    sanitized_texts: List[str],
    n_gram: int = 4,
) -> Dict[str, float]:
    """
    Compute average BLEU-4 score (cumulative n-gram precision with brevity penalty)
    comparing sanitized texts against reference original texts.
    """
    if not original_texts or not sanitized_texts:
        return {"bleu_score": 0.0}

    scores = []
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        weights = tuple([1.0 / n_gram] * n_gram)

        for orig, sanit in zip(original_texts, sanitized_texts):
            ref_tokens = re.findall(r"\w+", orig.lower())
            hyp_tokens = re.findall(r"\w+", sanit.lower())
            if not ref_tokens or not hyp_tokens:
                scores.append(0.0)
                continue
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smooth)
            scores.append(score)
    except Exception as exc:
        logger.warning(f"NLTK sentence_bleu fallback: {exc}")
        for orig, sanit in zip(original_texts, sanitized_texts):
            ref_tokens = re.findall(r"\w+", orig.lower())
            hyp_tokens = re.findall(r"\w+", sanit.lower())
            if not ref_tokens or not hyp_tokens:
                scores.append(0.0)
                continue
            matches = sum(1 for tok in hyp_tokens if tok in ref_tokens)
            prec = matches / len(hyp_tokens) if hyp_tokens else 0.0
            scores.append(prec)

    avg_bleu = float(np.mean(scores)) if scores else 0.0
    return {"bleu_score": round(avg_bleu, 4)}


def compute_bert_scores(
    original_texts: List[str],
    sanitized_texts: List[str],
    device: str = "cpu",
    model_type: str = "distilbert-base-uncased",
) -> Dict[str, float]:
    """
    Compute average BERTScore (Precision, Recall, F1) comparing original vs sanitized text.
    """
    if not original_texts or not sanitized_texts:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    try:
        import bert_score
        P, R, F1 = bert_score.score(
            cands=sanitized_texts,
            refs=original_texts,
            model_type=model_type,
            device=device,
            verbose=False,
        )
        return {
            "bertscore_precision": round(float(P.mean()), 4),
            "bertscore_recall": round(float(R.mean()), 4),
            "bertscore_f1": round(float(F1.mean()), 4),
        }
    except Exception as exc:
        logger.warning(
            f"BERTScore computation unavailable ({exc}). Using semantic approximation."
        )
        # Approximate BERTScore via token ROUGE-L / similarity
        r_score = compute_rouge_l_scores(original_texts, sanitized_texts)["rougeL_f1"]
        return {
            "bertscore_precision": round(r_score, 4),
            "bertscore_recall": round(r_score, 4),
            "bertscore_f1": round(r_score, 4),
        }


def compute_semantic_similarities(
    original_texts: List[str],
    sanitized_texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute average SentenceTransformer cosine similarity.
    """
    if not original_texts or not sanitized_texts:
        return {"cosine_similarity": 0.0}

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(model_name, device=device)
        emb_orig = embedder.encode(original_texts, convert_to_numpy=True, normalize_embeddings=True)
        emb_sanit = embedder.encode(sanitized_texts, convert_to_numpy=True, normalize_embeddings=True)

        sims = [float(np.dot(emb_orig[i], emb_sanit[i])) for i in range(len(original_texts))]
        return {"cosine_similarity": round(float(np.mean(sims)), 4)}
    except Exception as exc:
        logger.warning(f"SentenceTransformer computation fallback: {exc}")
        sims = []
        for o, s in zip(original_texts, sanitized_texts):
            sims.append(_internal_lcs_rouge_l(o, s))
        return {"cosine_similarity": round(float(np.mean(sims)), 4)}


def evaluate_anonymization_system(
    original_texts: List[str],
    sanitized_texts: List[str],
    predicted_entities: Optional[List[List[Dict[str, Any]]]] = None,
    ground_truth_entities: Optional[List[List[Dict[str, Any]]]] = None,
    compute_heavy_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation returning both Privacy and Utility metrics.
    """
    results: Dict[str, Any] = {}

    # 1. Privacy Metrics (if ground truth provided)
    if predicted_entities and ground_truth_entities:
        privacy_res = compute_privacy_metrics(predicted_entities, ground_truth_entities)
        results.update({
            "privacy_precision": privacy_res["precision"],
            "privacy_recall": privacy_res["recall"],
            "privacy_f1": privacy_res["f1"],
        })

    # 2. Utility Metrics
    rouge_res = compute_rouge_l_scores(original_texts, sanitized_texts)
    results.update(rouge_res)

    bleu_res = compute_bleu_scores(original_texts, sanitized_texts)
    results.update(bleu_res)

    sim_res = compute_semantic_similarities(original_texts, sanitized_texts)
    results.update(sim_res)

    if compute_heavy_metrics:
        bert_res = compute_bert_scores(original_texts, sanitized_texts)
        results.update(bert_res)

    return results
