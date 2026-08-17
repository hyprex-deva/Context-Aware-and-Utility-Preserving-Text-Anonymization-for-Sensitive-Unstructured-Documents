"""
Context-Aware and Utility-Preserving Text Anonymization Framework.

Modules:
- stage1_direct_pii: Token-Level Direct PII Detection & Synthetic Surrogate Replacement.
- stage2_semantic_defense: Semantic Reasoning & Quasi-Identifier Generalization with MiniLM Drift Guardrail.
- baselines: Microsoft Presidio and Rigid Redaction baselines.
- evaluate_metrics: Evaluation utilities for Privacy and Utility retention.
"""

from src.stage1_direct_pii import DirectPIIAnonymizer, DetectedEntity
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense
from src.quasi_identifier_detector import QuasiIdentifierDetector
from src.baselines import BaselinePresidio, BaselineRedacted
from src.evaluate_metrics import (
    compute_privacy_metrics,
    compute_rouge_l_scores,
    compute_bert_scores,
    compute_semantic_similarities,
    evaluate_anonymization_system,
)

__all__ = [
    "DirectPIIAnonymizer",
    "DetectedEntity",
    "SemanticQuasiIdentifierDefense",
    "QuasiIdentifierDetector",
    "BaselinePresidio",
    "BaselineRedacted",
    "compute_privacy_metrics",
    "compute_rouge_l_scores",
    "compute_bert_scores",
    "compute_semantic_similarities",
    "evaluate_anonymization_system",
]
