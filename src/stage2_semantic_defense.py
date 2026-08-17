"""
Stage 2: Semantic Reasoning & Quasi-Identifier Generalization using Local SLM (Ollama qwen2.5:1.5b).

This module inspects the Stage 1 sanitized text using the local quantized Small Language Model
`qwen2.5:1.5b` via Ollama. It identifies contextual quasi-identifiers (rare job titles,
demographic outliers, hyper-specific dates/events, and financial specifics) and generalizes them
hierarchically.

A multi-criteria composite decision engine evaluates candidate rewrites across:
1. Semantic Similarity (S_semantic): SentenceTransformers (all-MiniLM-L6-v2) cosine similarity.
2. Quasi-Identifier Abstraction Score (S_qi_abstraction): Evaluates residual risk across
   SLM-independently detected single and compound quasi-identifiers (dates, money, rare roles, demographics, locations).
3. Readability & Text Integrity (S_readability): Length-ratio factor and syntactic consistency.

Three-Way Safety Guardrail Acceptance Rule:
    Candidate rewrite is accepted iff:
        S_composite >= tau (default: 0.72)
        AND S_semantic >= floor_sim (default: 0.60)
        AND S_qi_abstraction >= qi_floor (default: 0.50)
    Fails back safely to Stage 1 text if any condition is violated.

NOTE: All thresholds and weights are provisional development defaults intended for
subsequent experimental validation/tuning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from src.quasi_identifier_detector import QuasiIdentifierDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Stage2SemanticDefense")


class SemanticQuasiIdentifierDefense:
    """
    Stage 2 Contextual Quasi-Identifier Generalizer powered by local Ollama (qwen2.5:1.5b)
    and guarded by an SLM-Independent Deterministic Multi-Criteria Composite Decision Engine:
    - S_semantic: SentenceTransformers (all-MiniLM-L6-v2) embedding similarity
    - S_qi_abstraction: Deterministic quasi-identifier abstraction and residual risk evaluation
    - S_readability: Length ratio and syntactic/punctuation structural consistency
    """

    # Provisional development defaults (to be tuned experimentally on validation datasets)
    DEFAULT_TAU = 0.72
    DEFAULT_FLOOR_SIM = 0.60
    DEFAULT_QI_FLOOR = 0.50
    DEFAULT_WEIGHTS = {"sim": 0.50, "qi": 0.30, "read": 0.20}

    # Structured prompt for Qwen2.5 JSON quasi-identifier generalization
    SYSTEM_PROMPT = """You are a privacy-preserving text generalizer.
Your goal is to REWRITE the input document by replacing and generalizing contextual quasi-identifiers into broader categories:
- Coarsen exact dollar amounts (e.g. "$45,000" -> "a five-figure wire transfer", "$25" -> "a standard service charge").
- Coarsen exact dates (e.g. "2024-03-15" -> "in early 2024", "on August 14, 2023" -> "in late summer 2023").
- Coarsen hyper-specific job titles or rare specialties (e.g. "Senior Wealth Manager" -> "financial advisory staff", "sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors" -> "Senior Medical Specialist").
- Coarsen birth dates or age markers (e.g. "DOB: 12/04/1982" -> "in their early 40s").
- Preserve synthetic names, emails, and organizations created in Stage 1.

You MUST return a valid JSON object strictly with keys:
"generalized_text": "the rewritten text with coarsened numbers/dates/titles",
"modifications": [{"original_span": "...", "generalized_span": "...", "reason": "..."}]
"""

    def __init__(
        self,
        ollama_model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434",
        embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau: float = DEFAULT_TAU,
        floor_sim: float = DEFAULT_FLOOR_SIM,
        qi_floor: float = DEFAULT_QI_FLOOR,
        weights: Optional[Dict[str, float]] = None,
        similarity_threshold: Optional[float] = None,
        qi_detector: Optional[QuasiIdentifierDetector] = None,
        mitigation_threshold: float = 0.10,
        exposure_threshold: float = 0.90,
        compound_alpha: float = 0.40,
        device: str = "cpu",
        request_timeout: int = 60,
    ) -> None:
        """
        Initialize Stage 2 LLM Semantic Defense Engine with Independent QI Guardrail.

        Args:
            ollama_model: Local Ollama model tag (default: "qwen2.5:1.5b").
            ollama_url: Base endpoint URL for Ollama.
            embedder_model_name: SentenceTransformers model for semantic drift check.
            tau: Minimum composite score required to accept candidate rewrite (provisional default: 0.72).
            floor_sim: Hard semantic similarity floor required regardless of composite score (provisional default: 0.60).
            qi_floor: Hard QI abstraction floor required regardless of composite score (provisional default: 0.50).
            weights: Weight dictionary for criteria {"sim": w_sim, "qi": w_qi, "read": w_read}.
            similarity_threshold: Backward-compatibility alias for tau.
            qi_detector: Optional custom QuasiIdentifierDetector instance.
            mitigation_threshold: Configurable residual risk threshold for 'mitigated' status (default: 0.10).
            exposure_threshold: Configurable residual risk threshold for 'exposed' status (default: 0.90).
            compound_alpha: Conservative weighting factor for compound QI residual risk (default: 0.40).
            device: Computing device for embedder ("cpu" or "cuda").
            request_timeout: HTTP request timeout in seconds for Ollama calls (default: 60).
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.tau = similarity_threshold if similarity_threshold is not None else tau
        self.floor_sim = floor_sim
        self.qi_floor = qi_floor
        self.weights = self._normalize_weights(weights or self.DEFAULT_WEIGHTS)
        self.device = device
        self.request_timeout = request_timeout
        self._embedder = None

        self.qi_detector = qi_detector or QuasiIdentifierDetector(
            mitigation_threshold=mitigation_threshold,
            exposure_threshold=exposure_threshold,
            compound_alpha=compound_alpha,
        )

        self._init_embedder(embedder_model_name)

    @property
    def similarity_threshold(self) -> float:
        """Backward-compatibility property returning tau threshold."""
        return self.tau

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        """Backward-compatibility setter updating tau threshold."""
        self.tau = value

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights are non-negative and sum to 1.0 (with backward compatibility for 'priv' key)."""
        w_sim = max(0.0, float(weights.get("sim", 0.50)))
        # Support 'qi' or legacy 'priv' key
        w_qi = max(0.0, float(weights.get("qi", weights.get("priv", 0.30))))
        w_read = max(0.0, float(weights.get("read", 0.20)))
        total = w_sim + w_qi + w_read
        if total <= 0.0:
            return {"sim": 0.50, "qi": 0.30, "read": 0.20}
        return {
            "sim": round(w_sim / total, 4),
            "qi": round(w_qi / total, 4),
            "read": round(w_read / total, 4),
        }

    def _init_embedder(self, model_name: str) -> None:
        """Initialize SentenceTransformer embedding model for semantic drift verification."""
        if not model_name:
            self._embedder = None
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedder: {model_name} on {self.device}...")
            self._embedder = SentenceTransformer(model_name, device=self.device)
            logger.info("SentenceTransformer embedder successfully loaded.")
        except Exception as exc:
            logger.warning(
                f"Could not load SentenceTransformer ({exc}). Using internal token similarity fallback."
            )
            self._embedder = None

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity (S_semantic) between two texts using SentenceTransformers
        or an internal token-overlap fallback.

        Args:
            text_a: First text string (Stage 1 surrogate text).
            text_b: Second text string (Stage 2 candidate text).

        Returns:
            Cosine similarity float in range [0.0, 1.0].
        """
        if not text_a.strip() or not text_b.strip():
            return 1.0 if text_a == text_b else 0.0

        if self._embedder is not None:
            try:
                embeddings = self._embedder.encode(
                    [text_a, text_b], convert_to_numpy=True, normalize_embeddings=True
                )
                cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
                return float(np.clip(cosine_sim, 0.0, 1.0))
            except Exception as exc:
                logger.error(f"Error computing embedding similarity: {exc}")

        # Lightweight fallback: word token Dice coefficient & character n-gram overlap
        words_a = re.findall(r"\w+", text_a.lower())
        words_b = re.findall(r"\w+", text_b.lower())
        if not words_a or not words_b:
            return 0.0
        set_a, set_b = set(words_a), set(words_b)
        overlap_words = len(set_a & set_b)
        word_dice = (2.0 * overlap_words) / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0.0

        # Character trigrams for morphological and sub-token overlap
        clean_a, clean_b = text_a.lower().strip(), text_b.lower().strip()
        tri_a = {clean_a[i : i + 3] for i in range(len(clean_a) - 2)}
        tri_b = {clean_b[i : i + 3] for i in range(len(clean_b) - 2)}
        tri_dice = (
            (2.0 * len(tri_a & tri_b)) / (len(tri_a) + len(tri_b))
            if (tri_a and tri_b and (len(tri_a) + len(tri_b)) > 0)
            else word_dice
        )

        return float(np.clip(0.60 * word_dice + 0.40 * tri_dice, 0.0, 1.0))

    def compute_qi_abstraction(
        self,
        stage1_text: str,
        candidate_text: str,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute Quasi-Identifier Abstraction Score (S_qi_abstraction) using the
        SLM-Independent Deterministic QI Detection Layer.

        Evaluates per-QI residual risk across single and compound QIs in candidate_text.

        Formula:
            S_qi_abstraction = 1.0 - (sum(w_i * r_i) / sum(w_i))

        If no QIs are detected in stage1_text, defaults to 1.0.

        Args:
            stage1_text: Output text from Stage 1 (reference input to Stage 2).
            candidate_text: Candidate rewrite proposed by the SLM.

        Returns:
            Tuple of (qi_abstraction_score: float, qi_evaluations: List[Dict[str, Any]]).
        """
        if not stage1_text or not stage1_text.strip():
            return 1.0, []

        # 1. Independent Deterministic QI Detection on Stage 1 text
        detected_qis = self.qi_detector.detect_quasi_identifiers(stage1_text)

        if not detected_qis:
            # If no quasi-identifiers were independently detected, default S_qi_abstraction = 1.0
            return 1.0, []

        # 2. Evaluate residual risk per detected QI
        evaluations: List[Dict[str, Any]] = []
        total_weighted_risk = 0.0
        total_weight = 0.0

        for qi in detected_qis:
            eval_res = self.qi_detector.evaluate_qi_mitigation(qi, candidate_text)
            evaluations.append(eval_res)

            weight = qi.get("risk_level", 1.0)
            residual_risk = eval_res.get("residual_risk", 0.0)

            total_weighted_risk += weight * residual_risk
            total_weight += weight

        # 3. Compute weighted abstraction score
        if total_weight > 0:
            qi_score = 1.0 - (total_weighted_risk / total_weight)
        else:
            qi_score = 1.0

        qi_score = float(np.clip(qi_score, 0.0, 1.0))
        return round(qi_score, 4), evaluations

    def compute_privacy_reduction(
        self, candidate_text: str, modifications: List[Dict[str, Any]]
    ) -> float:
        """
        Backward-compatibility helper evaluating SLM modifications list.
        Note: The guardrail's authoritative score is compute_qi_abstraction.
        """
        if not modifications:
            return 1.0

        flagged_phrases: List[str] = []
        for m in modifications:
            if isinstance(m, dict):
                span = (
                    m.get("original_span")
                    or m.get("original")
                    or m.get("phrase")
                    or m.get("span")
                    or m.get("text")
                )
                if span and isinstance(span, str) and span.strip():
                    flagged_phrases.append(span.strip())
            elif isinstance(m, str) and m.strip():
                flagged_phrases.append(m.strip())

        if not flagged_phrases:
            return 1.0

        candidate_lower = candidate_text.lower()
        unabstracted_count = 0

        for phrase in flagged_phrases:
            if phrase.lower() in candidate_lower:
                unabstracted_count += 1

        total_flagged = len(flagged_phrases)
        privacy_score = 1.0 - (unabstracted_count / max(total_flagged, 1))
        return float(np.clip(privacy_score, 0.0, 1.0))

    def compute_readability_score(self, original_text: str, candidate_text: str) -> float:
        """
        Compute Readability & Text Integrity metric (S_readability).

        Combines:
        1. Length Ratio Factor: Penalizes severe truncation or extreme hallucinated expansion.
        2. Syntactic Integrity Factor: Checks sentence termination and bracket/quote balance.

        Args:
            original_text: Reference text (Stage 1 surrogate text).
            candidate_text: Candidate rewritten text.

        Returns:
            Readability score in range [0.0, 1.0].
        """
        orig_clean = original_text.strip()
        cand_clean = candidate_text.strip()

        if not orig_clean and not cand_clean:
            return 1.0
        if not cand_clean:
            return 0.0

        # 1. Length Ratio Factor
        len_orig = len(orig_clean)
        len_cand = len(cand_clean)
        length_diff = abs(len_orig - len_cand)
        length_factor = max(0.0, 1.0 - (length_diff / max(len_orig, 1)))

        # 2. Syntactic / Punctuation Integrity Factor
        valid_terminals = {".", "!", "?", '"', "'", ")", "}", "]", "”", "’", "\n"}

        # Sentence termination check
        termination_score = 1.0
        if orig_clean and orig_clean[-1] in {".", "!", "?"}:
            if cand_clean and cand_clean[-1] not in valid_terminals:
                termination_score = 0.5
            elif cand_clean and cand_clean[-1] in {",", ":", ";", "-", "—"}:
                termination_score = 0.4

        # Bracket and quote parity check
        structural_penalty = 0.0
        if cand_clean.count('"') % 2 != 0:
            structural_penalty += 0.2
        if cand_clean.count("(") != cand_clean.count(")"):
            structural_penalty += 0.2
        if cand_clean.count("[") != cand_clean.count("]"):
            structural_penalty += 0.2
        if cand_clean.count("{") != cand_clean.count("}"):
            structural_penalty += 0.2

        syntactic_score = max(0.0, termination_score - structural_penalty)

        # Weighted blend (70% length factor + 30% syntactic integrity)
        readability_score = (0.70 * length_factor) + (0.30 * syntactic_score)
        return float(np.clip(readability_score, 0.0, 1.0))

    def evaluate_composite_decision(
        self,
        stage1_text: str,
        candidate_text: str,
        modifications: Optional[List[Dict[str, Any]]] = None,
        tau: Optional[float] = None,
        floor_sim: Optional[float] = None,
        qi_floor: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate rewrite using the multi-criteria composite decision function
        with Three-Way Safety Guardrail enforcement.

        Composite Score:
            S_composite = (w_sim * S_semantic) + (w_qi * S_qi_abstraction) + (w_read * S_readability)

        Three-Way Acceptance Rule:
            ACCEPT iff S_composite >= tau AND S_semantic >= floor_sim AND S_qi_abstraction >= qi_floor
            REJECT otherwise (fallback to Stage 1 text)
        """
        effective_tau = self.tau if tau is None else tau
        effective_floor = self.floor_sim if floor_sim is None else floor_sim
        effective_qi_floor = self.qi_floor if qi_floor is None else qi_floor
        effective_weights = self._normalize_weights(weights) if weights else self.weights
        mods = modifications or []

        # 1. Compute Individual Component Scores
        s_semantic = self.compute_similarity(stage1_text, candidate_text)
        s_qi_abstraction, qi_analysis = self.compute_qi_abstraction(stage1_text, candidate_text)
        s_readability = self.compute_readability_score(stage1_text, candidate_text)

        # 2. Compute Weighted Composite Score
        w_sim = effective_weights["sim"]
        w_qi = effective_weights["qi"]
        w_read = effective_weights["read"]
        s_composite = (w_sim * s_semantic) + (w_qi * s_qi_abstraction) + (w_read * s_readability)
        s_composite = float(np.clip(s_composite, 0.0, 1.0))

        # 3. Three-Way Guardrail Decision Rule
        is_accepted = bool(
            s_composite >= effective_tau
            and s_semantic >= effective_floor
            and s_qi_abstraction >= effective_qi_floor
        )
        fallback_triggered = not is_accepted

        if is_accepted:
            final_text = candidate_text
            logger.info(
                f"Stage 2 Decision ACCEPTED: Composite={s_composite:.4f} >= {effective_tau:.2f} "
                f"(Semantic={s_semantic:.4f} >= {effective_floor:.2f}, "
                f"QI-Abstraction={s_qi_abstraction:.4f} >= {effective_qi_floor:.2f}, "
                f"Readability={s_readability:.4f})"
            )
        else:
            final_text = stage1_text
            reasons = []
            if s_semantic < effective_floor:
                reasons.append(f"Semantic {s_semantic:.4f} < floor {effective_floor:.2f}")
            if s_qi_abstraction < effective_qi_floor:
                reasons.append(f"QI-Abstraction {s_qi_abstraction:.4f} < floor {effective_qi_floor:.2f}")
            if s_composite < effective_tau:
                reasons.append(f"Composite {s_composite:.4f} < tau {effective_tau:.2f}")
            logger.warning(
                f"Stage 2 Decision REJECTED ({', '.join(reasons)}). "
                "Retaining Stage 1 surrogate text to prevent privacy/utility degradation."
            )

        return {
            "final_text": final_text,
            "candidate_text": candidate_text,
            "is_accepted": is_accepted,
            "fallback_triggered": fallback_triggered,
            "composite_score": round(s_composite, 4),
            "metrics_breakdown": {
                "semantic_similarity": round(s_semantic, 4),
                "qi_abstraction_score": round(s_qi_abstraction, 4),
                "readability_score": round(s_readability, 4),
                # Backward-compatibility alias
                "privacy_reduction_score": round(s_qi_abstraction, 4),
            },
            "thresholds": {
                "tau": effective_tau,
                "floor_sim": effective_floor,
                "qi_floor": effective_qi_floor,
                "weights": effective_weights,
            },
            "qi_analysis": qi_analysis,
            "modifications": mods,
            # Backward-compatibility aliases
            "drift_passed": is_accepted,
            "similarity_score": round(s_semantic, 4),
            "generalized_spans": mods,
            "backend_used": f"Ollama LLM ({self.ollama_model})",
        }

    def _query_ollama(self, stage1_text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Query local Ollama server running qwen2.5:1.5b to perform LLM-driven
        hierarchical quasi-identifier generalization and contextual rephrasing.
        """
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Input Document to Generalize:\n{stage1_text}\n\n"
            f"Output JSON:"
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.request_timeout,
            )
            if response.status_code == 200:
                data = response.json()
                raw_response = data.get("response", "{}").strip()

                # Parse JSON output from Ollama
                clean_json = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", raw_response, flags=re.MULTILINE
                ).strip()
                match = re.search(r"\{.*\}", clean_json, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    gen_text = parsed.get("generalized_text", stage1_text)
                    raw_mods = parsed.get("modifications", [])

                    # Normalize modifications list
                    mods: List[Dict[str, str]] = []
                    if isinstance(raw_mods, list):
                        for m in raw_mods:
                            if isinstance(m, dict):
                                mods.append(m)
                            elif isinstance(m, str):
                                mods.append({"description": m, "original_span": m})
                    elif isinstance(raw_mods, str):
                        mods.append({"description": raw_mods, "original_span": raw_mods})

                    if gen_text and isinstance(gen_text, str) and len(gen_text.strip()) > 5:
                        return gen_text.strip(), mods

        except Exception as exc:
            logger.error(f"Failed to query Ollama ({self.ollama_model}): {exc}")

        return stage1_text, []

    def generalize_text(
        self,
        stage1_text: str,
        original_text: Optional[str] = None,
        tau: Optional[float] = None,
        floor_sim: Optional[float] = None,
        qi_floor: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 semantic reasoning via Ollama LLM (qwen2.5:1.5b),
        apply hierarchical quasi-identifier generalization, and verify candidate
        rewrite with the independent multi-criteria composite decision guardrail.

        Args:
            stage1_text: Output text from Stage 1 (with synthetic surrogates).
            original_text: Optional original un-anonymized text.
            tau: Optional override for composite threshold.
            floor_sim: Optional override for hard semantic similarity floor.
            qi_floor: Optional override for hard QI abstraction safety floor.
            weights: Optional override for component weights {"sim": ..., "qi": ..., "read": ...}.

        Returns:
            Structured dictionary with final text and detailed decision telemetry.
        """
        if not stage1_text or not stage1_text.strip():
            effective_tau = self.tau if tau is None else tau
            effective_floor = self.floor_sim if floor_sim is None else floor_sim
            effective_qi_floor = self.qi_floor if qi_floor is None else qi_floor
            effective_weights = self._normalize_weights(weights) if weights else self.weights
            return {
                "final_text": stage1_text or "",
                "candidate_text": stage1_text or "",
                "is_accepted": True,
                "fallback_triggered": False,
                "composite_score": 1.0,
                "metrics_breakdown": {
                    "semantic_similarity": 1.0,
                    "qi_abstraction_score": 1.0,
                    "readability_score": 1.0,
                    "privacy_reduction_score": 1.0,
                },
                "thresholds": {
                    "tau": effective_tau,
                    "floor_sim": effective_floor,
                    "qi_floor": effective_qi_floor,
                    "weights": effective_weights,
                },
                "qi_analysis": [],
                "modifications": [],
                "drift_passed": True,
                "similarity_score": 1.0,
                "generalized_spans": [],
                "backend_used": f"Ollama LLM ({self.ollama_model})",
            }

        # Step 1: Query Ollama LLM
        candidate_text, mods = self._query_ollama(stage1_text)

        # Step 2 & 3: Evaluate Composite Decision
        return self.evaluate_composite_decision(
            stage1_text=stage1_text,
            candidate_text=candidate_text,
            modifications=mods,
            tau=tau,
            floor_sim=floor_sim,
            qi_floor=qi_floor,
            weights=weights,
        )

