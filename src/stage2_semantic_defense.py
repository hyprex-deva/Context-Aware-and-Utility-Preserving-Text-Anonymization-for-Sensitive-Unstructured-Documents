"""
Stage 2: Semantic Reasoning & Quasi-Identifier Generalization using Local SLM (Ollama qwen2.5:0.5b).

This module inspects the Stage 1 sanitized text using the local quantized Small Language Model
`qwen2.5:0.5b` via Ollama. It identifies contextual quasi-identifiers (rare job titles,
demographic outliers, hyper-specific dates/events, and financial specifics) and generalizes them
hierarchically. A SentenceTransformers embedding guardrail (all-MiniLM-L6-v2) verifies that
semantic cosine similarity remains >= 0.80 to guarantee preservation of document utility.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Stage2SemanticDefense")


class SemanticQuasiIdentifierDefense:
    """
    Stage 2 Contextual Quasi-Identifier Generalizer powered by local Ollama (qwen2.5:0.5b)
    and guarded by SentenceTransformers (all-MiniLM-L6-v2) semantic drift verification.
    """

    SIMILARITY_THRESHOLD = 0.80

    # Structured prompt for Qwen2.5-0.5B JSON quasi-identifier generalization
    SYSTEM_PROMPT = """You are an expert Privacy-Preserving NLP Engine.
Your task is to identify and hierarchically generalize high-risk CONTEXTUAL QUASI-IDENTIFIERS in text that has already undergone direct PII anonymization in Stage 1.

Quasi-identifiers to generalize:
- Hyper-specific or unique job titles (e.g., "sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors" -> "Senior Medical Specialist")
- Specific combined dates or birth dates (e.g., "DOB: 12/04/1982" -> "in their early 40s", "on August 14, 2023" -> "in recent months")
- Hyper-specific rare accomplishments/events (e.g., "only employee holding 4 patents in topological quantum gates" -> "a specialized researcher with multiple domain patents")
- Exact high monetary amounts or account specifics (e.g., "$45,000" -> "a substantial commercial fund")
- Specific customer dispute or infrastructure details that could re-identify someone through background knowledge.

Rules:
1. Preserve all synthetic names, places, and organizations created in Stage 1.
2. Maintain complete grammatical fluency, sentence structure, and core semantic meaning.
3. Output your response strictly as a JSON object with keys:
   - "generalized_text": string containing the rewritten, privacy-safe text.
   - "modifications": list of objects [{"original_span": "...", "generalized_span": "...", "reason": "..."}]
"""

    def __init__(
        self,
        ollama_model: str = "qwen2.5:0.5b",
        ollama_url: str = "http://localhost:11434",
        embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.80,
        device: str = "cpu",
    ) -> None:
        """
        Initialize Stage 2 LLM Semantic Defense Engine.

        Args:
            ollama_model: Local Ollama model tag (default: "qwen2.5:0.5b").
            ollama_url: Base endpoint URL for Ollama.
            embedder_model_name: SentenceTransformers model for semantic drift check.
            similarity_threshold: Minimum cosine similarity required to accept generalized text.
            device: Computing device for embedder ("cpu" or "cuda").
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.similarity_threshold = similarity_threshold
        self.device = device
        self._embedder = None

        self._init_embedder(embedder_model_name)

    def _init_embedder(self, model_name: str) -> None:
        """Initialize SentenceTransformer embedding model for semantic drift verification."""
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
        Compute cosine similarity between two texts using SentenceTransformers
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
                return max(0.0, min(1.0, cosine_sim))
            except Exception as exc:
                logger.error(f"Error computing embedding similarity: {exc}")

        # Lightweight fallback: character/word token Jaccard-Cosine metric
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return float(intersection / union) if union > 0 else 0.0

    def _query_ollama(self, stage1_text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Query local Ollama server running qwen2.5:0.5b to perform LLM-driven
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
                timeout=45,
            )
            if response.status_code == 200:
                data = response.json()
                raw_response = data.get("response", "{}").strip()
                
                # Parse JSON output from Ollama
                clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response, flags=re.MULTILINE).strip()
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
                                mods.append({"description": m})
                    elif isinstance(raw_mods, str):
                        mods.append({"description": raw_mods})

                    if gen_text and isinstance(gen_text, str) and len(gen_text.strip()) > 5:
                        return gen_text.strip(), mods

            logger.warning(f"Ollama returned non-200 status code: {response.status_code}")
        except Exception as exc:
            logger.error(f"Failed to query Ollama ({self.ollama_model}): {exc}")

        return stage1_text, []

    def generalize_text(
        self, stage1_text: str, original_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 semantic reasoning via Ollama LLM (qwen2.5:0.5b),
        apply hierarchical quasi-identifier generalization, and verify
        semantic drift using the cosine similarity guardrail.

        Args:
            stage1_text: Output text from Stage 1 (with synthetic surrogates).
            original_text: Optional original un-anonymized text.

        Returns:
            Dict containing:
            - 'final_text': Accepted anonymized text (or Stage 1 fallback if drift threshold failed).
            - 'candidate_text': Raw output from Ollama LLM before guardrail check.
            - 'similarity_score': Cosine similarity float between Stage 1 and Stage 2 candidate.
            - 'drift_passed': Boolean (similarity >= threshold).
            - 'generalized_spans': List of modification metadata dicts.
            - 'backend_used': "Ollama LLM (qwen2.5:0.5b)".
        """
        if not stage1_text or not stage1_text.strip():
            return {
                "final_text": stage1_text or "",
                "candidate_text": stage1_text or "",
                "similarity_score": 1.0,
                "drift_passed": True,
                "generalized_spans": [],
                "backend_used": f"Ollama LLM ({self.ollama_model})",
            }

        # Step 1: Query Ollama LLM
        candidate_text, mods = self._query_ollama(stage1_text)

        # Step 2: Compute Semantic Drift
        sim_score = self.compute_similarity(stage1_text, candidate_text)

        # Step 3: Guardrail Check (Threshold >= 0.80 or user-specified)
        drift_passed = sim_score >= self.similarity_threshold

        if drift_passed:
            final_text = candidate_text
            logger.info(
                f"Stage 2 Guardrail PASSED (Cosine Sim: {sim_score:.4f} >= {self.similarity_threshold})."
            )
        else:
            final_text = stage1_text
            logger.warning(
                f"Stage 2 Guardrail REJECTED (Cosine Sim: {sim_score:.4f} < {self.similarity_threshold}). "
                "Retaining Stage 1 surrogate text to prevent utility degradation."
            )

        return {
            "final_text": final_text,
            "candidate_text": candidate_text,
            "similarity_score": round(sim_score, 4),
            "drift_passed": drift_passed,
            "generalized_spans": mods,
            "backend_used": f"Ollama LLM ({self.ollama_model})",
        }
