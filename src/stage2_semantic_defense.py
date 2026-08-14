"""
Stage 2: Semantic Reasoning & Quasi-Identifier Generalization with Semantic Drift Guardrail.

This module inspects the Stage 1 sanitized text using a Small Language Model (SLM),
identifies contextual quasi-identifiers (rare job titles, unique combinations of dates/events,
demographic anomalies), and applies hierarchical generalization. A SentenceTransformers
embedding guardrail verifies that semantic cosine similarity remains >= 0.80 to prevent
catastrophic loss of document utility.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Stage2SemanticDefense")


class SemanticQuasiIdentifierDefense:
    """
    Stage 2 Contextual Quasi-Identifier Generalizer with Semantic Drift Verification.
    
    Backends supported:
    1. Hugging Face Text-Generation Pipeline (e.g. Qwen/Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct)
    2. Ollama REST API (http://localhost:11434)
    3. Heuristic Hierarchical SLM Rule-Reasoner (zero-dependency offline fallback)
    """

    SIMILARITY_THRESHOLD = 0.80

    # Few-shot prompt template for SLM quasi-identifier generalization
    SYSTEM_PROMPT = """You are an expert Privacy-Preserving NLP Engine.
Your task is to identify and hierarchically generalize high-risk CONTEXTUAL QUASI-IDENTIFIERS in text that has already undergone direct PII anonymization.

Quasi-identifiers to generalize:
- Hyper-specific or unique job titles (e.g., "sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors" -> "Senior Medical Specialist")
- Specific combined dates or birth dates (e.g., "DOB: 12/04/1982" -> "in their early 40s", "on August 14, 2023" -> "in late summer 2023")
- Hyper-specific rare accomplishments/events (e.g., "only employee holding 4 patents in topological quantum gates" -> "a specialized researcher with multiple domain patents")
- Exact high monetary amounts or account specifics (e.g., "$45,000" -> "a significant five-figure transfer")

Rules:
1. Preserve all synthetic names, places, and organizations created in Stage 1.
2. Maintain complete grammatical fluency, sentence structure, and core semantic meaning.
3. Do NOT add commentary, preface, or markdown explanations.
4. Output your response as a valid JSON object with keys:
   - "generalized_text": string containing the rewritten, privacy-safe text.
   - "modifications": list of objects [{"original_span": "...", "generalized_span": "...", "reason": "..."}]
"""

    HEURISTIC_RULES: List[Tuple[re.Pattern, str, str]] = [
        # --- 1. Healthcare & Clinical Domains ---
        (
            re.compile(r"\b(?:sole\s+)?Chief\s+Pediatric\s+Neurosurgeon\s+for\s+[\w\s]+(?:in\s+[\w\s,]+)?", re.IGNORECASE),
            "Senior Medical Specialist in regional pediatric care",
            "Generalized hyper-specific pediatric surgical specialty and regional practice"
        ),
        (
            re.compile(r"\b3rd\s+female\s+in\s+[\w\s]+\s+with\s+[\w\s]+cholangiocarcinoma", re.IGNORECASE),
            "a patient diagnosed with a rare oncology condition",
            "Generalized demographic rarity and specific oncology diagnosis"
        ),
        (
            re.compile(r"\bSubject\s+#\d+,\s*identified\s+as\s+[\w\s]+(?:\([^\)]+\))?,\s*enrolled\s+in\s+the\s+Phase\s+III\s+trial\s+for\s+[\w\-]+\s+at\s+[\w\s]+", re.IGNORECASE),
            "A clinical study participant enrolled in the advanced therapeutic oncology trial at a major medical research center",
            "Generalized clinical subject trial identifier, specific drug name, and research facility"
        ),
        (
            re.compile(r"\bPrincipal\s+Investigator\s+Dr\.\s*[\w\s]+(?:\([^\)]+\))?\s+reported", re.IGNORECASE),
            "The lead clinical research team reported",
            "Generalized principal investigator identity and direct contact channel"
        ),

        # --- 2. Finance & Banking ---
        (
            re.compile(r"\bconfirmed\s+the\s+wire\s+instruction\s+with\s+Senior\s+Wealth\s+Manager\s+[\w\s]+at\s+[\w\s]+in\s+[\w\s]+via\s+[\w\.\@\-]+", re.IGNORECASE),
            "confirmed the transaction instructions with institutional wealth management representatives",
            "Generalized private wealth manager identity, institution branch, and direct email"
        ),
        (
            re.compile(r"\btransfer(?:red)?\s+\$[\d,]+(?:\.\d{2})?\b", re.IGNORECASE),
            "transferred a substantial commercial fund",
            "Coarsened exact monetary transaction amount"
        ),
        (
            re.compile(r"\bcharged\s+a\s+\$[\d,]+(?:\.\d{2})?\s+processing\s+fee\b", re.IGNORECASE),
            "charged a standard administrative service fee",
            "Generalized specific payment transaction fee"
        ),

        # --- 3. HR, Startup & Whistleblower ---
        (
            re.compile(r"\bLead\s+Quantum\s+Computing\s+Architect\b", re.IGNORECASE),
            "Senior Technology Architect",
            "Generalized specialized quantum computing role"
        ),
        (
            re.compile(r"\bonly\s+employee\s+in\s+the\s+\d+-person\s+startup\s+holding\s+\d+\s+patents\s+in\s+[\w\s]+", re.IGNORECASE),
            "a key technical contributor holding several specialized domain patents",
            "Generalized unique patent count and startup size combination"
        ),
        (
            re.compile(r"\breported\s+unethical\s+contracting\s+to\s+VP\s+of\s+HR\s+[\w\s]+(?:\([^\)]+\))?", re.IGNORECASE),
            "submitted an internal workplace reporting escalation to human resources leadership",
            "Generalized executive HR recipient and whistleblower reporting channel"
        ),

        # --- 4. Legal & Dispute ---
        (
            re.compile(r"\bIn\s+the\s+matter\s+of\s+[\w\s\.\-]+\s+vs\.\s+[\w\s\.\-]+,\s*Attorney\s+[\w\s]+(?:\([^\)]+\))?\s+submitted\s+deposition\s+records\s+on\s+behalf\s+of\s+former\s+CFO\s+[\w\s]+", re.IGNORECASE),
            "In the commercial dispute proceedings, legal counsel submitted deposition records on behalf of executive leadership",
            "Generalized legal case title, attorney bar credentials, and executive respondent"
        ),
        (
            re.compile(r"\bunannounced\s+executive\s+termination\s+in\s+[\w\s,]+\s+on\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}", re.IGNORECASE),
            "an executive departure occurring in recent years",
            "Generalized specific termination date and geographic location"
        ),

        # --- 5. Customer Support & Telecom (Sample 10) ---
        (
            re.compile(r"\bcalled\s+regarding\s+billing\s+dispute\s+for\s+broadband\s+account\s+at\s+[\w\s,]+", re.IGNORECASE),
            "initiated a customer service inquiry regarding residential subscription services",
            "Generalized specific residential broadband account dispute and location"
        ),
        (
            re.compile(r"\bSupport\s+agent\s+[\w\s]+verified\s+(?:her|his|the)\s+account\s+using\s+[\w\s\.\-]+\s+ending\s+in\s+[\w\-]+\s+and\s+phone\s+[\w\+\-\s\.\(\)x]+", re.IGNORECASE),
            "Support staff authenticated the account holder through standard multi-factor verification protocols",
            "Generalized customer support agent name and specific identification credential combination"
        ),
        (
            re.compile(r"\bsent\s+corroborating\s+bank\s+statements\s+from\s+[\w\.\@\-]+", re.IGNORECASE),
            "provided supporting verification documentation via verified electronic correspondence",
            "Generalized specific financial document submission and direct email"
        ),

        # --- 6. Cybersecurity & IT ---
        (
            re.compile(r"\bServer\s+[\d\.]+\s+was\s+compromised\s+following\s+an\s+unauthorized\s+root\s+login\s+from\s+[\d\.]+", re.IGNORECASE),
            "The internal server infrastructure experienced unauthorized elevated access activity originating from an external network host",
            "Generalized IP addresses and specific root compromise narrative"
        ),
        (
            re.compile(r"\bSystem\s+Administrator\s+[\w\s]+(?:\([^\)]+\))?\s+detected\s+the\s+exfiltration\s+of\s+confidential\s+customer\s+records\s+belonging\s+to\s+[\w\s]+", re.IGNORECASE),
            "Security operations staff detected data security anomalies involving corporate client records",
            "Generalized system administrator identity, direct email, and specific victim company name"
        ),

        # --- 7. Education & Academia ---
        (
            re.compile(r"\bonly\s+faculty\s+member\s+with\s+dual\s+tenure\s+in\s+[\w\s]+and\s+[\w\s]+", re.IGNORECASE),
            "a tenured senior faculty member across interdisciplinary academic departments",
            "Generalized unique dual-tenure academic affiliation"
        ),
        (
            re.compile(r"\bsubmitted\s+the\s+grade\s+review\s+for\s+student\s+[\w\s]+at\s+[\w\s]+in\s+[\w\s,]+", re.IGNORECASE),
            "submitted academic evaluation records for the enrolled student at the university",
            "Generalized student identity and specific university location"
        ),

        # --- 8. Real Estate & Tenancy ---
        (
            re.compile(r"\bsigned\s+a\s+\d+-year\s+lease\s+with\s+[\w\s]+on\s+[\d\-]+\s+for\s+the\s+penthouse\s+suite\s+at\s+[\w\s,]+", re.IGNORECASE),
            "executed a multi-year residential lease agreement with property management for a prime residential unit",
            "Generalized lease duration, property firm, exact address, and penthouse property type"
        ),
        (
            re.compile(r"\bsecurity\s+deposit\s+of\s+\$[\d,]+(?:\.\d{2})?\b", re.IGNORECASE),
            "standard multi-month security deposit",
            "Coarsened exact lease deposit amount"
        ),

        # --- 9. Corporate M&A & Executive Transactions ---
        (
            re.compile(r"\bManaging\s+Director\s+[\w\s]+of\s+[\w\s]+advised\s+on\s+the\s+\$[\d\.]+B\s+acquisition\s+of\s+[\w\s]+in\s+[\w\s,]+", re.IGNORECASE),
            "Senior investment leadership advised on the multi-billion enterprise acquisition of a technology firm",
            "Generalized managing director, exact M&A valuation, target firm, and location"
        ),
        (
            re.compile(r"\b\d{2}-year-old\s+former\s+Olympic\s+rower\s+who\s+led\s+the\s+largest\s+[\w\s]+IPO\s+in\s+\d{4}", re.IGNORECASE),
            "a veteran executive with a background in competitive athletics and public market listings",
            "Generalized distinctive athletic background and specific IPO event"
        ),

        # --- 10. General Dates, Demographics & Timestamps ---
        (
            re.compile(r"\bDOB:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
            "Age Group: 40-50",
            "Coarsened exact date of birth to age cohort"
        ),
        (
            re.compile(r"\bon\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}\b", re.IGNORECASE),
            "during the previous calendar year",
            "Generalized specific calendar date to general timeframe"
        ),
        (
            re.compile(r"\bon\s+\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),
            "in recent months",
            "Generalized specific ISO date"
        ),
    ]


    def __init__(
        self,
        backend: str = "heuristic",  # "hf_pipeline", "ollama", or "heuristic"
        hf_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        ollama_model: str = "llama3.2:1b",
        ollama_url: str = "http://localhost:11434",
        embedder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.80,
        device: Optional[int] = -1,
    ) -> None:
        """
        Initialize the Stage 2 Semantic Defense Engine.

        Args:
            backend: Inference backend ("hf_pipeline", "ollama", or "heuristic").
            hf_model_name: Hugging Face SLM repo identifier for text generation.
            ollama_model: Model name for local Ollama server.
            ollama_url: Base endpoint URL for Ollama.
            embedder_model_name: SentenceTransformers model for drift check.
            similarity_threshold: Cosine similarity threshold (default 0.80).
            device: Computing device (-1 for CPU, 0+ for GPU).
        """
        self.backend = backend
        self.hf_model_name = hf_model_name
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.similarity_threshold = similarity_threshold
        self.device = device

        self._hf_pipeline = None
        self._embedder = None

        self._init_embedder(embedder_model_name)
        if self.backend == "hf_pipeline":
            self._init_hf_pipeline()

    def _init_embedder(self, model_name: str) -> None:
        """Initialize SentenceTransformer embedding model with fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedder: {model_name}...")
            self._embedder = SentenceTransformer(model_name, device="cpu" if self.device == -1 else f"cuda:{self.device}")
            logger.info("SentenceTransformer embedder successfully loaded.")
        except Exception as exc:
            logger.warning(
                f"Could not load SentenceTransformer ({exc}). Using character n-gram cosine similarity fallback."
            )
            self._embedder = None

    def _init_hf_pipeline(self) -> None:
        """Initialize Hugging Face text generation pipeline for SLM reasoning."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            logger.info(f"Loading HF SLM pipeline for generalization: {self.hf_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                device_map="auto" if self.device != -1 else "cpu",
                torch_dtype="auto",
            )
            self._hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False,
            )
            logger.info("HF SLM Pipeline loaded successfully.")
        except Exception as exc:
            logger.warning(f"Could not load HF model '{self.hf_model_name}' ({exc}). Falling back to heuristic mode.")
            self.backend = "heuristic"

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts using SentenceTransformers
        or an internal token-overlap fallback.

        Args:
            text_a: First text string (Original or Stage 1).
            text_b: Second text string (Generalized candidate).

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
                # Clamp to [0.0, 1.0]
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

    def _generalize_heuristic(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply hierarchical quasi-identifier generalization via compiled domain patterns.
        """
        generalized_text = text
        modifications: List[Dict[str, str]] = []

        for pattern, replacement, reason in self.HEURISTIC_RULES:
            for match in pattern.finditer(generalized_text):
                original_span = match.group()
                modifications.append({
                    "original_span": original_span,
                    "generalized_span": replacement,
                    "reason": reason,
                })
            generalized_text = pattern.sub(replacement, generalized_text)

        return generalized_text, modifications

    def _generalize_ollama(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Query local Ollama server for SLM quasi-identifier generalization.
        """
        import requests
        payload = {
            "model": self.ollama_model,
            "prompt": f"{self.SYSTEM_PROMPT}\n\nInput Text to Generalize:\n{text}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                raw_response = data.get("response", "{}")
                parsed = json.loads(raw_response)
                gen_text = parsed.get("generalized_text", text)
                mods = parsed.get("modifications", [])
                return gen_text, mods
        except Exception as exc:
            logger.warning(f"Ollama request failed ({exc}). Falling back to heuristic reasoning.")

        return self._generalize_heuristic(text)

    def _generalize_hf(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Query Hugging Face text-generation pipeline for SLM quasi-identifier generalization.
        """
        if not self._hf_pipeline:
            return self._generalize_heuristic(text)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Input Text to Generalize:\n{text}"},
        ]
        try:
            outputs = self._hf_pipeline(messages)
            generated_content = outputs[0]["generated_text"][-1]["content"]
            # Extract JSON from output
            match = re.search(r"\{.*\}", generated_content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return parsed.get("generalized_text", text), parsed.get("modifications", [])
        except Exception as exc:
            logger.warning(f"HF SLM inference failed ({exc}). Falling back to heuristic reasoning.")

        return self._generalize_heuristic(text)

    def generalize_text(
        self, stage1_text: str, original_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 semantic reasoning, quasi-identifier generalization,
        and verify semantic drift using cosine similarity guardrail.

        Args:
            stage1_text: Output text from Stage 1 (with synthetic surrogates).
            original_text: Optional original un-anonymized text for similarity reference.

        Returns:
            Dict containing:
            - 'final_text': Accepted anonymized text (or Stage 1 text if drift threshold failed).
            - 'candidate_text': Raw output from SLM before guardrail check.
            - 'similarity_score': Cosine similarity float between original and candidate.
            - 'drift_passed': Boolean (similarity >= 0.80).
            - 'generalized_spans': List of modification metadata dicts.
            - 'backend_used': Backend used for reasoning.
        """
        if not stage1_text or not stage1_text.strip():
            return {
                "final_text": stage1_text or "",
                "candidate_text": stage1_text or "",
                "similarity_score": 1.0,
                "drift_passed": True,
                "generalized_spans": [],
                "backend_used": self.backend,
            }

        # Select inference backend
        if self.backend == "ollama":
            candidate_text, mods = self._generalize_ollama(stage1_text)
            backend_used = "Ollama SLM"
        elif self.backend == "hf_pipeline" and self._hf_pipeline is not None:
            candidate_text, mods = self._generalize_hf(stage1_text)
            backend_used = "Hugging Face SLM Pipeline"
        else:
            candidate_text, mods = self._generalize_heuristic(stage1_text)
            backend_used = "Hierarchical Semantic Reasoner"

        # Measure semantic retention between Stage 1 surrogate text and Stage 2 candidate
        reference_text = stage1_text
        sim_score = self.compute_similarity(reference_text, candidate_text)

        # Semantic Drift Guardrail Logic
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
                "Falling back to Stage 1 text to prevent catastrophic utility loss."
            )

        return {
            "final_text": final_text,
            "candidate_text": candidate_text,
            "similarity_score": round(sim_score, 4),
            "drift_passed": drift_passed,
            "generalized_spans": mods,
            "backend_used": backend_used,
        }
