"""
Baseline PII Anonymization Models.

This module implements two established enterprise and research baselines for comparison:
1. Baseline A: Microsoft Presidio (Enterprise Standard Tag Replacement: <PERSON>, <EMAIL_ADDRESS>, etc.)
2. Baseline B: Rigid Redaction (Direct Redaction Tagging: [REDACTED])
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Baselines")


class BaselinePresidio:
    """
    Microsoft Presidio Baseline Anonymizer.
    
    Uses Presidio AnalyzerEngine and AnonymizerEngine to perform standard
    enterprise tag-based masking (e.g. replacing 'John' with '<PERSON>').
    """

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        self.languages = languages or ["en"]
        self._analyzer = None
        self._anonymizer = None
        self._init_presidio()

    def _init_presidio(self) -> None:
        """Initialize Presidio Analyzer and Anonymizer with graceful fallback."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Microsoft Presidio Analyzer and Anonymizer engines initialized.")
        except Exception as exc:
            logger.warning(
                f"Presidio initialization failed ({exc}). Using internal regex tag-replacer fallback."
            )
            self._analyzer = None
            self._anonymizer = None

    def _fallback_anonymize(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based tag anonymization if Presidio engine is unavailable."""
        patterns = [
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL_ADDRESS>"),
            (re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"), "<PHONE_NUMBER>"),
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<US_SSN>"),
            (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP_ADDRESS>"),
            (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "<CREDIT_CARD>"),
            (re.compile(r"\b(?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"), "<PERSON>"),
        ]
        sanitized = text
        detected = []
        for pat, tag in patterns:
            for m in pat.finditer(text):
                detected.append({
                    "entity_type": tag.strip("<>"),
                    "start": m.start(),
                    "end": m.end(),
                    "text": m.group(),
                })
            sanitized = pat.sub(tag, sanitized)

        return {
            "sanitized_text": sanitized,
            "detected_entities": detected,
            "engine": "Presidio (Regex Fallback)",
        }

    def anonymize(self, text: str) -> Dict[str, Any]:
        """
        Anonymize text using Microsoft Presidio.

        Args:
            text: Input raw text.

        Returns:
            Dict with 'sanitized_text' and 'detected_entities'.
        """
        if not text or not text.strip():
            return {
                "sanitized_text": text or "",
                "detected_entities": [],
                "engine": "Microsoft Presidio",
            }

        if self._analyzer is None or self._anonymizer is None:
            return self._fallback_anonymize(text)

        try:
            results = self._analyzer.analyze(text=text, language="en")
            anonymized_result = self._anonymizer.anonymize(text=text, analyzer_results=results)
            detected = [
                {
                    "entity_type": res.entity_type,
                    "start": res.start,
                    "end": res.end,
                    "score": res.score,
                    "text": text[res.start:res.end],
                }
                for res in results
            ]
            return {
                "sanitized_text": anonymized_result.text,
                "detected_entities": detected,
                "engine": "Microsoft Presidio",
            }
        except Exception as exc:
            logger.error(f"Error during Presidio execution: {exc}")
            return self._fallback_anonymize(text)


class BaselineRedacted:
    """
    Rigid Redaction Baseline Anonymizer.
    
    Detects direct PII entities via Regex and Transformer NER and replaces
    each entity span with static [REDACTED] tags.
    """

    def __init__(self, stage1_anonymizer: Optional[Any] = None) -> None:
        """
        Args:
            stage1_anonymizer: Optional existing DirectPIIAnonymizer instance to share NER weights.
        """
        if stage1_anonymizer is not None:
            self._detector = stage1_anonymizer
        else:
            from src.stage1_direct_pii import DirectPIIAnonymizer
            self._detector = DirectPIIAnonymizer(use_ner=True)

    def anonymize(self, text: str, tag_format: str = "[REDACTED]") -> Dict[str, Any]:
        """
        Anonymize text with rigid [REDACTED] tags.

        Args:
            text: Input raw text.
            tag_format: Format string, e.g. '[REDACTED]' or '[REDACTED_{TYPE}]'.

        Returns:
            Dict with 'sanitized_text' and 'detected_entities'.
        """
        if not text or not text.strip():
            return {
                "sanitized_text": text or "",
                "detected_entities": [],
                "engine": "Rigid Redaction",
            }

        # Step 1: Detect entities via Stage 1 detector
        regex_matches = self._detector.detect_regex_entities(text)
        ner_matches = self._detector.detect_ner_entities(text)
        all_matches = self._detector._resolve_conflicts(regex_matches + ner_matches, text)

        # Step 2: Replace spans in reverse order
        chars = list(text)
        detected = []
        for m in sorted(all_matches, key=lambda x: x["start"], reverse=True):
            ent_type = m["entity_type"]
            start, end = m["start"], m["end"]
            replacement = tag_format.replace("{TYPE}", ent_type) if "{TYPE}" in tag_format else tag_format
            chars[start:end] = list(replacement)
            detected.append({
                "entity_value": m["entity_value"],
                "entity_type": ent_type,
                "start": start,
                "end": end,
                "replacement": replacement,
            })

        sanitized_text = "".join(chars)
        return {
            "sanitized_text": sanitized_text,
            "detected_entities": detected,
            "engine": "Rigid Redaction",
        }
