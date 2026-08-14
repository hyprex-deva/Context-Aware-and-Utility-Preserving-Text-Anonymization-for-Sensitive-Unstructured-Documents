"""
Stage 1: Token-Level Direct PII Detection & Deterministic Synthetic Surrogate Replacement.

This module combines compiled regular expressions for structured PII (emails, phones,
SSNs, IP addresses, credit cards) with Transformer-based Named Entity Recognition (NER)
for unstructured entity spans (PER, ORG, LOC, MISC). Detected entities are replaced with
realistic synthetic surrogates generated via Faker while guaranteeing document-level
consistency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from faker import Faker
    HAS_FAKER = True
except ImportError:
    Faker = None
    HAS_FAKER = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Stage1DirectPII")



@dataclass
class DetectedEntity:
    """Represents a detected PII entity with its text span and surrogate."""
    entity_value: str
    entity_type: str
    start: int
    end: int
    surrogate_value: str
    detector: str  # 'regex' or 'transformer_ner'
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_value": self.entity_value,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "surrogate_value": self.surrogate_value,
            "detector": self.detector,
            "score": round(self.score, 4),
        }


class DirectPIIAnonymizer:
    """
    Two-pronged token-level direct PII detection and synthetic replacement.
    
    Features:
    - Compiled high-precision Regex patterns for structured identifiers.
    - Hugging Face Transformer NER pipeline for unstructured entities.
    - Deterministic, document-level consistent Faker surrogate mapping.
    - Offset-aware span collision resolver preventing overlapping replacements.
    """

    # Compiled regex patterns for structured PII (IPv4/IPv6, Emails, SSN, ID Cards, Licenses, Passwords)
    REGEX_PATTERNS = {
        "EMAIL": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:\.[A-Za-z]{2,}|\b)",
            re.IGNORECASE,
        ),
        "PHONE": re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
        ),
        "SSN": re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s\d{3}\s\d{4}\b"
        ),
        "IP_ADDRESS": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b|"
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
            r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:[0-9a-fA-F]{1,4}\b|"
            r"\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
        ),
        "CREDIT_CARD": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b"
        ),
        "LICENSE": re.compile(
            r"(?i)(?:License|Driver(?:'s)?\s*License|Driving\s*License)\s*[:#\-]?\s*([A-Za-z0-9.\-\s]{6,30})"
        ),
        "ID_CARD": re.compile(
            r"(?i)(?:ID\s*Card(?:\s*Number)?|National\s*ID|ID\s*Doc(?:ument)?|ID\s*Card)\s*[:#\-]?\s*([A-Za-z0-9]{5,20})"
        ),
        "PASSWORD": re.compile(
            r'(?i)(?:Password|Passphrase|Passwd|Secret\s*Key)\s*[:#\-]?\s*([^\s\n]{4,30})'
        ),
    }


    # Standard NER entity label mapping to canonical tags
    LABEL_NORMALIZATION = {
        "B-PER": "PER", "I-PER": "PER", "PER": "PER", "PERSON": "PER",
        "B-ORG": "ORG", "I-ORG": "ORG", "ORG": "ORG", "ORGANIZATION": "ORG",
        "B-LOC": "LOC", "I-LOC": "LOC", "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC",
        "B-MISC": "MISC", "I-MISC": "MISC", "MISC": "MISC",
    }

    def __init__(
        self,
        ner_model_name: str = "dslim/bert-base-NER",
        device: Optional[int] = -1,
        faker_seed: Optional[int] = 42,
        use_ner: bool = True,
    ) -> None:
        """
        Initialize the Direct PII Anonymizer.

        Args:
            ner_model_name: Hugging Face model identifier for token classification.
            device: Computing device (-1 for CPU, 0+ for CUDA GPU ID).
            faker_seed: Optional seed for reproducible surrogate generation.
            use_ner: Flag to toggle Hugging Face NER pipeline loading.
        """
        self.ner_model_name = ner_model_name
        self.device = device
        self.use_ner = use_ner
        if HAS_FAKER and Faker is not None:
            self.faker = Faker()
            if faker_seed is not None:
                self.faker.seed_instance(faker_seed)
        else:
            self.faker = None

        self._ner_pipeline = None
        if self.use_ner:
            self._init_ner_pipeline()

    def _init_ner_pipeline(self) -> None:
        """Lazy load and initialize Hugging Face NER pipeline with fallback."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            logger.info(f"Loading NER pipeline with model: {self.ner_model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name)
            self._ner_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device,
            )
            logger.info("Hugging Face NER pipeline successfully initialized.")
        except Exception as exc:
            logger.warning(
                f"Failed to load Transformer NER model ({exc}). "
                "Falling back to Regex-only mode. (Install transformers/torch and ensure model download)."
            )
            self._ner_pipeline = None

    def _generate_surrogate(self, entity_type: str, raw_value: str) -> str:
        """
        Generate a realistic synthetic surrogate corresponding to entity type.

        Args:
            entity_type: Canonical entity type string (PER, ORG, LOC, EMAIL, etc.).
            raw_value: The original entity string for context-aware shape matching.

        Returns:
            A realistic synthetic replacement string.
        """
        entity_type_upper = entity_type.upper()
        
        if self.faker is not None:
            if entity_type_upper in ("PER", "PERSON"):
                parts = raw_value.strip().split()
                if len(parts) == 1:
                    return self.faker.first_name()
                return self.faker.name()
            
            elif entity_type_upper in ("ORG", "ORGANIZATION"):
                return self.faker.company()
            
            elif entity_type_upper in ("LOC", "LOCATION", "GPE"):
                if any(char.isdigit() for char in raw_value):
                    return self.faker.street_address()
                return self.faker.city()
            
            elif entity_type_upper == "EMAIL":
                domain = self.faker.free_email_domain()
                user_name = self.faker.user_name()
                return f"{user_name}@{domain}"
            
            elif entity_type_upper == "PHONE":
                return self.faker.phone_number()
            
            elif entity_type_upper == "SSN":
                return self.faker.ssn()
            
            elif entity_type_upper == "IP_ADDRESS":
                if ":" in raw_value:
                    return self.faker.ipv6()
                return self.faker.ipv4()
            
            elif entity_type_upper in ("ID_CARD", "IDCARD", "NATIONAL_ID", "ID"):
                return self.faker.bothify(text="??#####??").upper()
            
            elif entity_type_upper in ("LICENSE", "DRIVER_LICENSE", "DRIVERLICENSE"):
                return self.faker.bothify(text="?????.######.??.###").upper()
            
            elif entity_type_upper in ("PASSWORD", "PASS", "PASSWD"):
                return self.faker.password(length=8)

            elif entity_type_upper == "CREDIT_CARD":
                return self.faker.credit_card_number(card_type=None)
            
            elif entity_type_upper == "MISC":
                return self.faker.catch_phrase()

        # Deterministic fallback surrogates if Faker is not active
        fallback_map = {
            "PER": "Alex Morgan",
            "PERSON": "Alex Morgan",
            "ORG": "Apex Global Solutions",
            "ORGANIZATION": "Apex Global Solutions",
            "LOC": "Riverdale",
            "LOCATION": "Riverdale",
            "GPE": "Riverdale",
            "EMAIL": "synthetic.user@example.org",
            "PHONE": "555-019-9832",
            "SSN": "987-65-4321",
            "IP_ADDRESS": "10.0.0.1",
            "CREDIT_CARD": "4000-1234-5678-9010",
            "ID_CARD": "AB12345CD",
            "LICENSE": "ABCDE.123456.FG.789",
            "PASSWORD": "SecurePass!123",
            "MISC": "Enterprise Asset",
        }
        return fallback_map.get(entity_type_upper, f"SYNTHETIC_{entity_type_upper}")

    def detect_regex_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect structured PII entities using compiled regex patterns.

        Args:
            text: Input document text.

        Returns:
            List of raw detected entity dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for pii_type, pattern in self.REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                # If regex contains capture groups (e.g. ID Card value), extract group 1
                if match.groups():
                    val = match.group(1).strip()
                    start, end = match.start(1), match.end(1)
                else:
                    val = match.group().strip()
                    start, end = match.start(), match.end()
                
                # Sanity filters
                if pii_type == "PHONE" and len(re.sub(r"\D", "", val)) < 7:
                    continue
                if not val or len(val) < 2:
                    continue

                results.append({
                    "entity_value": val,
                    "entity_type": pii_type,
                    "start": start,
                    "end": end,
                    "detector": "regex",
                    "score": 1.0,
                })
        return results


    IGNORE_NER_TOKENS = {
        "ss", "ssn", "id", "dob", "bar", "bar id", "ip", "phone", "cc",
        "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "prof", "prof.",
        "cfo", "ceo", "hr", "vp", "the", "a", "an", "at", "in", "on", "of", "to", "for"
    }

    def detect_ner_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect unstructured entities using Transformer NER pipeline.

        Args:
            text: Input document text.

        Returns:
            List of raw detected entity dictionaries.
        """
        if not self._ner_pipeline or not text.strip():
            return []
        
        results: List[Dict[str, Any]] = []
        try:
            ner_predictions = self._ner_pipeline(text)
            for item in ner_predictions:
                raw_label = item.get("entity_group") or item.get("entity") or "MISC"
                canonical_label = self.LABEL_NORMALIZATION.get(raw_label, "MISC")
                span_text = text[item["start"]:item["end"]].strip()
                
                # Filter out pure punctuation, single characters, or known acronyms
                if not span_text or len(span_text) < 2:
                    continue
                if span_text.lower() in self.IGNORE_NER_TOKENS:
                    continue

                results.append({
                    "entity_value": span_text,
                    "entity_type": canonical_label,
                    "start": item["start"],
                    "end": item["end"],
                    "detector": "transformer_ner",
                    "score": float(item.get("score", 0.95)),
                })
        except Exception as exc:
            logger.error(f"Error during Transformer NER inference: {exc}")
        
        return results


    def _resolve_conflicts(
        self, raw_candidates: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """
        Resolve overlapping spans between Regex and NER detectors.
        
        Priority rules:
        1. Exact/Enclosing structured Regex matches take priority over NER partial spans.
        2. Longer character span is preferred on tie.
        3. Higher detection confidence score is preferred.
        """
        if not raw_candidates:
            return []

        # Sort candidates: start index ascending, length descending, score descending
        sorted_candidates = sorted(
            raw_candidates,
            key=lambda x: (x["start"], -(x["end"] - x["start"]), -x["score"]),
        )

        resolved: List[Dict[str, Any]] = []
        occupied_indices: Set[int] = set()

        for cand in sorted_candidates:
            start, end = cand["start"], cand["end"]
            span_indices = set(range(start, end))

            # If this span overlaps with any previously accepted span
            if span_indices & occupied_indices:
                continue

            occupied_indices.update(span_indices)
            resolved.append(cand)

        # Sort final resolved entities by start offset ascending
        resolved.sort(key=lambda x: x["start"])
        return resolved

    def anonymize(self, text: str) -> Dict[str, Any]:
        """
        Execute token-level direct PII detection and replace entities with
        deterministic synthetic surrogates with document-level consistency.

        Args:
            text: Unstructured raw input text.

        Returns:
            Dict containing:
            - 'sanitized_text': Text with all direct PII replaced with synthetic surrogates.
            - 'detected_entities': List of DetectedEntity dictionaries.
            - 'entity_map': Mapping of original text to synthetic surrogate.
        """
        if not text or not text.strip():
            return {
                "sanitized_text": text or "",
                "detected_entities": [],
                "entity_map": {},
            }

        # Step 1: Detect raw entities via Regex and Transformer NER
        regex_matches = self.detect_regex_entities(text)
        ner_matches = self.detect_ner_entities(text)
        all_matches = regex_matches + ner_matches

        # Step 2: Resolve overlapping spans
        resolved_matches = self._resolve_conflicts(all_matches, text)

        # Step 3: Deterministic Surrogate Mapping with document-level cache
        # key format: (canonical_entity_type, lower_entity_value)
        entity_map: Dict[Tuple[str, str], str] = {}
        detected_entities: List[DetectedEntity] = []

        for match in resolved_matches:
            raw_val = match["entity_value"]
            ent_type = match["entity_type"]
            cache_key = (ent_type, raw_val.strip().lower())

            if cache_key in entity_map:
                surrogate = entity_map[cache_key]
            else:
                surrogate = self._generate_surrogate(ent_type, raw_val)
                entity_map[cache_key] = surrogate

            detected_entities.append(
                DetectedEntity(
                    entity_value=raw_val,
                    entity_type=ent_type,
                    start=match["start"],
                    end=match["end"],
                    surrogate_value=surrogate,
                    detector=match["detector"],
                    score=match["score"],
                )
            )

        # Step 4: Construct sanitized text by replacing spans in reverse order
        # Reverse order guarantees that subsequent replacement indices remain valid
        sanitized_chars = list(text)
        for entity in sorted(detected_entities, key=lambda e: e.start, reverse=True):
            sanitized_chars[entity.start:entity.end] = list(entity.surrogate_value)

        sanitized_text = "".join(sanitized_chars)

        # Format clean entity map output (original_string -> surrogate)
        formatted_entity_map = {
            f"{k[0]}:{k[1]}": v for k, v in entity_map.items()
        }

        return {
            "sanitized_text": sanitized_text,
            "detected_entities": [e.to_dict() for e in detected_entities],
            "entity_map": formatted_entity_map,
        }
