"""
SLM-Independent Deterministic Quasi-Identifier (QI) Detection & Abstraction Engine.

Provides independent, deterministic detection of contextual quasi-identifiers across:
1. Hyper-specific exact dates (e.g. "March 15, 2024", "2024-03-15", "15/03/2024")
2. Exact monetary values (e.g. "$45,382.19", "₹4,73,829", "€12,381.50")
3. Rare and specialized occupations with multi-signal specificity heuristics
4. Demographic & date-of-birth attributes (e.g. "DOB: 12/04/1982", "42-year-old male")
5. Highly precise street and facility locations
6. Compound / Relational Quasi-Identifiers (COMPOUND_QI) combining co-occurring attributes

Evaluates candidate rewrites for genuine partial mitigation, precision reduction, and
qualifier stripping, yielding deterministic residual risk scores in [0.0, 1.0].
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("QuasiIdentifierDetector")


class QuasiIdentifierDetector:
    """
    Deterministic Quasi-Identifier Detector and Residual Risk Evaluator.
    Operates independently of SLM output to ensure objective privacy gating.
    """

    DEFAULT_MITIGATION_THRESHOLD = 0.10
    DEFAULT_EXPOSURE_THRESHOLD = 0.90

    # Month name mapping for date granularity parsing
    MONTH_NAMES = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    # Role specificity keywords
    UNIQUENESS_MARKERS = {
        "sole", "only", "chief", "senior", "principal", "lead", "head",
        "exclusive", "regional", "forensic", "distinguished", "emeritus",
    }

    SPECIALTY_MODIFIERS = {
        "pediatric", "neonatal", "neurosurgeon", "neurooncology", "odontologist",
        "cardiothoracic", "immunogenetic", "cryptographic", "toxicologist",
        "epidemiologist", "radiologist", "subspecialist", "hematologist",
        "dipg", "glioblastoma", "sarcoma",
    }

    GENERIC_ROLES = {
        "ceo", "chief executive officer", "manager", "teacher", "nurse",
        "officer", "engineer", "worker", "clerk", "attendant", "driver",
        "assistant", "specialist", "doctor", "physician", "lawyer",
    }

    DEFAULT_MITIGATION_THRESHOLD = 0.10
    DEFAULT_EXPOSURE_THRESHOLD = 0.90
    DEFAULT_COMPOUND_ALPHA = 0.40

    def __init__(
        self,
        mitigation_threshold: float = DEFAULT_MITIGATION_THRESHOLD,
        exposure_threshold: float = DEFAULT_EXPOSURE_THRESHOLD,
        compound_alpha: float = DEFAULT_COMPOUND_ALPHA,
    ) -> None:
        """
        Initialize detector with configurable status classification boundaries
        and conservative compound QI aggregation parameters.

        Args:
            mitigation_threshold: Residual risk <= this is classified as 'mitigated' (default: 0.10).
            exposure_threshold: Residual risk >= this is classified as 'exposed' (default: 0.90).
            compound_alpha: Conservative weighting factor between mean component risk and joint product risk
                            (default: 0.40, ensures a single mitigated component does not force compound risk to 0).
        """
        self.mitigation_threshold = mitigation_threshold
        self.exposure_threshold = exposure_threshold
        self.compound_alpha = float(np.clip(compound_alpha, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # 1. Individual QI Category Detectors
    # -------------------------------------------------------------------------

    def detect_dates(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect hyper-specific exact dates (day-month-year, ISO, slash/dash formats).
        Excludes already-coarse temporal phrases like 'in 2024' or 'in early 2024'.
        """
        qis: List[Dict[str, Any]] = []
        if not text:
            return qis

        # Pattern 1: Month DD, YYYY (e.g., "March 15, 2024", "August 14, 2023")
        p1 = re.compile(
            r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
            re.IGNORECASE,
        )
        for m in p1.finditer(text):
            month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
            qis.append({
                "text": m.group(0).strip(),
                "type": "DATE",
                "start": m.start(),
                "end": m.end(),
                "risk_level": 1.0,
                "specificity": 0.95,
                "parsed": {
                    "day": int(day_str),
                    "month": month_str.lower(),
                    "year": int(year_str),
                },
            })

        # Pattern 2: DD Month YYYY (e.g., "15 March 2024", "14th August 2023")
        p2 = re.compile(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
            r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
            r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+(\d{4})\b",
            re.IGNORECASE,
        )
        for m in p2.finditer(text):
            day_str, month_str, year_str = m.group(1), m.group(2), m.group(3)
            # Avoid duplicate overlapping match
            if not any(q["start"] == m.start() and q["end"] == m.end() for q in qis):
                qis.append({
                    "text": m.group(0).strip(),
                    "type": "DATE",
                    "start": m.start(),
                    "end": m.end(),
                    "risk_level": 1.0,
                    "specificity": 0.95,
                    "parsed": {
                        "day": int(day_str),
                        "month": month_str.lower(),
                        "year": int(year_str),
                    },
                })

        # Pattern 3: YYYY-MM-DD or YYYY/MM/DD (ISO standard)
        p3 = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
        for m in p3.finditer(text):
            year_str, month_str, day_str = m.group(1), m.group(2), m.group(3)
            qis.append({
                "text": m.group(0).strip(),
                "type": "DATE",
                "start": m.start(),
                "end": m.end(),
                "risk_level": 1.0,
                "specificity": 0.95,
                "parsed": {
                    "day": int(day_str),
                    "month": str(int(month_str)),
                    "year": int(year_str),
                },
            })

        # Pattern 4: MM/DD/YYYY or DD/MM/YYYY
        p4 = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b")
        for m in p4.finditer(text):
            # Avoid matching inside ISO already captured
            if not any(q["start"] <= m.start() and q["end"] >= m.end() for q in qis):
                qis.append({
                    "text": m.group(0).strip(),
                    "type": "DATE",
                    "start": m.start(),
                    "end": m.end(),
                    "risk_level": 1.0,
                    "specificity": 0.95,
                    "parsed": {
                        "day": int(m.group(2)),
                        "month": m.group(1),
                        "year": int(m.group(3)),
                    },
                })

        return qis

    def detect_monetary_values(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect exact currency amounts ($45,382.19, €12,381.50, ₹4,73,829, $45,000).
        """
        qis: List[Dict[str, Any]] = []
        if not text:
            return qis

        # Symbols + formatted numbers
        p1 = re.compile(r"([$€£₹¥])\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
        for m in p1.finditer(text):
            symbol, val_str = m.group(1), m.group(2)
            # Remove commas and spaces for clean numeric representation
            numeric_val = val_str.replace(",", "").replace(" ", "")
            try:
                num = float(numeric_val)
                # Specificity: higher if contains non-zero cents or multiple exact digits
                has_cents = "." in numeric_val and not numeric_val.endswith(".00")
                specificity = 0.95 if has_cents or len(numeric_val.split(".")[0]) > 4 else 0.85
                qis.append({
                    "text": m.group(0).strip(),
                    "type": "MONEY",
                    "start": m.start(),
                    "end": m.end(),
                    "risk_level": 1.0,
                    "specificity": specificity,
                    "parsed": {
                        "symbol": symbol,
                        "numeric_value": num,
                        "raw_digits": numeric_val,
                    },
                })
            except ValueError:
                continue

        # Numbers with currency words (e.g., "45000 USD", "12,000 euros")
        p2 = re.compile(
            r"\b(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?|\d+)\s*(USD|EUR|GBP|INR|dollars|euros|pounds|rupees)\b",
            re.IGNORECASE,
        )
        for m in p2.finditer(text):
            if not any(q["start"] <= m.start() and q["end"] >= m.end() for q in qis):
                val_str, curr = m.group(1), m.group(2)
                numeric_val = val_str.replace(",", "").replace(" ", "")
                try:
                    num = float(numeric_val)
                    qis.append({
                        "text": m.group(0).strip(),
                        "type": "MONEY",
                        "start": m.start(),
                        "end": m.end(),
                        "risk_level": 0.95,
                        "specificity": 0.85,
                        "parsed": {
                            "symbol": curr.upper(),
                            "numeric_value": num,
                            "raw_digits": numeric_val,
                        },
                    })
                except ValueError:
                    continue

        return qis

    def detect_demographics(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect demographic identifiers: exact date-of-birth, exact age + gender combinations.
        """
        qis: List[Dict[str, Any]] = []
        if not text:
            return qis

        # Pattern 1: DOB / Birth Date marker
        p1 = re.compile(
            r"\b(?:DOB|Date of Birth|born(?:\s+on)?|birthdate)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        )
        for m in p1.finditer(text):
            qis.append({
                "text": m.group(0).strip(),
                "type": "DEMOGRAPHIC",
                "start": m.start(),
                "end": m.end(),
                "risk_level": 1.0,
                "specificity": 0.95,
                "parsed": {"subtype": "DOB", "value": m.group(1).strip()},
            })

        # Pattern 2: Age + Gender (e.g. "42-year-old male", "38 year old woman")
        p2 = re.compile(
            r"\b(\d{1,2})[-\s]year[-\s]old\s+(male|female|man|woman|boy|girl|patient|individual|person)\b",
            re.IGNORECASE,
        )
        for m in p2.finditer(text):
            qis.append({
                "text": m.group(0).strip(),
                "type": "DEMOGRAPHIC",
                "start": m.start(),
                "end": m.end(),
                "risk_level": 0.90,
                "specificity": 0.85,
                "parsed": {
                    "subtype": "AGE_GENDER",
                    "age": int(m.group(1)),
                    "gender": m.group(2).lower(),
                },
            })

        return qis

    def detect_rare_roles(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect rare/specialized occupations using a multi-signal specificity heuristic.
        Distinguishes rare roles (e.g. "sole Chief Pediatric Neurosurgeon specializing in DIPG")
        from generic professional titles (e.g. "Chief Executive Officer", "Teacher").
        """
        qis: List[Dict[str, Any]] = []
        if not text:
            return qis

        # Candidate pattern for professional roles with qualifiers
        pattern = re.compile(
            r"\b((?:the\s+)?(?:sole|only|chief|senior|lead|principal|head|exclusive|regional|forensic)?\s*"
            r"(?:pediatric|neonatal|cardiothoracic|forensic|neurological|orthopedic|cryptographic|investigative)?\s*"
            r"(?:neurosurgeon|odontologist|surgeon|physician|oncologist|specialist|manager|officer|director|counsel|investigator|scientist|architect|engineer)"
            r"(?:\s+(?:for|in|specializing in|focused on)\s+[A-Za-z\s]{3,35})?)\b",
            re.IGNORECASE,
        )

        for m in pattern.finditer(text):
            role_span = m.group(0).strip()
            role_lower = role_span.lower()
            words = set(re.findall(r"\w+", role_lower))

            # Multi-signal specificity score calculation
            has_uniqueness = bool(words & self.UNIQUENESS_MARKERS)
            has_specialty = bool(words & self.SPECIALTY_MODIFIERS)
            has_subclause = any(k in role_lower for k in ["specializing in", "for rare", "focused on"])
            is_generic = any(role_lower == g or role_lower == f"the {g}" for g in self.GENERIC_ROLES)

            # Score specificity from 0.0 to 1.0
            specificity = 0.0
            if has_uniqueness:
                specificity += 0.35
            if has_specialty:
                specificity += 0.40
            if has_subclause:
                specificity += 0.25
            if len(words) >= 4:
                specificity += 0.15

            if is_generic and not (has_specialty or has_subclause):
                specificity = min(specificity, 0.25)

            # Filter: only classify as high-risk QI if specificity is significant
            if specificity >= 0.50:
                qis.append({
                    "text": role_span,
                    "type": "ROLE",
                    "start": m.start(),
                    "end": m.end(),
                    "risk_level": round(min(1.0, specificity + 0.10), 2),
                    "specificity": round(min(1.0, specificity), 2),
                    "parsed": {
                        "tokens": list(words),
                        "has_uniqueness": has_uniqueness,
                        "has_specialty": has_specialty,
                    },
                })

        return qis

    def detect_precise_locations(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect highly specific street addresses and fine-grained facility locations.
        """
        qis: List[Dict[str, Any]] = []
        if not text:
            return qis

        # Street address pattern: number + street name + street suffix (with optional suite/floor/apt)
        p1 = re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
            r"Terrace|Ter|Place|Pl|Square|Sq|Parkway|Pkwy|Circle|Cir|Highway|Hwy)"
            r"(?:,\s*(?:Suite|Floor|Apt|Building|Ste|Fl)\s*#?\d+)?\b",
            re.IGNORECASE,
        )
        for m in p1.finditer(text):
            qis.append({
                "text": m.group(0).strip(),
                "type": "LOCATION",
                "start": m.start(),
                "end": m.end(),
                "risk_level": 0.95,
                "specificity": 0.90,
                "parsed": {"subtype": "STREET_ADDRESS"},
            })

        return qis

    # -------------------------------------------------------------------------
    # 2. Compound / Relational QI Detection
    # -------------------------------------------------------------------------

    def detect_compound_qis(
        self, single_qis: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """
        Discover co-occurring relational attribute combinations (COMPOUND_QI).
        E.g. DEMOGRAPHIC + ROLE, ROLE + LOCATION, ROLE + SPECIALIZATION, DATE + ROLE.
        """
        compound_qis: List[Dict[str, Any]] = []
        if len(single_qis) < 2:
            return compound_qis

        # Pairwise relational discovery
        valid_pairs = {
            ("DEMOGRAPHIC", "ROLE"),
            ("ROLE", "LOCATION"),
            ("DATE", "ROLE"),
            ("DEMOGRAPHIC", "LOCATION"),
            ("MONEY", "ROLE"),
        }

        for i in range(len(single_qis)):
            for j in range(i + 1, len(single_qis)):
                qi_a = single_qis[i]
                qi_b = single_qis[j]
                pair_type = (qi_a["type"], qi_b["type"])
                rev_pair = (qi_b["type"], qi_a["type"])

                if pair_type in valid_pairs or rev_pair in valid_pairs:
                    # Combine into a Compound QI
                    combined_text = f"{qi_a['text']} + {qi_b['text']}"
                    joint_risk = round(min(1.0, (qi_a["risk_level"] + qi_b["risk_level"]) / 1.8), 2)

                    compound_qis.append({
                        "text": combined_text,
                        "type": "COMPOUND_QI",
                        "start": min(qi_a["start"], qi_b["start"]),
                        "end": max(qi_a["end"], qi_b["end"]),
                        "risk_level": joint_risk,
                        "specificity": round(max(qi_a["specificity"], qi_b["specificity"]), 2),
                        "components": [qi_a, qi_b],
                    })

        return compound_qis

    def detect_quasi_identifiers(self, text: str) -> List[Dict[str, Any]]:
        """
        Master detection pipeline: extracts all single QIs and compound QIs from text.
        """
        if not text or not text.strip():
            return []

        single_qis: List[Dict[str, Any]] = []
        single_qis.extend(self.detect_dates(text))
        single_qis.extend(self.detect_monetary_values(text))
        single_qis.extend(self.detect_demographics(text))
        single_qis.extend(self.detect_rare_roles(text))
        single_qis.extend(self.detect_precise_locations(text))

        # Deduplicate overlapping single spans
        deduped: List[Dict[str, Any]] = []
        for item in sorted(single_qis, key=lambda x: x["start"]):
            if not any(
                d["type"] == item["type"]
                and max(d["start"], item["start"]) < min(d["end"], item["end"])
                for d in deduped
            ):
                deduped.append(item)

        # Detect compound relational QIs
        compound_qis = self.detect_compound_qis(deduped, text)

        all_qis = deduped + compound_qis
        logger.debug(f"Detected {len(deduped)} single QIs and {len(compound_qis)} compound QIs.")
        return all_qis

    # -------------------------------------------------------------------------
    # 3. Dynamic Residual Risk & Genuine Partial Mitigation Evaluation
    # -------------------------------------------------------------------------

    def evaluate_qi_mitigation(
        self,
        qi_item: Dict[str, Any],
        candidate_text: str,
    ) -> Dict[str, Any]:
        """
        Evaluate residual risk r_i in [0.0, 1.0] for a detected QI in candidate_text.

        Distinguishes:
        - Exact exposure: r = 1.0 (status = 'exposed')
        - Partial mitigation: 0.0 < r < 1.0 (status = 'partial')
        - Full mitigation: r <= mitigation_threshold (status = 'mitigated')
        """
        qi_type = qi_item.get("type", "UNKNOWN")
        orig_text = qi_item.get("text", "")
        cand_lower = candidate_text.lower()
        orig_lower = orig_text.lower()

        # Handle Compound QIs with conservative convex aggregation rule:
        # r_compound = alpha * mean(r_k) + (1 - alpha) * prod(r_k)
        # Prevents a single mitigated component from forcing the entire compound risk to 0.
        if qi_type == "COMPOUND_QI":
            components = qi_item.get("components", [])
            comp_evals = [self.evaluate_qi_mitigation(c, candidate_text) for c in components]
            comp_risks = [e["residual_risk"] for e in comp_evals]
            if comp_risks:
                r_mean = float(np.mean(comp_risks))
                r_prod = float(np.prod(comp_risks))
                residual_risk = (self.compound_alpha * r_mean) + ((1.0 - self.compound_alpha) * r_prod)
            else:
                residual_risk = 0.0
            residual_risk = float(np.clip(residual_risk, 0.0, 1.0))

            status = self._classify_status(residual_risk)
            return {
                "text": orig_text,
                "type": "COMPOUND_QI",
                "initial_risk": qi_item.get("risk_level", 1.0),
                "residual_risk": round(residual_risk, 4),
                "status": status,
                "component_details": comp_evals,
            }

        # ---------------------------------------------------------------------
        # Evaluation by Single QI Type
        # ---------------------------------------------------------------------

        # 1. Exact Dates
        if qi_type == "DATE":
            # Check exact string match
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                parsed = qi_item.get("parsed", {})
                day = str(parsed.get("day", ""))
                month = str(parsed.get("month", "")).lower()
                year = str(parsed.get("year", ""))

                has_year = year in cand_lower if year else False
                has_month = month in cand_lower if month else False
                has_day = bool(day and re.search(rf"\b{day}(?:st|nd|rd|th)?\b", cand_lower))

                # Measurable date granularity reduction:
                # 3/3 components present -> 1.0 (exposed)
                # Month + Year present (day removed) -> 2/3 ≈ 0.67 (partial)
                # Year only present -> 1/3 ≈ 0.33 (partial)
                # No specific components present (coarsened to qualitative e.g. "early 2024" or removed) -> 0.0
                if has_day and has_month and has_year:
                    residual_risk = 1.0
                elif has_month and has_year:
                    residual_risk = 0.67
                elif has_year:
                    # If only year remains, check if qualitative modifier was added (e.g. "in early 2024")
                    is_qualitative = any(q in cand_lower for q in ["early", "late", "mid", "spring", "summer", "fall", "winter"])
                    residual_risk = 0.25 if is_qualitative else 0.33
                else:
                    residual_risk = 0.0

        # 2. Exact Monetary Values
        elif qi_type == "MONEY":
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                parsed = qi_item.get("parsed", {})
                raw_digits = parsed.get("raw_digits", "")
                num_val = parsed.get("numeric_value", 0.0)

                # Check if exact raw digits survive
                if raw_digits and raw_digits in cand_lower.replace(",", ""):
                    residual_risk = 1.0
                elif num_val > 0:
                    # Check if rounded or approximate number appears
                    clean_cand_nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", candidate_text) if n]
                    # Check relative distance to original money value
                    close_matches = [n for n in clean_cand_nums if abs(n - num_val) / max(num_val, 1) < 0.20]
                    if close_matches:
                        # Approximate numeric value preserved -> partial mitigation (e.g. $45,000 for $45,382)
                        residual_risk = 0.40
                    else:
                        # Completely converted to non-numeric coarsened text ("a five-figure wire")
                        residual_risk = 0.0
                else:
                    residual_risk = 0.0

        # 3. Rare / Specialized Occupations
        elif qi_type == "ROLE":
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                orig_words = set(re.findall(r"\w+", orig_lower)) - {"a", "an", "the", "in", "of", "for", "at", "to", "and"}
                cand_words = set(re.findall(r"\w+", cand_lower))

                if not orig_words:
                    residual_risk = 0.0
                else:
                    # Weighted modifier overlap calculation
                    total_weight = 0.0
                    retained_weight = 0.0

                    for w in orig_words:
                        weight = 1.0
                        if w in self.UNIQUENESS_MARKERS:
                            weight = 2.0
                        elif w in self.SPECIALTY_MODIFIERS:
                            weight = 2.5
                        elif w in self.GENERIC_ROLES:
                            weight = 0.5

                        total_weight += weight
                        if w in cand_words:
                            retained_weight += weight

                    residual_risk = retained_weight / max(total_weight, 1.0)
                    residual_risk = float(np.clip(residual_risk, 0.0, 1.0))

        # 4. Demographic & DOB
        elif qi_type == "DEMOGRAPHIC":
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                parsed = qi_item.get("parsed", {})
                subtype = parsed.get("subtype", "")

                if subtype == "DOB":
                    val = str(parsed.get("value", "")).lower()
                    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", val)
                    has_birth_year = year_match.group(1) in cand_lower if year_match else False
                    if has_birth_year:
                        # Birth year only retained -> partial
                        residual_risk = 0.33
                    else:
                        # Coarsened to "in their 40s" or removed
                        residual_risk = 0.0
                elif subtype == "AGE_GENDER":
                    age = str(parsed.get("age", ""))
                    gender = parsed.get("gender", "")
                    has_age = bool(age and re.search(rf"\b{age}\b", cand_lower))
                    has_gender = gender in cand_lower if gender else False

                    if has_age and has_gender:
                        residual_risk = 1.0
                    elif has_age:
                        residual_risk = 0.60
                    elif has_gender:
                        residual_risk = 0.15
                    else:
                        residual_risk = 0.0
                else:
                    residual_risk = 0.0

        # 5. Precise Locations
        elif qi_type == "LOCATION":
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                orig_words = set(re.findall(r"\w+", orig_lower)) - {"a", "an", "the", "in", "at", "suite", "floor", "st", "ave", "rd"}
                cand_words = set(re.findall(r"\w+", cand_lower))
                overlap = len(orig_words & cand_words)
                residual_risk = overlap / max(len(orig_words), 1) if orig_words else 0.0
                residual_risk = float(np.clip(residual_risk, 0.0, 1.0))

        # Default fallback
        else:
            if orig_lower in cand_lower:
                residual_risk = 1.0
            else:
                orig_words = set(re.findall(r"\w+", orig_lower))
                cand_words = set(re.findall(r"\w+", cand_lower))
                overlap = len(orig_words & cand_words)
                residual_risk = overlap / max(len(orig_words), 1) if orig_words else 0.0
                residual_risk = float(np.clip(residual_risk, 0.0, 1.0))

        residual_risk = float(np.clip(residual_risk, 0.0, 1.0))
        status = self._classify_status(residual_risk)

        return {
            "text": orig_text,
            "type": qi_type,
            "initial_risk": qi_item.get("risk_level", 1.0),
            "residual_risk": round(residual_risk, 4),
            "status": status,
        }

    def _classify_status(self, residual_risk: float) -> str:
        """Assign categorical status using configurable thresholds."""
        if residual_risk <= self.mitigation_threshold:
            return "mitigated"
        elif residual_risk >= self.exposure_threshold:
            return "exposed"
        else:
            return "partial"
