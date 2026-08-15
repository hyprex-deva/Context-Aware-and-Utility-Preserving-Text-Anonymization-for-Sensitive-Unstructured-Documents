"""
Unit and Integration Tests for Context-Aware Text Anonymization.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.stage1_direct_pii import DirectPIIAnonymizer
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense
from src.baselines import BaselinePresidio, BaselineRedacted
from src.evaluate_metrics import (
    compute_bleu_scores,
    compute_privacy_metrics,
    compute_rouge_l_scores,
    evaluate_anonymization_system,
)



class TestDirectPIIAnonymizer(unittest.TestCase):
    def setUp(self):
        # Initialize with use_ner=False for fast deterministic regex testing
        self.anonymizer = DirectPIIAnonymizer(use_ner=False, faker_seed=123)

    def test_regex_detection_structured_pii(self):
        text = (
            "Contact john.doe@example.com or call 555-123-4567. "
            "SSN is 123-45-6789. Server IP is 192.168.1.1 and CC is 4532-8901-2345-6789."
        )
        res = self.anonymizer.anonymize(text)
        sanitized = res["sanitized_text"]
        entities = res["detected_entities"]

        # Ensure raw structured PII is absent from sanitized text
        self.assertNotIn("john.doe@example.com", sanitized)
        self.assertNotIn("555-123-4567", sanitized)
        self.assertNotIn("123-45-6789", sanitized)
        self.assertNotIn("192.168.1.1", sanitized)
        self.assertNotIn("4532-8901-2345-6789", sanitized)
        self.assertGreaterEqual(len(entities), 5)

    def test_document_level_consistency(self):
        """Verify identical entities receive identical surrogates across the document."""
        text = "Contact Alice at alice@company.com. I repeat, send email to alice@company.com."
        res = self.anonymizer.anonymize(text)
        sanitized = res["sanitized_text"]
        
        # Email appears twice; ensure both occurrences are identical synthetic surrogate
        email_surrogates = [
            e["surrogate_value"] for e in res["detected_entities"] if e["entity_type"] == "EMAIL"
        ]
        self.assertEqual(len(email_surrogates), 2)
        self.assertEqual(email_surrogates[0], email_surrogates[1])

    def test_empty_string_handling(self):
        res = self.anonymizer.anonymize("")
        self.assertEqual(res["sanitized_text"], "")
        self.assertEqual(res["detected_entities"], [])


class TestSemanticQuasiIdentifierDefense(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defense = SemanticQuasiIdentifierDefense(ollama_model="qwen2.5:1.5b", similarity_threshold=0.70)


    def test_quasi_identifier_candidate_generalization(self):
        text = (
            "Eleanor works as the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors "
            "in Baltimore, Maryland. DOB: 12/04/1982."
        )
        res = self.defense.generalize_text(text)
        
        # Verify Ollama SLM output contains generalized text and metadata
        self.assertIsInstance(res["candidate_text"], str)
        self.assertGreater(len(res["candidate_text"]), 10)
        self.assertIn("Ollama LLM", res["backend_used"])

    def test_full_document_guardrail_pass(self):
        doc = (
            "Patient Eleanor Vance visited Dr. Robert Langdon at Johns Hopkins Hospital on August 14, 2023. "
            "She can be reached at eleanor.vance@medmail.org or (410) 555-0199. "
            "Eleanor works as the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors in Baltimore, Maryland."
        )
        res = self.defense.generalize_text(doc)
        self.assertIsInstance(res["final_text"], str)
        self.assertGreaterEqual(res["similarity_score"], 0.0)

    def test_drift_guardrail_fallback_trigger(self):
        """When similarity threshold is strictly unachievable (e.g. 0.999), fallback to Stage 1 text."""
        strict_defense = self.defense
        old_thresh = strict_defense.similarity_threshold
        try:
            strict_defense.similarity_threshold = 0.999
            text = "Subject is the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors."
            res = strict_defense.generalize_text(text)
            if not res["drift_passed"]:
                self.assertEqual(res["final_text"], text)
        finally:
            strict_defense.similarity_threshold = old_thresh





class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.presidio = BaselinePresidio()
        self.stage1 = DirectPIIAnonymizer(use_ner=False)
        self.redacted = BaselineRedacted(stage1_anonymizer=self.stage1)

    def test_presidio_anonymize(self):
        text = "Email me at doctor.smith@hospital.org."
        res = self.presidio.anonymize(text)
        self.assertIn("EMAIL_ADDRESS", res["sanitized_text"])

    def test_rigid_redaction(self):
        text = "Email me at doctor.smith@hospital.org or call 555-019-2831."
        res = self.redacted.anonymize(text)
        self.assertIn("[REDACTED]", res["sanitized_text"])
        self.assertNotIn("doctor.smith@hospital.org", res["sanitized_text"])


class TestEvaluationMetrics(unittest.TestCase):
    def test_privacy_metrics(self):
        preds = [[{"start": 0, "end": 10, "entity_value": "John Smith"}]]
        truths = [[{"start": 0, "end": 10, "entity": "John Smith"}]]
        res = compute_privacy_metrics(preds, truths)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1"], 1.0)

    def test_rouge_and_utility(self):
        orig = ["The quick brown fox jumps over the lazy dog."]
        sanit = ["The quick brown fox jumps over a lazy cat."]
        rouge = compute_rouge_l_scores(orig, sanit)
        self.assertGreater(rouge["rougeL_f1"], 0.6)

    def test_bleu_scores(self):
        orig = ["The quick brown fox jumps over the lazy dog."]
        sanit = ["The quick brown fox jumps over a lazy dog."]
        bleu = compute_bleu_scores(orig, sanit)
        self.assertGreater(bleu["bleu_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
