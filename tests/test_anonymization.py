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
        cls.defense = SemanticQuasiIdentifierDefense(
            ollama_model="qwen2.5:1.5b",
            tau=0.72,
            floor_sim=0.60,
            request_timeout=2,
        )

    def test_quasi_identifier_candidate_generalization(self):
        text = (
            "Eleanor works as the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors "
            "in Baltimore, Maryland. DOB: 12/04/1982."
        )
        res = self.defense.generalize_text(text)
        
        # Verify output contains structured telemetry
        self.assertIsInstance(res["candidate_text"], str)
        self.assertIn("is_accepted", res)
        self.assertIn("composite_score", res)
        self.assertIn("metrics_breakdown", res)
        self.assertIn("thresholds", res)
        self.assertIn("Ollama LLM", res["backend_used"])

    def test_full_document_guardrail_pass(self):
        doc = (
            "Patient Eleanor Vance visited Dr. Robert Langdon at Johns Hopkins Hospital on August 14, 2023. "
            "She can be reached at eleanor.vance@medmail.org or (410) 555-0199. "
            "Eleanor works as the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors in Baltimore, Maryland."
        )
        res = self.defense.generalize_text(doc)
        self.assertIsInstance(res["final_text"], str)
        self.assertGreaterEqual(res["composite_score"], 0.0)
        self.assertIn("semantic_similarity", res["metrics_breakdown"])
        self.assertIn("privacy_reduction_score", res["metrics_breakdown"])
        self.assertIn("readability_score", res["metrics_breakdown"])

    def test_drift_guardrail_fallback_trigger(self):
        """When tau is strictly unachievable (e.g. 0.999), fallback to Stage 1 text."""
        strict_defense = self.defense
        old_thresh = strict_defense.tau
        try:
            strict_defense.tau = 0.999
            text = "Subject is the sole Chief Pediatric Neurosurgeon for Rare Brain Stem Tumors."
            res = strict_defense.generalize_text(text)
            if not res["is_accepted"]:
                self.assertEqual(res["final_text"], text)
                self.assertTrue(res["fallback_triggered"])
        finally:
            strict_defense.tau = old_thresh


class TestCompositeDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.defense = SemanticQuasiIdentifierDefense(
            tau=0.70,
            floor_sim=0.60,
            weights={"sim": 0.50, "priv": 0.30, "read": 0.20},
            embedder_model_name="",
            request_timeout=1,
        )

    def test_privacy_reduction_metric_calculation(self):
        # 1. No modifications flagged -> default 1.0
        score_empty = self.defense.compute_privacy_reduction("Some candidate text", [])
        self.assertEqual(score_empty, 1.0)

        # 2. Flagged phrases completely removed/abstracted -> 1.0
        mods = [
            {"original_span": "$45,000", "generalized_span": "a five-figure wire"},
            {"original_span": "Chief Neurosurgeon", "generalized_span": "Medical Specialist"},
        ]
        cand_clean = "Transfer of a five-figure wire approved by Medical Specialist."
        score_full = self.defense.compute_privacy_reduction(cand_clean, mods)
        self.assertEqual(score_full, 1.0)

        # 3. Partial removal (1 of 2 phrases remains) -> 0.5
        cand_partial = "Transfer of $45,000 approved by Medical Specialist."
        score_partial = self.defense.compute_privacy_reduction(cand_partial, mods)
        self.assertEqual(score_partial, 0.5)

        # 4. Zero removal (both remain) -> 0.0
        cand_none = "Transfer of $45,000 approved by Chief Neurosurgeon."
        score_none = self.defense.compute_privacy_reduction(cand_none, mods)
        self.assertEqual(score_none, 0.0)

    def test_readability_metric_calculation(self):
        orig = "The patient underwent surgery at Johns Hopkins on March 15, 2024."
        cand_good = "The patient underwent surgery at a regional hospital in early 2024."
        
        score_good = self.defense.compute_readability_score(orig, cand_good)
        self.assertGreater(score_good, 0.80)
        self.assertLessEqual(score_good, 1.0)

        # Severe truncation penalty
        cand_truncated = "The patient"
        score_trunc = self.defense.compute_readability_score(orig, cand_truncated)
        self.assertLess(score_trunc, score_good)

        # Unbalanced quotes / syntax penalty
        cand_unbalanced = 'The patient underwent "surgery at hospital in early 2024.'
        score_unbalanced = self.defense.compute_readability_score(orig, cand_unbalanced)
        self.assertLess(score_unbalanced, score_good)

    def test_composite_decision_acceptance(self):
        stage1 = (
            "The patient is admitted to the hospital for treatment. "
            "The procedure cost is $45,000 and the attending doctor is Chief Pediatric Neurosurgeon "
            "scheduled on August 14, 2023."
        )
        cand = (
            "The patient is admitted to the hospital for treatment. "
            "The procedure cost is a five-figure sum and the attending doctor is Medical Specialist "
            "scheduled in late summer 2023."
        )
        mods = [
            {"original_span": "$45,000"},
            {"original_span": "Chief Pediatric Neurosurgeon"},
            {"original_span": "August 14, 2023"},
        ]

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand,
            modifications=mods,
            tau=0.70,
            floor_sim=0.60,
        )

        self.assertTrue(decision["is_accepted"])
        self.assertFalse(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], cand)
        self.assertGreaterEqual(decision["composite_score"], 0.70)
        self.assertGreaterEqual(decision["metrics_breakdown"]["semantic_similarity"], 0.60)
        self.assertEqual(decision["metrics_breakdown"]["privacy_reduction_score"], 1.0)

    def test_hard_similarity_floor_rejection(self):
        """Even if composite score is high, semantic similarity below floor_sim MUST trigger fallback."""
        stage1 = "The patient was prescribed amoxicillin 500mg twice daily."
        # Completely off-topic candidate
        cand = "Quantum computers leverage superposition to calculate prime factors."
        mods = [{"original_span": "amoxicillin 500mg"}]

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand,
            modifications=mods,
            tau=0.50,
            floor_sim=0.60,  # hard floor
        )

        self.assertFalse(decision["is_accepted"])
        self.assertTrue(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], stage1)

    def test_custom_weights_and_telemetry_structure(self):
        stage1 = "Report filed on May 4, 2022."
        cand = "Report filed in spring 2022."
        mods = [{"original_span": "May 4, 2022"}]

        custom_weights = {"sim": 0.40, "priv": 0.40, "read": 0.20}
        res = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand,
            modifications=mods,
            weights=custom_weights,
        )

        self.assertEqual(res["thresholds"]["weights"]["sim"], 0.40)
        self.assertEqual(res["thresholds"]["weights"]["priv"], 0.40)
        self.assertEqual(res["thresholds"]["weights"]["read"], 0.20)
        self.assertIn("composite_score", res)
        self.assertIn("semantic_similarity", res["metrics_breakdown"])
        self.assertIn("privacy_reduction_score", res["metrics_breakdown"])
        self.assertIn("readability_score", res["metrics_breakdown"])





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
