"""
Unit tests specifically for Stage 2 Composite Decision Engine and Guardrail Metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense


class TestStage2CompositeDecision(unittest.TestCase):
    def setUp(self):
        self.defense = SemanticQuasiIdentifierDefense(
            tau=0.72,
            floor_sim=0.60,
            weights={"sim": 0.50, "priv": 0.30, "read": 0.20},
            embedder_model_name="",
            request_timeout=1,
        )

    def test_privacy_reduction_metric(self):
        # 1. No modifications -> 1.0
        self.assertEqual(self.defense.compute_privacy_reduction("Candidate rewrite", []), 1.0)

        # 2. Perfect abstraction
        mods = [
            {"original_span": "$50,000", "generalized_span": "a five-figure sum"},
            {"original_span": "Senior Neurosurgeon", "generalized_span": "Senior Specialist"},
        ]
        cand_abstracted = "Transfer of a five-figure sum to Senior Specialist."
        self.assertEqual(self.defense.compute_privacy_reduction(cand_abstracted, mods), 1.0)

        # 3. Partial abstraction (1 left)
        cand_partial = "Transfer of $50,000 to Senior Specialist."
        self.assertEqual(self.defense.compute_privacy_reduction(cand_partial, mods), 0.5)

        # 4. Zero abstraction
        cand_none = "Transfer of $50,000 to Senior Neurosurgeon."
        self.assertEqual(self.defense.compute_privacy_reduction(cand_none, mods), 0.0)

    def test_readability_metric(self):
        orig = "Patient underwent surgery at Johns Hopkins on March 15, 2024."
        cand_fluent = "Patient underwent surgery at a regional hospital in early 2024."
        score = self.defense.compute_readability_score(orig, cand_fluent)
        self.assertGreater(score, 0.80)
        self.assertLessEqual(score, 1.0)

        # Truncation penalty
        cand_trunc = "Patient underwent"
        score_trunc = self.defense.compute_readability_score(orig, cand_trunc)
        self.assertLess(score_trunc, score)

        # Unbalanced quotes penalty
        cand_unbalanced = 'Patient underwent "surgery in early 2024.'
        score_unbal = self.defense.compute_readability_score(orig, cand_unbalanced)
        self.assertLess(score_unbal, score)

    def test_composite_acceptance(self):
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
        self.assertEqual(decision["metrics_breakdown"]["privacy_reduction_score"], 1.0)
        self.assertGreaterEqual(decision["metrics_breakdown"]["semantic_similarity"], 0.60)

    def test_hard_similarity_floor_enforcement(self):
        stage1 = "The patient was prescribed amoxicillin 500mg twice daily."
        # Hallucinated text totally unrelated
        cand = "Superconductors display zero electrical resistance below critical temperatures."
        mods = [{"original_span": "amoxicillin 500mg"}]

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand,
            modifications=mods,
            tau=0.50,
            floor_sim=0.60,
        )

        # Must reject because semantic similarity < 0.60 floor
        self.assertFalse(decision["is_accepted"])
        self.assertTrue(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], stage1)

    def test_empty_string_handling(self):
        res = self.defense.generalize_text("")
        self.assertEqual(res["final_text"], "")
        self.assertTrue(res["is_accepted"])
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual(res["composite_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
