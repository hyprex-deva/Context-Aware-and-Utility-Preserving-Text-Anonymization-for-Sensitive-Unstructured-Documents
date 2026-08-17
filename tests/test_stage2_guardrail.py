"""
Unit tests specifically for Stage 2 Independent Quasi-Identifier Detection & Abstraction Layer,
Genuine Partial Mitigation Scoring, and Three-Way Composite Decision Engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.quasi_identifier_detector import QuasiIdentifierDetector
from src.stage2_semantic_defense import SemanticQuasiIdentifierDefense


class TestQuasiIdentifierDetector(unittest.TestCase):
    """Unit tests for SLM-Independent Deterministic QI Detection Layer."""

    def setUp(self):
        self.detector = QuasiIdentifierDetector(
            mitigation_threshold=0.10,
            exposure_threshold=0.90,
        )

    def test_date_detection(self):
        text_iso = "The incident took place on 2024-03-15 at the clinic."
        qis_iso = self.detector.detect_dates(text_iso)
        self.assertEqual(len(qis_iso), 1)
        self.assertEqual(qis_iso[0]["type"], "DATE")
        self.assertEqual(qis_iso[0]["text"], "2024-03-15")

        text_us = "The appointment is scheduled for March 15, 2024."
        qis_us = self.detector.detect_dates(text_us)
        self.assertEqual(len(qis_us), 1)
        self.assertEqual(qis_us[0]["type"], "DATE")
        self.assertEqual(qis_us[0]["text"], "March 15, 2024")

        text_eu = "Review date: 15/03/2024."
        qis_eu = self.detector.detect_dates(text_eu)
        self.assertEqual(len(qis_eu), 1)
        self.assertEqual(qis_eu[0]["type"], "DATE")

    def test_money_detection(self):
        text = "Wire transfer of $45,382.19 and international fee of €12,381.50 was authorized."
        qis = self.detector.detect_monetary_values(text)
        self.assertEqual(len(qis), 2)
        self.assertEqual(qis[0]["type"], "MONEY")
        self.assertIn("45,382.19", qis[0]["text"])
        self.assertEqual(qis[1]["type"], "MONEY")
        self.assertIn("12,381.50", qis[1]["text"])

    def test_demographic_detection(self):
        text = "Subject is a 42-year-old male with recorded DOB: 12/04/1982."
        qis = self.detector.detect_demographics(text)
        self.assertEqual(len(qis), 2)
        types = [q["type"] for q in qis]
        self.assertTrue(all(t == "DEMOGRAPHIC" for t in types))

    def test_rare_role_detection_vs_generic(self):
        # Rare specialized role with uniqueness marker and specialty modifiers
        rare_text = "Dr. Smith is the sole Chief Pediatric Neurosurgeon specializing in DIPG at the facility."
        qis_rare = self.detector.detect_rare_roles(rare_text)
        self.assertGreaterEqual(len(qis_rare), 1)
        self.assertEqual(qis_rare[0]["type"], "ROLE")
        self.assertGreaterEqual(qis_rare[0]["specificity"], 0.60)

        # Generic executive role should receive low specificity / not classified as high-risk QI
        generic_text = "The Chief Executive Officer met with the general manager."
        qis_generic = self.detector.detect_rare_roles(generic_text)
        self.assertEqual(len(qis_generic), 0)

    def test_precise_location_detection(self):
        text = "Emergency dispatched to 742 Evergreen Terrace, Suite 400."
        qis = self.detector.detect_precise_locations(text)
        self.assertEqual(len(qis), 1)
        self.assertEqual(qis[0]["type"], "LOCATION")
        self.assertIn("742 Evergreen Terrace", qis[0]["text"])

    def test_compound_qi_detection(self):
        text = "The 42-year-old male is the sole Chief Pediatric Neurosurgeon at 123 Main Street."
        all_qis = self.detector.detect_quasi_identifiers(text)
        compound_qis = [q for q in all_qis if q["type"] == "COMPOUND_QI"]
        self.assertGreaterEqual(len(compound_qis), 1)
        self.assertIn("components", compound_qis[0])
        self.assertEqual(len(compound_qis[0]["components"]), 2)

    def test_no_qis_detected(self):
        text = "The weather was sunny and the meeting proceeded as scheduled."
        qis = self.detector.detect_quasi_identifiers(text)
        self.assertEqual(len(qis), 0)


class TestQIRiskAndMitigation(unittest.TestCase):
    """Unit tests for genuine partial mitigation, granularity reduction, and residual risk scoring."""

    def setUp(self):
        self.detector = QuasiIdentifierDetector(
            mitigation_threshold=0.10,
            exposure_threshold=0.90,
        )

    def test_date_full_exposure(self):
        qi = {"text": "March 15, 2024", "type": "DATE", "risk_level": 1.0, "parsed": {"day": 15, "month": "march", "year": 2024}}
        cand = "The surgery occurred on March 15, 2024."
        res = self.detector.evaluate_qi_mitigation(qi, cand)
        self.assertEqual(res["residual_risk"], 1.0)
        self.assertEqual(res["status"], "exposed")

    def test_date_full_mitigation(self):
        qi = {"text": "March 15, 2024", "type": "DATE", "risk_level": 1.0, "parsed": {"day": 15, "month": "march", "year": 2024}}
        cand = "The surgery occurred in early 2024."
        res = self.detector.evaluate_qi_mitigation(qi, cand)
        self.assertLessEqual(res["residual_risk"], 0.25)
        # Check qualitative year is significantly mitigated
        cand_coarse = "The surgery occurred in late summer."
        res_coarse = self.detector.evaluate_qi_mitigation(qi, cand_coarse)
        self.assertEqual(res_coarse["residual_risk"], 0.0)
        self.assertEqual(res_coarse["status"], "mitigated")

    def test_date_partial_mitigation(self):
        qi = {"text": "March 15, 2024", "type": "DATE", "risk_level": 1.0, "parsed": {"day": 15, "month": "march", "year": 2024}}
        cand = "The surgery occurred in March 2024."
        res = self.detector.evaluate_qi_mitigation(qi, cand)
        self.assertGreater(res["residual_risk"], 0.10)
        self.assertLess(res["residual_risk"], 0.90)
        self.assertEqual(res["status"], "partial")

    def test_role_partial_mitigation(self):
        """Verify that stripping modifiers from a specialized role yields 'partial', NOT 'mitigated'."""
        qi = {
            "text": "sole Chief Pediatric Neurosurgeon specializing in DIPG",
            "type": "ROLE",
            "risk_level": 1.0,
        }
        cand_partial = "The attending doctor was a pediatric neurosurgeon."
        res_partial = self.detector.evaluate_qi_mitigation(qi, cand_partial)
        # Must be classified as partial because 'pediatric neurosurgeon' still carries identifying specificity
        self.assertGreater(res_partial["residual_risk"], 0.10)
        self.assertLess(res_partial["residual_risk"], 0.90)
        self.assertEqual(res_partial["status"], "partial")

        # Complete generalization to broad category
        cand_full = "The attending doctor was a Medical Specialist."
        res_full = self.detector.evaluate_qi_mitigation(qi, cand_full)
        self.assertLessEqual(res_full["residual_risk"], 0.10)
        self.assertEqual(res_full["status"], "mitigated")

    def test_money_mitigation(self):
        qi = {"text": "$45,382.19", "type": "MONEY", "risk_level": 1.0, "parsed": {"raw_digits": "45382.19", "numeric_value": 45382.19}}
        
        # Exact exposure
        res_exp = self.detector.evaluate_qi_mitigation(qi, "Transferred $45,382.19 to vendor.")
        self.assertEqual(res_exp["residual_risk"], 1.0)
        self.assertEqual(res_exp["status"], "exposed")

        # Partial (approximate rounded value)
        res_part = self.detector.evaluate_qi_mitigation(qi, "Transferred approximately $45,000 to vendor.")
        self.assertGreater(res_part["residual_risk"], 0.10)
        self.assertLess(res_part["residual_risk"], 0.90)
        self.assertEqual(res_part["status"], "partial")

        # Full mitigation (coarsened to qualitative category)
        res_mit = self.detector.evaluate_qi_mitigation(qi, "Transferred a five-figure wire sum to vendor.")
        self.assertEqual(res_mit["residual_risk"], 0.0)
        self.assertEqual(res_mit["status"], "mitigated")

    def test_compound_qi_aggregation(self):
        """Verify conservative convex compound aggregation prevents a single mitigated attribute from zeroing risk."""
        qi_demo = {"text": "42-year-old male", "type": "DEMOGRAPHIC", "risk_level": 0.90, "parsed": {"subtype": "AGE_GENDER", "age": 42, "gender": "male"}}
        qi_role = {"text": "sole Chief Pediatric Neurosurgeon", "type": "ROLE", "risk_level": 1.0}
        compound_qi = {
            "text": "42-year-old male + sole Chief Pediatric Neurosurgeon",
            "type": "COMPOUND_QI",
            "risk_level": 0.95,
            "components": [qi_demo, qi_role],
        }

        # Case 1: Both attributes exposed (r1=1.0, r2=1.0 -> r_comp=1.0)
        cand_both = "The 42-year-old male is the sole Chief Pediatric Neurosurgeon."
        res_both = self.detector.evaluate_qi_mitigation(compound_qi, cand_both)
        self.assertEqual(res_both["residual_risk"], 1.0)
        self.assertEqual(res_both["status"], "exposed")

        # Case 2: One attribute mitigated (r1=1.0, r2=0.0 -> r_mean=0.5, r_prod=0.0 -> r_comp = 0.4*0.5 + 0.6*0 = 0.20)
        # Conservative aggregation guarantees risk remains non-zero (partial) rather than dropping to 0
        cand_partial_broken = "The 42-year-old male is a Medical Specialist."
        res_partial_broken = self.detector.evaluate_qi_mitigation(compound_qi, cand_partial_broken)
        self.assertGreater(res_partial_broken["residual_risk"], 0.10)
        self.assertLess(res_partial_broken["residual_risk"], 0.90)
        self.assertEqual(res_partial_broken["status"], "partial")
        self.assertAlmostEqual(res_partial_broken["residual_risk"], 0.20, places=2)

        # Case 3: Both attributes fully mitigated (r1=0.0, r2=0.0 -> r_comp = 0.0)
        cand_all_mitigated = "The patient was treated by a Medical Specialist."
        res_all_mitigated = self.detector.evaluate_qi_mitigation(compound_qi, cand_all_mitigated)
        self.assertEqual(res_all_mitigated["residual_risk"], 0.0)
        self.assertEqual(res_all_mitigated["status"], "mitigated")


class TestStage2CompositeDecision(unittest.TestCase):
    """Unit tests for Stage 2 Three-Way Safety Guardrail and Decision Engine."""

    def setUp(self):
        self.defense = SemanticQuasiIdentifierDefense(
            tau=0.72,
            floor_sim=0.60,
            qi_floor=0.50,
            weights={"sim": 0.50, "qi": 0.30, "read": 0.20},
            embedder_model_name="",
            request_timeout=1,
        )

    def test_three_way_guardrail_acceptance(self):
        stage1 = (
            "The patient was admitted to the hospital for observation and care. "
            "The procedure cost is $45,000 and the attending doctor is the sole Chief Pediatric Neurosurgeon "
            "scheduled on August 14, 2023 for comprehensive medical evaluation."
        )
        cand_good = (
            "The patient was admitted to the hospital for observation and care. "
            "The procedure cost is a five-figure wire transfer and the attending doctor is Medical Specialist "
            "scheduled in late summer 2023 for comprehensive medical evaluation."
        )

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand_good,
            tau=0.70,
            floor_sim=0.60,
            qi_floor=0.50,
        )

        self.assertTrue(decision["is_accepted"])
        self.assertFalse(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], cand_good)
        self.assertGreaterEqual(decision["composite_score"], 0.70)
        self.assertGreaterEqual(decision["metrics_breakdown"]["qi_abstraction_score"], 0.50)
        self.assertGreaterEqual(decision["metrics_breakdown"]["semantic_similarity"], 0.60)

    def test_rejection_due_to_qi_floor_failure(self):
        """Candidate preserves all high-risk QIs intact -> qi_abstraction fails floor -> REJECT."""
        stage1 = (
            "The patient is admitted to the hospital. "
            "The cost is $45,382.19 and the attending doctor is the sole Chief Pediatric Neurosurgeon "
            "scheduled on August 14, 2023."
        )
        # Candidate keeps exact $45,382.19, exact role, exact date
        cand_unmasked = (
            "The patient is admitted to the hospital. "
            "The cost is $45,382.19 and the attending doctor is the sole Chief Pediatric Neurosurgeon "
            "scheduled on August 14, 2023."
        )

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand_unmasked,
            tau=0.50,  # low tau
            floor_sim=0.50,  # low floor_sim
            qi_floor=0.70,  # strict QI floor
        )

        # Must reject because QI abstraction score < 0.70 floor
        self.assertFalse(decision["is_accepted"])
        self.assertTrue(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], stage1)

    def test_rejection_due_to_semantic_floor_failure(self):
        """Candidate hallucinates unrelated text -> semantic similarity fails floor -> REJECT."""
        stage1 = "The patient was prescribed amoxicillin 500mg twice daily."
        cand_hallucinated = "Superconductors display zero electrical resistance below critical temperatures."

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand_hallucinated,
            tau=0.40,
            floor_sim=0.60,
            qi_floor=0.20,
        )

        # Must reject because semantic similarity < 0.60 floor
        self.assertFalse(decision["is_accepted"])
        self.assertTrue(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], stage1)

    def test_rejection_due_to_composite_tau_failure(self):
        """High individual floors but composite < tau -> REJECT."""
        stage1 = "The patient arrived at the regional clinic for routine consultation."
        cand = "The patient arrived at the clinic for consultation."

        decision = self.defense.evaluate_composite_decision(
            stage1_text=stage1,
            candidate_text=cand,
            tau=0.99,  # excessively high tau
            floor_sim=0.50,
            qi_floor=0.50,
        )

        self.assertFalse(decision["is_accepted"])
        self.assertTrue(decision["fallback_triggered"])
        self.assertEqual(decision["final_text"], stage1)

    def test_empty_string_handling(self):
        res = self.defense.generalize_text("")
        self.assertEqual(res["final_text"], "")
        self.assertTrue(res["is_accepted"])
        self.assertFalse(res["fallback_triggered"])
        self.assertEqual(res["composite_score"], 1.0)
        self.assertEqual(res["metrics_breakdown"]["qi_abstraction_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
