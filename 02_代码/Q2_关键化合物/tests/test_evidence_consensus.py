from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evidence_consensus import (  # noqa: E402
    audit_payload_passed,
    fuse_evidence_families,
    reference_half_jaccard,
    replacement_decision,
)


class EvidenceConsensusTests(unittest.TestCase):
    def test_family_fusion_prioritizes_cross_family_support(self) -> None:
        frame = pd.DataFrame(
            {
                "target_code": ["t", "t"],
                "feature_code": ["a", "b"],
                "predictive_rank": [1, 4],
                "partial_rank": [20, 2],
                "module_rank": [30, 3],
            }
        )
        result = fuse_evidence_families(
            frame,
            rank_columns=["predictive_rank", "partial_rank", "module_rank"],
            rrf_k=60,
        )
        self.assertEqual(result.iloc[0]["feature_code"], "b")
        self.assertEqual(result.iloc[0]["consensus_rank"], 1)

    def test_replacement_requires_two_stability_gains_and_all_hard_gates(self) -> None:
        decision = replacement_decision(
            complete_rows=150,
            prediction_ok=True,
            stability_gains={"frequency": True, "jaccard": True, "style": False},
            anchors_preserved=True,
            fdp_ok=True,
            leakage_ok=True,
        )
        self.assertTrue(decision["promoted"])
        decision = replacement_decision(
            complete_rows=150,
            prediction_ok=True,
            stability_gains={"frequency": True, "jaccard": False, "style": False},
            anchors_preserved=True,
            fdp_ok=True,
            leakage_ok=True,
        )
        self.assertFalse(decision["promoted"])

    def test_reference_half_jaccard_uses_each_target_half(self) -> None:
        top3 = pd.DataFrame(
            {
                "target_code": ["t", "t", "t"],
                "feature_code": ["a", "b", "c"],
            }
        )
        halves = pd.DataFrame(
            {
                "pair": [0, 0, 0, 0, 0, 0],
                "half": ["left"] * 3 + ["right"] * 3,
                "target_code": ["t"] * 6,
                "feature_code": ["a", "b", "c", "a", "d", "e"],
            }
        )
        values = reference_half_jaccard(top3, halves)
        self.assertEqual(values.tolist(), [1.0, 0.2])

    def test_audit_payload_accepts_legacy_complete_scope_verified_schema(self) -> None:
        self.assertTrue(
            audit_payload_passed(
                {"status": "complete", "scope_verified": True, "errors": []}
            )
        )
        self.assertTrue(audit_payload_passed({"audit_status": "pass", "failed_checks": []}))
        self.assertFalse(
            audit_payload_passed(
                {"status": "complete", "scope_verified": False, "errors": ["leak"]}
            )
        )


if __name__ == "__main__":
    unittest.main()
