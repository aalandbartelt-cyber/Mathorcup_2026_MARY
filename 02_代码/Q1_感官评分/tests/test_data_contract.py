from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_contract import (  # noqa: E402
    ATTACHMENTS,
    compound_name_map,
    feature_scale_audit,
    load_q1,
    missingness_audit,
    preprocessing_steps,
    validate_compound_name_map,
)


class DataContractTests(unittest.TestCase):
    def test_official_compound_name_map_is_complete_and_one_to_one(self) -> None:
        x, _, _ = load_q1()
        mapping = compound_name_map(x)
        audit = validate_compound_name_map(mapping, x)
        self.assertEqual(audit["rows"], 231)
        self.assertEqual(audit["unique_compound_names"], 231)
        self.assertEqual(audit["unique_feature_codes"], 231)
        self.assertEqual(audit["attachment8_field_codes_matched"], 231)
        self.assertEqual(audit["status"], "PASS")

    def test_feature_scale_audit_identifies_official_log10_fields(self) -> None:
        chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
        audit = feature_scale_audit(chemistry)
        self.assertEqual(len(audit), 70)
        self.assertEqual((audit["scale_in_1_csv"] == "log10").sum(), 37)
        self.assertEqual((audit["scale_in_1_csv"] == "raw").sum(), 33)
        self.assertTrue(
            (audit.loc[audit["scale_in_1_csv"] == "log10", "positive_value_match_rate"] == 1.0).all()
        )

    def test_missingness_and_preprocessing_preserve_all_samples(self) -> None:
        x, y, _ = load_q1()
        missing = missingness_audit(x)
        self.assertEqual(int(missing["missing_cells"].sum()), 73)
        self.assertEqual(int((missing["missing_cells"] > 0).sum()), 9)
        self.assertEqual(int(x.isna().any(axis=1).sum()), 63)
        steps = preprocessing_steps(x, y)
        self.assertEqual(steps.iloc[0]["rows_after"], "250")
        self.assertTrue((steps["sample_deletion"] == "no").all())
        self.assertIn("n_train×(231+k)", steps["shape_after"].tolist())


if __name__ == "__main__":
    unittest.main()
