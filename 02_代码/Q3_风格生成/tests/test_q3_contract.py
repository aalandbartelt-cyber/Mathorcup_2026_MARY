from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_contract import align_by_beer, defect_registry, feature_contract, source_defect_table


class ContractTests(unittest.TestCase):
    def fixture(self):
        n = 21
        values = np.arange(n, dtype=float) + 2
        chemistry = pd.DataFrame({"beer": np.arange(n), "beer_id": np.arange(n) + 100,
                                  "tasting_category_fine": ["Pils/Lager"] * n,
                                  "DMS": np.log10(values), "X2.3.butanedione": values,
                                  "X4VG": np.log10(values / 100), "X.E..2.nonenal": values / 1000,
                                  "unverified": values})
        calibrated = chemistry.drop(columns="unverified").copy()
        calibrated["DMS"] = values
        calibrated["X4VG"] = values / 100
        metadata = pd.DataFrame({"field_name_R": list(chemistry.columns[3:]),
                                 "unit": ["ug/L", "mg/L", "mg/L", "mg/L", "A.U."],
                                 "threshold_mean": [17.25, .29925, .206333, np.nan, np.nan],
                                 "threshold_median": [19., .09, .2, np.nan, np.nan]})
        return chemistry, calibrated, metadata

    def test_alignment_uses_beer_and_preserves_values_not_beer_id(self):
        x, c, _ = self.fixture()
        c["beer_id"] = -1
        aligned = align_by_beer(x, c.sample(frac=1, random_state=1), "calibrated")
        np.testing.assert_array_equal(aligned.beer, x.beer)
        np.testing.assert_array_equal(aligned.DMS, c.DMS)
        self.assertTrue(aligned.beer_id.eq(-1).all())

    def test_duplicate_missing_key_or_disagreeing_style_rejected(self):
        x, c, _ = self.fixture()
        for bad in [pd.concat([c, c.iloc[:1]]), c.iloc[1:], c.assign(tasting_category_fine="other")]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                align_by_beer(x, bad, "calibrated")
        c.loc[0, "beer"] = np.nan
        with self.assertRaises(ValueError):
            align_by_beer(x, c, "calibrated")

    def test_identifies_log_raw_and_unverified_without_mutation(self):
        x, c, m = self.fixture()
        original = x.copy(deep=True)
        fields, cells = feature_contract(x, c, m)
        scale = fields.set_index("feature_code").scale_in_1_csv.to_dict()
        self.assertEqual(scale["DMS"], "log10")
        self.assertEqual(scale["X2.3.butanedione"], "raw")
        self.assertEqual(scale["unverified"], "unverified")
        self.assertEqual(len(cells), 21 * 4)
        self.assertTrue(cells.status.eq("observed").all())
        pd.testing.assert_frame_equal(original, x)

    def test_field_recognition_does_not_certify_conflicting_cell(self):
        x, c, m = self.fixture()
        c.loc[0, "X2.3.butanedione"] = 1000
        fields, cells = feature_contract(x, c, m)
        row = fields.set_index("feature_code").loc["X2.3.butanedione"]
        self.assertEqual(row.scale_in_1_csv, "raw")
        self.assertEqual(row.conflict_count, 1)
        self.assertEqual(row.scale_status, "recognized_with_conflicts")
        bad = cells.query('beer==0 and feature_code=="X2.3.butanedione"').iloc[0]
        self.assertEqual(bad.status, "reference_conflict")
        source = source_defect_table(x, c, fields)
        self.assertFalse(source.query('beer==0 and feature_code=="X2.3.butanedione"').iloc[0].source_known)

    def test_reference_zeros_and_missing_are_not_replaced_by_concentrations(self):
        x, c, m = self.fixture()
        x.loc[0, "X2.3.butanedione"] = np.nan
        c.loc[0, "X2.3.butanedione"] = 0
        c.loc[1, "X4VG"] = 0
        fields, cells = feature_contract(x, c, m)
        self.assertEqual(cells.query('beer==0 and feature_code=="X2.3.butanedione"').iloc[0].status,
                         "missing_stored")
        self.assertEqual(cells.query('beer==1 and feature_code=="X4VG"').iloc[0].status,
                         "reference_zero_ambiguous")
        source = source_defect_table(x, c, fields)
        self.assertFalse(source.query('beer==0 and feature_code=="X2.3.butanedione"').iloc[0].source_known)
        self.assertFalse(source.query('beer==1 and feature_code=="X4VG"').iloc[0].source_known)

    def test_ambiguous_scale_is_unverified_not_guessed(self):
        x, c, m = self.fixture()
        c["DMS"] = 10**(np.arange(21) + 4.)
        fields, _ = feature_contract(x, c, m)
        self.assertEqual(fields.set_index("feature_code").loc["DMS", "scale_in_1_csv"], "unverified")

    def test_registry_preserves_missing_threshold_and_sensory_only_defect(self):
        _, _, m = self.fixture()
        reg = defect_registry(m).set_index("defect")
        self.assertEqual(len(reg), 5)
        self.assertTrue(pd.isna(reg.loc["trans-2-nonenal", "threshold_median"]))
        self.assertEqual(reg.loc["Stale hops", "evidence_type"], "sensory_proxy_only")
        self.assertEqual(reg.loc["DMS", "unit"], "ug/L")

    def test_unit_mismatch_and_duplicate_metadata_rejected(self):
        x, c, m = self.fixture()
        wrong = m.copy()
        wrong.loc[wrong.field_name_R.eq("DMS"), "unit"] = "mg/L"
        with self.assertRaises(ValueError):
            defect_registry(wrong)
        with self.assertRaises(ValueError):
            feature_contract(x, c, pd.concat([m, m.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
