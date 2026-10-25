import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_e200_contract import run_contract, validate_cv


class E200Tests(unittest.TestCase):
    def tables(self):
        value = np.arange(6.) + 2
        x = pd.DataFrame({"beer": np.arange(6), "beer_id": -1,
                          "tasting_category_fine": "Pils/Lager", "DMS": np.log10(value),
                          "X2.3.butanedione": value, "X4VG": np.log10(value / 100),
                          "X.E..2.nonenal": value / 1000, "unknown": value})
        c = x.drop(columns="unknown").copy()
        c["DMS"], c["X4VG"] = value, value / 100
        y = x.iloc[:, :3].copy()
        y["dms"], y["stale_hops"] = value, -value
        meta = pd.DataFrame({"field_name_R": list(x.columns[3:]),
                             "unit": ["ug/L", "mg/L", "mg/L", "mg/L", "A.U."],
                             "threshold_mean": [17.25, .29925, .2063, np.nan, np.nan],
                             "threshold_median": [19., .09, .2, np.nan, np.nan]})
        split = pd.DataFrame([{"repeat": r, "fold": i % 3, "row_position": i,
                               "beer": i, "tasting_category_fine": "Pils/Lager"}
                              for r in range(2) for i in range(6)])
        return x, y, c, meta, split

    def test_cv_rejects_duplicate_slots_wrong_beer_or_style(self):
        x, _, _, _, splits = self.tables()
        validate_cv(x, splits, n_repeats=2, n_folds=3)
        for column, value in [("beer", 99), ("tasting_category_fine", "other"), ("row_position", 5)]:
            bad = splits.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_cv(x, bad, n_repeats=2, n_folds=3)
        with self.assertRaises(ValueError):
            validate_cv(x, splits.iloc[:-1], n_repeats=2, n_folds=3)

    def test_writer_exports_contract_only_and_preserves_input_hashes(self):
        x, y, c, meta, splits = self.tables()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attachment = root / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件"
            attachment.mkdir(parents=True)
            x.to_csv(attachment / "1.csv", index=False)
            y.sample(frac=1, random_state=3).to_csv(attachment / "4.csv", index=False)
            with pd.ExcelWriter(attachment / "8.xlsx") as writer:
                c.sample(frac=1, random_state=1).to_excel(writer, sheet_name="Calibrated chemical dataset", index=False)
                meta.to_excel(writer, sheet_name="Sensory thresholds", index=False)
            (root / "01_数据").mkdir()
            pd.DataFrame({"feature_code": x.columns[3:]}).to_csv(root / "01_数据/compound_name_map.csv", index=False)
            cv = root / "03_结果/Q1_感官评分"
            cv.mkdir(parents=True)
            splits.to_csv(cv / "q1_cv_splits.csv", index=False)
            settings = {"expected_beers": 6, "expected_features": 5, "expected_targets": 2,
                        "expected_styles": 1, "expected_scale_counts": {"log10": 2, "raw": 2, "unverified": 1},
                        "expected_missing_cells": 0, "expected_source_beers": 6,
                        "n_repeats": 2, "n_folds": 3, "source_style": "Pils/Lager",
                        "minimum_scale_pairs": 3, "minimum_scale_match": .95}
            out = root / "q3_outputs"
            first = run_contract(root, out, settings=settings)
            self.assertFalse(first["model_fitted"])
            self.assertTrue(first["input_hashes_unchanged"])
            self.assertEqual(first["counts"]["n_source_defect_rows"], 24)
            self.assertEqual(len(list(out.iterdir())), 8)
            self.assertTrue(all(f.suffix in [".csv", ".json"] for f in out.iterdir()))
            manifest = json.loads((out / "q3_run_manifest_E200.json").read_text())
            second = run_contract(root, out, settings=settings)
            self.assertEqual(manifest["output_sha256"], second["output_sha256"])


if __name__ == "__main__":
    unittest.main()
