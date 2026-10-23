from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    q2 = importlib.import_module("q2_contract")
except ModuleNotFoundError:
    q2 = None


class Q2ContractTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(q2, "q2_contract module has not been implemented")
        return q2

    def test_load_q2_reuses_official_aligned_shapes(self) -> None:
        module = self.require_module()
        x, y, style, beer = module.load_q2()
        self.assertEqual(x.shape, (250, 231))
        self.assertEqual(y.shape, (250, 50))
        self.assertEqual(style.nunique(), 22)
        self.assertEqual(beer.nunique(), 250)
        self.assertEqual(int(x.isna().sum().sum()), 73)
        self.assertFalse(y.isna().any().any())

    def test_candidate_dictionary_freezes_strict_compound_scope(self) -> None:
        module = self.require_module()
        x, _, _, _ = module.load_q2()
        candidates = module.build_candidate_dictionary(x.columns)
        self.assertEqual(len(candidates), 231)
        self.assertTrue(candidates["feature_code"].is_unique)
        self.assertEqual(candidates["feature_code"].tolist(), x.columns.tolist())
        self.assertTrue(candidates["eligible_all"].all())
        self.assertEqual(int(candidates["eligible_strict"].sum()), 218)
        excluded = candidates.loc[~candidates["eligible_strict"]]
        self.assertTrue(excluded["exclusion_reason"].str.len().gt(0).all())
        expected_excluded = {
            "color.EBC.",
            "betaglucan.mg.L.",
            "iron",
            "pH",
            "protein.g.L.",
            "bitterness",
            "GFS",
            "unfermentables",
            "kcalperc",
            "esters_sum",
            "aroma_hops_sum",
            "acids_sum",
            "sulfur_sum",
        }
        self.assertEqual(set(excluded["feature_code"]), expected_excluded)
        self.assertTrue(
            candidates.set_index("feature_code").loc[
                ["DMS", "X2.3.butanedione", "ethanol..v.v.", "CO2.PSI."],
                "eligible_strict",
            ].all()
        )

    def test_official_attachment_is_spearman_not_pearson(self) -> None:
        module = self.require_module()
        x, _, _, _ = module.load_q2()
        audit = module.audit_official_graph(x)
        self.assertLess(audit["spearman_max_abs_diff"], 1e-12)
        self.assertGreater(audit["pearson_max_abs_diff"], 1.0)
        self.assertEqual(audit["matrix_size"], 231)
        self.assertTrue(audit["symmetric"])
        self.assertTrue(audit["diagonal_is_one"])

    def test_soft_signed_edges_preserve_weight_and_direction(self) -> None:
        module = self.require_module()
        corr = pd.DataFrame(
            [[1.0, 0.5, -0.25], [0.5, 1.0, 0.0], [-0.25, 0.0, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        edges = module.build_soft_edge_table(corr, betas=(1, 2))
        self.assertEqual(len(edges), 3)
        edge_ab = edges.query("source == 'a' and target == 'b'").iloc[0]
        edge_ac = edges.query("source == 'a' and target == 'c'").iloc[0]
        self.assertEqual(edge_ab["sign"], 1)
        self.assertEqual(edge_ac["sign"], -1)
        self.assertAlmostEqual(edge_ab["weight_beta_2"], 0.25)
        self.assertAlmostEqual(edge_ac["signed_weight_beta_2"], -0.0625)

    def test_signed_normalized_laplacian_is_symmetric_psd(self) -> None:
        module = self.require_module()
        corr = pd.DataFrame(
            [[1.0, 0.7, -0.4], [0.7, 1.0, 0.2], [-0.4, 0.2, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        laplacian = module.signed_normalized_laplacian(corr, beta=2)
        np.testing.assert_allclose(laplacian, laplacian.T, atol=1e-12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(laplacian).min()), -1e-10)

    def test_fold_graph_ignores_rows_outside_training_indices(self) -> None:
        module = self.require_module()
        x = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0, 100.0],
                "b": [0.0, 2.0, 1.0, -100.0],
                "c": [3.0, 1.0, 2.0, 999.0],
            }
        )
        train = np.array([0, 1, 2])
        first = module.fold_spearman_graph(x, train)
        changed = x.copy()
        changed.loc[3] = [-9999.0, 8888.0, -7777.0]
        second = module.fold_spearman_graph(changed, train)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
