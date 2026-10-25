from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from partial_graph import (  # noqa: E402
    FoldGaussianizer,
    coclustering_matrix,
    edge_table,
    heldout_gaussian_log_likelihood,
    louvain_membership,
    precision_to_partial_correlation,
    stable_edge_table,
)


class PartialGraphTests(unittest.TestCase):
    def test_gaussianizer_does_not_refit_on_validation(self) -> None:
        train = np.arange(30, dtype=float).reshape(10, 3)
        validation = np.full((2, 3), 999.0)
        transformer = FoldGaussianizer().fit(train)
        before = transformer.quantiles_.copy()

        transformed = transformer.transform(validation)

        np.testing.assert_array_equal(before, transformer.quantiles_)
        self.assertTrue(np.isfinite(transformed).all())

    def test_partial_correlation_is_symmetric_with_unit_diagonal(self) -> None:
        precision = np.array([[2.0, -0.5], [-0.5, 3.0]])

        partial = precision_to_partial_correlation(precision)

        np.testing.assert_allclose(partial, partial.T)
        np.testing.assert_allclose(np.diag(partial), 1.0)
        self.assertGreater(partial[0, 1], 0.0)

    def test_coclustering_is_label_invariant(self) -> None:
        first = np.array([0, 0, 1, 1])
        relabelled = np.array([7, 7, 3, 3])

        np.testing.assert_allclose(
            coclustering_matrix([first]), coclustering_matrix([relabelled])
        )

    def test_edge_stability_counts_absent_folds_and_sign_consistency(self) -> None:
        first = pd.DataFrame(
            {
                "source": ["a", "b"],
                "target": ["b", "c"],
                "partial_correlation": [0.4, -0.2],
            }
        )
        second = pd.DataFrame(
            {
                "source": ["a"],
                "target": ["b"],
                "partial_correlation": [0.5],
            }
        )

        result = stable_edge_table([first, second])

        ab = result.loc[
            result["source"].eq("a") & result["target"].eq("b")
        ].iloc[0]
        bc = result.loc[
            result["source"].eq("b") & result["target"].eq("c")
        ].iloc[0]
        self.assertAlmostEqual(ab["selection_frequency"], 1.0)
        self.assertAlmostEqual(ab["sign_consistency"], 1.0)
        self.assertAlmostEqual(bc["selection_frequency"], 0.5)

    def test_louvain_gives_isolates_distinct_singleton_modules(self) -> None:
        partial = np.eye(4)
        partial[0, 1] = partial[1, 0] = 0.6

        membership = louvain_membership(
            partial, ["a", "b", "c", "d"], edge_threshold=0.1, seed=17
        )

        self.assertEqual(membership[0], membership[1])
        self.assertNotEqual(membership[2], membership[3])
        self.assertNotEqual(membership[0], membership[2])

    def test_edge_table_uses_deterministic_upper_triangle_order(self) -> None:
        partial = np.array(
            [[1.0, -0.3, 0.1], [-0.3, 1.0, 0.4], [0.1, 0.4, 1.0]]
        )

        result = edge_table(partial, ["z", "a", "m"], edge_threshold=0.2)

        self.assertEqual(result[["source", "target"]].values.tolist(), [["z", "a"], ["a", "m"]])
        self.assertEqual(result["sign"].tolist(), [-1, 1])

    def test_heldout_log_likelihood_rewards_matching_center(self) -> None:
        validation = np.zeros((5, 2), dtype=float)
        precision = np.eye(2)

        centered = heldout_gaussian_log_likelihood(
            validation, mean=np.zeros(2), precision=precision
        )
        shifted = heldout_gaussian_log_likelihood(
            validation, mean=np.full(2, 3.0), precision=precision
        )

        self.assertGreater(centered, shifted)


if __name__ == "__main__":
    unittest.main()
