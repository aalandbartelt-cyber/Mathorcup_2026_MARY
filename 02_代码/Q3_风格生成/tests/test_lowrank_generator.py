from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from lowrank_generator import SharedLowRankStyleGenerator


class SharedLowRankStyleGeneratorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.styles = np.array(["a"] * 30 + ["b"] * 30)
        latent = rng.normal(size=(60, 3))
        loadings = rng.normal(size=(3, 8))
        self.x = latent @ loadings + rng.normal(scale=0.15, size=(60, 8))
        self.x[self.styles == "b"] += 0.6
        self.x[:, 2] = rng.uniform(-1.5, 2.0, size=60)
        self.x[:22, 2] = -2.0
        self.x[30:38, 2] = -2.0

    def test_registered_methods_are_reproducible_and_supported(self):
        model = SharedLowRankStyleGenerator().fit(
            self.x,
            self.styles,
            rank=3,
            covariance_kappa=12.0,
            mean_kappa=8.0,
            floor_specs={2: {"value": -2.0, "tolerance": 1e-12, "prior_strength": 5.0}},
        )
        for method in ["shared_lowrank", "style_residual_shrunk", "style_residual_floor"]:
            first = model.generate("a", 120, method=method, seed=9)
            second = model.generate("a", 120, method=method, seed=9)
            np.testing.assert_array_equal(first["values"], second["values"])
            self.assertEqual(first["values"].shape, (120, 8))
            self.assertTrue(np.isfinite(first["values"]).all())
            self.assertTrue(np.all(first["values"] >= self.x[:30].min(0) - 1e-12))
            self.assertTrue(np.all(first["values"] <= self.x[:30].max(0) + 1e-12))

    def test_floor_state_is_explicit_and_coupled_to_generated_latent(self):
        model = SharedLowRankStyleGenerator().fit(
            self.x,
            self.styles,
            rank=3,
            covariance_kappa=12.0,
            mean_kappa=8.0,
            floor_specs={2: {"value": -2.0, "tolerance": 1e-12, "prior_strength": 5.0}},
        )
        sample = model.generate("a", 4000, method="style_residual_floor", seed=3)
        observed = np.mean(np.isclose(sample["values"][:, 2], -2.0, atol=1e-12))
        expected = np.mean(np.isclose(self.x[:30, 2], -2.0, atol=1e-12))
        self.assertLess(abs(observed - expected), 0.08)
        self.assertAlmostEqual(sample["floor_fraction"][2], observed)
        self.assertTrue(np.all(sample["values"][~np.isclose(sample["values"][:, 2], -2.0), 2] > -2.0))

    def test_diagnostics_expose_valid_shrinkage_and_psd_covariances(self):
        model = SharedLowRankStyleGenerator().fit(
            self.x, self.styles, rank=4, covariance_kappa=20.0, mean_kappa=10.0
        )
        diagnostic = model.diagnostics("a")
        self.assertEqual(diagnostic["rank"], 4)
        self.assertAlmostEqual(diagnostic["mean_weight"], 30 / 40)
        self.assertAlmostEqual(diagnostic["covariance_weight"], 30 / 50)
        self.assertGreaterEqual(diagnostic["latent_min_eigenvalue"], -1e-10)
        self.assertGreater(diagnostic["residual_variance_min"], 0.0)

    def test_invalid_contract_and_unseen_style_are_rejected(self):
        with self.assertRaises(ValueError):
            SharedLowRankStyleGenerator().fit(self.x, self.styles, rank=0)
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            SharedLowRankStyleGenerator().fit(bad, self.styles, rank=3)
        model = SharedLowRankStyleGenerator().fit(self.x, self.styles, rank=3)
        with self.assertRaises(ValueError):
            model.generate("c", 3)
        with self.assertRaises(ValueError):
            model.generate("a", 3, method="unknown")

    def test_singleton_non_target_style_shrinks_to_shared_residual(self):
        labels = self.styles.astype(object).copy()
        labels[0] = "rare"
        model = SharedLowRankStyleGenerator().fit(self.x, labels, rank=3, covariance_kappa=10.0)
        sample = model.generate("rare", 4, method="style_residual_shrunk", seed=4)
        self.assertTrue(np.isfinite(sample["values"]).all())
        self.assertEqual(model.diagnostics("rare")["n_training_style"], 1)


if __name__ == "__main__":
    unittest.main()
