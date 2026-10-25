import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_evaluation import distribution_diagnostics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.real = np.array([[0., 0.], [1., 2.], [2., 1.], [3., 3.]])
        self.training = self.real + 10

    def test_identical_samples_have_zero_distribution_loss(self):
        result = distribution_diagnostics(self.real, self.real.copy(), self.training, [1., 1.])
        self.assertAlmostEqual(result["mean_w1"], 0.)
        self.assertAlmostEqual(result["energy_v"], 0.)
        self.assertAlmostEqual(result["spearman_pair_rmse"], 0.)
        self.assertEqual(result["coverage"], 1.)
        self.assertEqual(result["near_duplicate_fraction"], 0.)

    def test_shifted_distribution_detected_even_when_correlation_unchanged(self):
        result = distribution_diagnostics(self.real, self.real + 100, self.training, [1., 1.])
        self.assertAlmostEqual(result["mean_w1"], 100.)
        self.assertGreater(result["energy_v"], 0)
        self.assertEqual(result["coverage"], 0.)

    def test_training_copies_are_not_mistaken_for_novel_samples(self):
        result = distribution_diagnostics(self.real, self.training.copy(), self.training, [1., 1.])
        self.assertEqual(result["near_duplicate_fraction"], 1.)

    def test_constants_do_not_produce_fake_zero_correlation_error(self):
        constant = np.ones((4, 2))
        result = distribution_diagnostics(constant, constant, constant, [1., 1.])
        self.assertEqual(result["n_spearman_pairs"], 0)
        self.assertTrue(np.isnan(result["spearman_pair_rmse"]))

    def test_provided_training_scale_is_respected_not_refitted(self):
        result = distribution_diagnostics(self.real, self.real + 2, self.training, [2., 2.])
        self.assertAlmostEqual(result["mean_w1"], 1.)

    def test_invalid_sample_counts_scale_and_neighbor_count_rejected(self):
        for generated, scale, k in [(self.real[:2], [1., 1.], 1),
                                     (self.real, [0., 1.], 1),
                                     (self.real, [1., 1.], 4),
                                     (self.real, [1., 1.], 0)]:
            with self.assertRaises(ValueError):
                distribution_diagnostics(self.real, generated, self.training, scale, knn_k=k)
        with self.assertRaises(ValueError):
            distribution_diagnostics(self.real, self.real * np.nan, self.training, [1., 1.])


if __name__ == "__main__":
    unittest.main()
