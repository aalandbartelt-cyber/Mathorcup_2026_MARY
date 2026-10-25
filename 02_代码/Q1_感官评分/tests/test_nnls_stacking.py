from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from nnls_stacking import fit_nnls_stacking


class NNLSStackingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.stack = rng.normal(size=(80, 3, 2))
        self.weights = np.array([[1.2, .2], [1.2, .2], [.1, 1.3]])
        self.y = np.einsum('ntm,tm->nt', self.stack, self.weights) + [-3., 1., -2.]

    def test_shared_nonnegative_weights_allow_a_free_negative_intercept(self):
        y = np.einsum('ntm,m->nt', self.stack, [1.3, .4]) + [-3., 1., -2.]
        model = fit_nnls_stacking(self.stack, y, scope="shared")
        np.testing.assert_allclose(model.weights, np.tile([1.3, .4], (3, 1)), atol=1e-10)
        np.testing.assert_allclose(model.intercept, [-3., 1., -2.], atol=1e-10)
        np.testing.assert_allclose(model.predict(self.stack), y, atol=1e-10)
        self.assertTrue((model.weights >= 0).all())

    def test_group_weights_share_only_within_registered_group(self):
        model = fit_nnls_stacking(self.stack, self.y, groups=["pair", "pair", "other"],
                                  scope="group", shrinkage=0.)
        np.testing.assert_allclose(model.weights, self.weights, atol=1e-10)
        self.assertFalse(np.allclose(model.weights[0], model.weights[2]))

    def test_fixed_half_shrinkage_is_between_local_and_global(self):
        global_model = fit_nnls_stacking(self.stack, self.y, scope="shared")
        local_model = fit_nnls_stacking(self.stack, self.y, scope="target", shrinkage=0.)
        shrunk_model = fit_nnls_stacking(self.stack, self.y, scope="target", shrinkage=.5)
        np.testing.assert_allclose(shrunk_model.weights,
                                   .5 * local_model.weights + .5 * global_model.weights)
        np.testing.assert_allclose(shrunk_model.predict(self.stack).mean(axis=0),
                                   self.y.mean(axis=0), atol=1e-12)

    def test_full_shrinkage_recovers_shared_prediction(self):
        shared = fit_nnls_stacking(self.stack, self.y, scope="shared")
        group = fit_nnls_stacking(self.stack, self.y, groups=["pair", "pair", "other"],
                                  scope="group", shrinkage=1.)
        np.testing.assert_allclose(shared.predict(self.stack), group.predict(self.stack))

    def test_constant_base_models_fall_back_to_training_target_mean(self):
        stack = np.ones_like(self.stack)
        model = fit_nnls_stacking(stack, self.y, scope="target", shrinkage=.5)
        np.testing.assert_allclose(model.weights, 0.)
        np.testing.assert_allclose(model.predict(stack[:5]),
                                   np.tile(self.y.mean(axis=0), (5, 1)))

    def test_prediction_cannot_change_training_weights(self):
        model = fit_nnls_stacking(self.stack, self.y, scope="shared")
        before = model.weights.copy(), model.intercept.copy()
        model.predict(self.stack[:5] * 1000 + 42)
        np.testing.assert_array_equal(model.weights, before[0])
        np.testing.assert_array_equal(model.intercept, before[1])

    def test_invalid_scope_shapes_or_groups_are_rejected(self):
        for kwargs in [{"scope": "unknown"}, {"scope": "group", "groups": ["a"]},
                       {"scope": "target", "shrinkage": 1.1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_nnls_stacking(self.stack, self.y, **kwargs)
        with self.assertRaises(ValueError):
            fit_nnls_stacking(self.stack[:-1], self.y)
        with self.assertRaises(ValueError):
            fit_nnls_stacking(self.stack * np.nan, self.y)


if __name__ == "__main__":
    unittest.main()
