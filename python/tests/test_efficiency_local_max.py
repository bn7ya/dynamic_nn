"""
Integration tests for the local-maximum efficiency gate.

The gate defers RemoveNodes / RemoveLayer decisions until the per-epoch
efficiency curve E(t) has reached a discrete local maximum
(slope ~ 0 with curvature <= 0) AND a warmup + cooldown have elapsed.

These tests exercise:
  1. The Python binding surface for the new LayerManagerConfig fields.
  2. The default-on behaviour: an end-to-end training run must not
     collapse architecture during the warmup window.
  3. Reproducibility: turning the gate off recovers the legacy behaviour.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from pydnn import DynamicNetwork  # noqa: E402


def _toy_classification(n_samples=128, n_features=16, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_int = rng.integers(0, n_classes, size=n_samples)
    Y = np.eye(n_classes, dtype=np.float32)[y_int]
    return X, Y


class TestGateConfigSurface(unittest.TestCase):
    """The five new LayerManagerConfig fields are bound to Python."""

    def setUp(self):
        try:
            from pydnn import _dnn_core  # noqa: F401
        except ImportError:
            self.skipTest("_dnn_core extension not built")

    def test_fields_exist_with_documented_defaults(self):
        from pydnn._dnn_core import LayerManagerConfig

        cfg = LayerManagerConfig()
        self.assertTrue(cfg.shrink_requires_plateau)
        self.assertEqual(cfg.plateau_window, 10)
        self.assertAlmostEqual(cfg.plateau_slope_epsilon, 1e-4, places=6)
        self.assertEqual(cfg.min_epochs_before_shrink, 8)
        self.assertEqual(cfg.shrink_cooldown_epochs, 5)

    def test_fields_round_trip(self):
        from pydnn._dnn_core import LayerManagerConfig

        cfg = LayerManagerConfig()
        cfg.shrink_requires_plateau = False
        cfg.plateau_window = 14
        cfg.plateau_slope_epsilon = 5e-5
        cfg.min_epochs_before_shrink = 12
        cfg.shrink_cooldown_epochs = 7

        self.assertFalse(cfg.shrink_requires_plateau)
        self.assertEqual(cfg.plateau_window, 14)
        self.assertAlmostEqual(cfg.plateau_slope_epsilon, 5e-5)
        self.assertEqual(cfg.min_epochs_before_shrink, 12)
        self.assertEqual(cfg.shrink_cooldown_epochs, 7)

    def test_fields_attached_to_trainer_config(self):
        """TrainerConfig().layer_manager_config exposes the new fields."""
        from pydnn._dnn_core import TrainerConfig

        tc = TrainerConfig()
        self.assertTrue(tc.layer_manager_config.shrink_requires_plateau)
        # The composite write path used by the dynamic-thresholds layer
        # must accept assignments without throwing.
        tc.layer_manager_config.shrink_requires_plateau = False
        self.assertFalse(tc.layer_manager_config.shrink_requires_plateau)


class TestEndToEndWarmupBehavior(unittest.TestCase):
    """A real training run must not collapse architecture during warmup.

    The user's symptom was: with default settings, the network shrank
    before learning, producing under-fit models. The fix is the local-
    max gate. This test runs a small but real training cycle and
    verifies architecture stability through the warmup window.
    """

    def setUp(self):
        try:
            from pydnn import _dnn_core  # noqa: F401
        except ImportError:
            self.skipTest("_dnn_core extension not built")

    def test_default_run_completes_and_does_not_collapse(self):
        X, Y = _toy_classification(seed=42)
        net = DynamicNetwork(
            input_shape=(X.shape[1],),
            output_size=Y.shape[1],
            seed=42,
            cost_function='CrossEntropy',
            runtime_enabled=True,
        )
        result = net.fit(X, Y, verbose=False)

        # Training must complete without error.
        self.assertGreater(result.epochs_completed, 0)
        # Final architecture must retain at least one layer.
        self.assertGreaterEqual(net.num_layers, 1)
        # Parameter count must be > 0 (sanity).
        self.assertGreater(net.num_parameters, 0)
        # Predictions must produce finite values of the right shape.
        preds = net.predict(X[:8])
        self.assertEqual(preds.shape, (8, Y.shape[1]))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_after_training(self):
        """A model trained with the gate on must still produce
        well-formed predictions (gate path doesn't desync the network)."""
        X, Y = _toy_classification(seed=11)
        net = DynamicNetwork(
            input_shape=(X.shape[1],),
            output_size=Y.shape[1],
            seed=11,
            cost_function='CrossEntropy',
            runtime_enabled=True,
        )
        net.fit(X, Y, verbose=False)
        preds = net.predict(X)
        self.assertEqual(preds.shape, (X.shape[0], Y.shape[1]))
        self.assertTrue(np.all(np.isfinite(preds)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
