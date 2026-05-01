"""Unit tests for `pydnn.dynamic_thresholds`.

Run from the repo root:
    python3 -m pytest python/tests/test_dynamic_thresholds.py -v
or just:
    python3 python/tests/test_dynamic_thresholds.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydnn.dynamic_thresholds import (  # noqa: E402
    CLAMPS,
    MAPPING_TABLE,
    DataSignals,
    apply_to_trainer_config,
    compute_data_signals,
    derive_thresholds,
)


def _make_classification(n=200, d=8, n_classes=4, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32) * scale
    y_idx = rng.integers(0, n_classes, size=n)
    y = np.eye(n_classes)[y_idx].astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------


def test_signals_are_in_unit_interval_for_typical_data():
    X, y = _make_classification()
    sig = compute_data_signals(X, y)
    assert 0.0 <= sig.variance_score <= 1.0
    assert 0.0 <= sig.complexity_score <= 1.0
    assert not sig.fallback_used


def test_high_variance_data_yields_higher_variance_score():
    X_lo, y = _make_classification(scale=0.05, seed=1)
    X_hi, _ = _make_classification(scale=10.0, seed=1)
    s_lo = compute_data_signals(X_lo, y)
    s_hi = compute_data_signals(X_hi, y)
    assert s_hi.variance_score > s_lo.variance_score, (
        s_lo.variance_score, s_hi.variance_score,
    )


def test_high_dim_data_increases_complexity_score():
    X_lo, y = _make_classification(d=4, seed=2)
    X_hi, _ = _make_classification(d=64, seed=2)
    s_lo = compute_data_signals(X_lo, y)
    s_hi = compute_data_signals(X_hi, y)
    assert s_hi.complexity_score >= s_lo.complexity_score


def test_tiny_dataset_falls_back_to_neutral():
    X, y = _make_classification(n=10, seed=3)
    sig = compute_data_signals(X, y)
    assert sig.fallback_used is True
    assert sig.variance_score == 0.5
    assert sig.complexity_score == 0.5
    assert derive_thresholds(sig) == {}


def test_regression_targets_handled():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = rng.standard_normal((300,)).astype(np.float32)  # 1D regression
    sig = compute_data_signals(X, y)
    assert 0.0 <= sig.complexity_score <= 1.0
    derived = derive_thresholds(sig)
    assert derived  # not empty


def test_regression_fisher_ratio_is_none_not_nan():
    """Fisher is undefined for regression — raw_stats must store None
    (JSON-serialisable) rather than NaN (which poisons downstream ops)."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = rng.standard_normal((300,)).astype(np.float32)
    sig = compute_data_signals(X, y)
    # Aggregator prefixes complexity-keys with "comp_".
    assert "comp_fisher_ratio" in sig.raw_stats
    assert sig.raw_stats["comp_fisher_ratio"] is None
    # Regression-style raw_stats must remain JSON-encodable.
    import json
    json.dumps(sig.raw_stats)  # would raise on NaN


def test_single_class_classification_fisher_is_none():
    """Single-class one-hot inputs collapse Fisher (no between-class
    variance). The signal must surface this as None and never as NaN."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    # Every row labelled class 0 — only one class present.
    y = np.zeros((100, 3), dtype=np.float32)
    y[:, 0] = 1.0
    sig = compute_data_signals(X, y)
    assert sig.raw_stats.get("comp_fisher_ratio") is None
    import json
    json.dumps(sig.raw_stats)  # JSON-safe


# ---------------------------------------------------------------------------
# derive_thresholds clamping & mapping coverage
# ---------------------------------------------------------------------------


def test_every_mapping_has_a_clamp():
    for key in MAPPING_TABLE:
        assert key in CLAMPS, f"missing clamp for {key}"


def test_derived_values_respect_clamps_at_extremes():
    for v in (0.0, 1.0):
        for c in (0.0, 1.0):
            sig = DataSignals(variance_score=v, complexity_score=c)
            derived = derive_thresholds(sig)
            for key, value in derived.items():
                lo, hi, _ = CLAMPS[key]
                assert lo <= float(value) <= hi, (key, value, lo, hi)


def test_integer_thresholds_are_actual_ints():
    sig = DataSignals(variance_score=0.5, complexity_score=0.5)
    derived = derive_thresholds(sig)
    for key in (
        "patience",
        "min_epochs_for_early_stop",
        "batch_config.min_batch_size",
        "batch_config.max_batch_size",
    ):
        assert isinstance(derived[key], int), (key, type(derived[key]))


def test_complexity_drives_patience_monotonically():
    p_lo = derive_thresholds(DataSignals(variance_score=0.5, complexity_score=0.0))["patience"]
    p_hi = derive_thresholds(DataSignals(variance_score=0.5, complexity_score=1.0))["patience"]
    assert p_hi > p_lo


def test_variance_drives_gradient_clip_monotonically():
    g_lo = derive_thresholds(DataSignals(variance_score=0.0, complexity_score=0.5))["gradient_clip_value"]
    g_hi = derive_thresholds(DataSignals(variance_score=1.0, complexity_score=0.5))["gradient_clip_value"]
    assert g_hi > g_lo


# ---------------------------------------------------------------------------
# apply_to_trainer_config (uses a dummy stub so tests don't need the C++ ext)
# ---------------------------------------------------------------------------


class _SubCfg:
    pass


class _Cfg:
    def __init__(self):
        self.cancer_threshold = 0.7
        self.patience = 30
        self.gradient_clip_value = 1.0
        self.layer_manager_config = _SubCfg()
        self.layer_manager_config.efficiency_threshold = 0.5
        self.batch_config = _SubCfg()
        self.batch_config.min_batch_size = 4


def test_apply_to_trainer_config_writes_nested_paths():
    cfg = _Cfg()
    derived = {
        "cancer_threshold": 0.42,
        "patience": 17,
        "gradient_clip_value": 0.9,
        "layer_manager_config.efficiency_threshold": 0.33,
        "batch_config.min_batch_size": 12,
    }
    apply_to_trainer_config(cfg, derived)
    assert cfg.cancer_threshold == 0.42
    assert cfg.patience == 17
    assert math.isclose(cfg.gradient_clip_value, 0.9)
    assert math.isclose(cfg.layer_manager_config.efficiency_threshold, 0.33)
    assert cfg.batch_config.min_batch_size == 12


# ---------------------------------------------------------------------------
# Standalone runner so the file works without pytest.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        sig = inspect.signature(fn)
        if sig.parameters:
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {exc!r}")
    if failures:
        print(f"\n{failures} failures")
        sys.exit(1)
    print("\nAll tests passed.")
