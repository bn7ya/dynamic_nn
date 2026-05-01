"""Dataset-driven dynamic thresholds for DynamicNetwork training.

Computes two scalars from the training inputs/targets — `variance_score`
and `complexity_score` — and uses them to derive every Tier-1 threshold in
TrainerConfig. The MAPPING_TABLE is the single source of truth for how a
signal becomes a threshold value; review and tune coefficients there.

When DynamicNetwork(dynamic_thresholds=True) — the default — `_fit_cpp()`
calls into this module before constructing the C++ TrainerConfig. With
runtime_enabled=True the seeded values flow into RuntimeAdaptiveConfig
via apply_static_config and continue to be nudged by the observer
(Tier 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import math
import numpy as np

# --- normalisation -----------------------------------------------------------

# Sigmoid form mirrors `_compute_efficiency()` in network.py:1747-1769 so
# variance/complexity scores live in the same [0, 1] space as the rest of
# the pipeline's adaptive scalars.
_SIGMOID_K = 5.0


def _sigmoid_score(raw: float, *, midpoint: float = 1.0) -> float:
    """Map a non-negative raw stat into [0, 1].

    `raw / midpoint = 1` lands on 0.5; growth past the midpoint saturates
    smoothly toward 1.0.
    """
    if not math.isfinite(raw) or raw <= 0.0:
        return 0.0
    scaled = (raw / max(midpoint, 1e-12)) - 1.0
    return float(1.0 / (1.0 + math.exp(-_SIGMOID_K * scaled / 5.0)))


# --- signals ----------------------------------------------------------------


@dataclass
class DataSignals:
    """The derived dataset signals plus the threshold dict they produced.

    Attached to `DynamicNetwork._last_data_signals` after each fit() so
    callers can inspect what was computed and (when verbose) why a given
    threshold ended up where it did.
    """

    variance_score: float
    complexity_score: float
    # Values are floats except for fields that are intentionally undefined
    # (e.g. fisher_ratio for regression / single-class inputs), which are
    # represented as None rather than NaN so the dict stays JSON-serialisable.
    raw_stats: Dict[str, Optional[float]] = field(default_factory=dict)
    derived_thresholds: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    notes: str = ""


# Datasets smaller than this fall back to static defaults — the SVD and
# Fisher proxies are too noisy to trust below it.
_MIN_SAMPLES_FOR_SIGNALS = 50

# Cap for the SVD effective-rank computation. SVD is O(min(n,d)^2 * max(n,d));
# bounding `n` keeps the one-shot cost negligible for any d we care about.
_SVD_SAMPLE_CAP = 1000


def _flatten_features(x: np.ndarray) -> np.ndarray:
    """Reshape (N, *feature_dims) into (N, prod(feature_dims))."""
    return x.reshape(x.shape[0], -1) if x.ndim > 2 else x


def _compute_variance_signal(
    x_flat: np.ndarray,
    *,
    input_std: Optional[np.ndarray] = None,
    input_mean: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute variance_score in [0, 1] plus the raw stats that fed it."""
    if input_std is None:
        std = x_flat.std(axis=0)
    else:
        std = np.asarray(input_std, dtype=np.float64).reshape(-1)
    if input_mean is None:
        mean = x_flat.mean(axis=0)
    else:
        mean = np.asarray(input_mean, dtype=np.float64).reshape(-1)

    feature_var = std.astype(np.float64) ** 2
    feature_var_mean = float(feature_var.mean())
    feature_var_spread = float(
        feature_var.std() / (feature_var_mean + 1e-12)
    )
    abs_mean = float(np.mean(np.abs(x_flat))) + 1e-12
    snr = feature_var_mean / abs_mean

    raw = {
        "feature_var_mean": feature_var_mean,
        "feature_var_spread": feature_var_spread,
        "snr": snr,
        "abs_mean": abs_mean,
    }

    # Aggregate the three pieces with weights that reflect their reliability.
    score = 0.5 * _sigmoid_score(feature_var_mean, midpoint=1.0) \
          + 0.3 * _sigmoid_score(snr,             midpoint=1.0) \
          + 0.2 * _sigmoid_score(feature_var_spread, midpoint=0.5)
    score = max(0.0, min(1.0, score))
    return score, raw


def _label_entropy(y: np.ndarray) -> Tuple[float, str]:
    """Return entropy in nats. Heuristically detect classification vs regression."""
    if y.ndim == 1:
        # 1-D label vector: assume integer class indices if dtype is int.
        if np.issubdtype(y.dtype, np.integer):
            _, counts = np.unique(y, return_counts=True)
            probs = counts.astype(np.float64) / counts.sum()
            return float(-(probs * np.log(probs + 1e-12)).sum()), "classification"
        # Float regression target — entropy of a histogram approximation.
        hist, _ = np.histogram(y, bins=min(32, max(4, y.size // 8)))
        probs = hist.astype(np.float64) / max(1, hist.sum())
        probs = probs[probs > 0]
        return float(-(probs * np.log(probs)).sum()), "regression"

    # 2-D: one-hot or soft labels.
    if y.ndim == 2 and np.allclose(y.sum(axis=1), 1.0, atol=1e-3):
        marg = y.sum(axis=0)
        marg = marg / max(marg.sum(), 1e-12)
        marg = marg[marg > 0]
        return float(-(marg * np.log(marg)).sum()), "classification"

    # Generic regression-ish fallback: per-output histogram entropy.
    flat = y.reshape(-1)
    hist, _ = np.histogram(flat, bins=min(32, max(4, flat.size // 8)))
    probs = hist.astype(np.float64) / max(1, hist.sum())
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum()), "regression"


def _effective_rank(x_flat: np.ndarray) -> float:
    """Singular-value entropy in [0, log(min(n,d))]; normalised in caller."""
    n_samples = x_flat.shape[0]
    if n_samples > _SVD_SAMPLE_CAP:
        # Deterministic stride sampling so signals are reproducible.
        idx = np.linspace(0, n_samples - 1, _SVD_SAMPLE_CAP, dtype=np.int64)
        sample = x_flat[idx]
    else:
        sample = x_flat
    sample = sample - sample.mean(axis=0, keepdims=True)
    if sample.size == 0 or np.all(sample == 0.0):
        return 0.0
    try:
        sv = np.linalg.svd(sample, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 0.0
    p = (sv ** 2) / (sv ** 2).sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _fisher_separability(
    x_flat: np.ndarray, y: np.ndarray
) -> Optional[float]:
    """Between-class / within-class variance ratio. None for regression."""
    if y.ndim == 2 and np.allclose(y.sum(axis=1), 1.0, atol=1e-3):
        labels = y.argmax(axis=1)
    elif y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
        labels = y
    else:
        return None  # regression or unknown

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        return None

    overall_mean = x_flat.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls, cnt in zip(classes, counts):
        members = x_flat[labels == cls]
        if members.size == 0:
            continue
        cls_mean = members.mean(axis=0)
        between += float(cnt) * float(np.sum((cls_mean - overall_mean) ** 2))
        within += float(np.sum((members - cls_mean) ** 2))

    if within < 1e-12:
        return None
    return between / within


def _compute_complexity_signal(
    x_flat: np.ndarray, y: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    n_samples, n_features = x_flat.shape
    entropy, task_kind = _label_entropy(y)
    eff_rank_entropy = _effective_rank(x_flat)
    max_rank_entropy = math.log(min(n_samples, n_features) + 1.0) + 1e-12
    eff_rank_norm = eff_rank_entropy / max_rank_entropy

    fisher = _fisher_separability(x_flat, y)
    # High Fisher → easy separation → LOW complexity. Map inversely.
    if fisher is None:
        sep_score = 0.5  # neutral when undefined
    else:
        sep_score = 1.0 - _sigmoid_score(fisher, midpoint=2.0)

    dim_ratio = n_features / max(math.log(max(n_samples, 2)), 1.0)

    raw = {
        "label_entropy": entropy,
        "task_kind": 1.0 if task_kind == "classification" else 0.0,
        "effective_rank_norm": float(eff_rank_norm),
        # None (not NaN) signals "undefined" so the dict stays JSON-serialisable
        # and downstream numeric ops can branch cleanly. Fisher is undefined for
        # regression and single-class inputs.
        "fisher_ratio": float(fisher) if fisher is not None else None,
        "separability_score": sep_score,
        "dim_ratio": float(dim_ratio),
    }

    entropy_norm = _sigmoid_score(entropy, midpoint=1.0)
    rank_score = max(0.0, min(1.0, eff_rank_norm))
    dim_score = _sigmoid_score(dim_ratio, midpoint=2.0)

    score = (
        0.30 * entropy_norm
        + 0.30 * rank_score
        + 0.25 * sep_score
        + 0.15 * dim_score
    )
    return max(0.0, min(1.0, score)), raw


def compute_data_signals(
    X: np.ndarray,
    y: np.ndarray,
    *,
    input_mean: Optional[np.ndarray] = None,
    input_std: Optional[np.ndarray] = None,
) -> DataSignals:
    """Compute (variance_score, complexity_score) and the derived thresholds.

    Pass already-computed normalisation stats from `_normalize_data` to
    avoid a redundant pass over `X`.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    x_flat = _flatten_features(X)

    n_samples = x_flat.shape[0]
    if n_samples < _MIN_SAMPLES_FOR_SIGNALS:
        return DataSignals(
            variance_score=0.5,
            complexity_score=0.5,
            raw_stats={"n_samples": float(n_samples)},
            fallback_used=True,
            notes=f"n_samples<{_MIN_SAMPLES_FOR_SIGNALS}, using neutral 0.5",
        )

    variance, var_raw = _compute_variance_signal(
        x_flat, input_mean=input_mean, input_std=input_std
    )
    complexity, comp_raw = _compute_complexity_signal(x_flat, y)

    raw = {"n_samples": float(n_samples), "n_features": float(x_flat.shape[1])}
    raw.update({f"var_{k}": v for k, v in var_raw.items()})
    raw.update({f"comp_{k}": v for k, v in comp_raw.items()})

    return DataSignals(
        variance_score=variance,
        complexity_score=complexity,
        raw_stats=raw,
    )


# --- mapping table ----------------------------------------------------------

# Each callable receives (variance, complexity) ∈ [0, 1] and returns a raw
# value. The clamp/cast is applied separately from CLAMPS.
MAPPING_TABLE: Dict[str, Callable[[float, float], float]] = {
    "cancer_threshold":            lambda v, c: 0.5 + 0.3 * (1.0 - c),
    "alzheimer_threshold":         lambda v, c: 0.5 + 0.3 * c,
    "patience":                    lambda v, c: 15.0 + 30.0 * c,
    "min_improvement":             lambda v, c: 1e-5 + 1e-3 * (1.0 - c),
    "min_epochs_for_early_stop":   lambda v, c: 5.0 + 15.0 * c,
    "gradient_clip_value":         lambda v, c: 0.5 + 1.5 * v,
    "phase4_min_improvement":      lambda v, c: 1e-6 + 1e-4 * (1.0 - c),

    # LayerManagerConfig
    "layer_manager_config.efficiency_threshold":
        lambda v, c: 0.3 + 0.3 * (1.0 - v),
    "layer_manager_config.saturation_threshold":
        lambda v, c: 0.5 + 0.3 * v,
    "layer_manager_config.redundancy_threshold":
        lambda v, c: 0.005 + 0.02 * (1.0 - c),

    # RewardPenaltyConfig
    "reward_penalty_config.cost_improvement_threshold":
        lambda v, c: 0.001 + 0.01 * (1.0 - c),
    "reward_penalty_config.efficiency_improvement_threshold":
        lambda v, c: 0.005 + 0.04 * (1.0 - c),
    "reward_penalty_config.extreme_threshold":
        lambda v, c: 0.5 + 0.3 * c,
    "reward_penalty_config.moderate_threshold":
        lambda v, c: 0.4 + 0.3 * c,

    # BatchConfig
    "batch_config.min_batch_size":
        lambda v, c: max(4.0, 8.0 * v + 4.0),
    "batch_config.max_batch_size":
        lambda v, c: max(64.0, 128.0 + 256.0 * v),
    "batch_config.growth_rate":
        lambda v, c: 1.2 + 0.6 * (1.0 - v),
}

# (min, max, cast) per threshold. The clamp ranges mirror the
# RuntimeAdaptiveConfig limits so observer-driven nudges later stay
# inside the same envelope.
CLAMPS: Dict[str, Tuple[float, float, Callable[[float], Any]]] = {
    "cancer_threshold":                                       (0.1, 0.99, float),
    "alzheimer_threshold":                                    (0.1, 0.99, float),
    "patience":                                               (1.0, 500.0, lambda x: int(round(x))),
    "min_improvement":                                        (1e-9, 1e-1, float),
    "min_epochs_for_early_stop":                              (1.0, 500.0, lambda x: int(round(x))),
    "gradient_clip_value":                                    (0.1, 10.0, float),
    "phase4_min_improvement":                                 (1e-9, 1e-1, float),
    "layer_manager_config.efficiency_threshold":              (0.05, 0.95, float),
    "layer_manager_config.saturation_threshold":              (0.05, 0.95, float),
    "layer_manager_config.redundancy_threshold":              (1e-4, 0.5, float),
    "reward_penalty_config.cost_improvement_threshold":       (1e-6, 0.5, float),
    "reward_penalty_config.efficiency_improvement_threshold": (1e-6, 0.5, float),
    "reward_penalty_config.extreme_threshold":                (0.5, 0.99, float),
    "reward_penalty_config.moderate_threshold":               (0.3, 0.95, float),
    "batch_config.min_batch_size":                            (1.0, 1024.0, lambda x: int(round(x))),
    "batch_config.max_batch_size":                            (4.0, 8192.0, lambda x: int(round(x))),
    "batch_config.growth_rate":                               (1.0, 4.0, float),
}


def derive_thresholds(signals: DataSignals) -> Dict[str, Any]:
    """Apply MAPPING_TABLE + CLAMPS, returning a {field_path: value} dict.

    Field paths use dotted notation for nested config structs
    (e.g. "layer_manager_config.efficiency_threshold").
    """
    if signals.fallback_used:
        return {}

    v = signals.variance_score
    c = signals.complexity_score
    out: Dict[str, Any] = {}
    for key, fn in MAPPING_TABLE.items():
        raw = fn(v, c)
        lo, hi, cast = CLAMPS[key]
        clamped = min(hi, max(lo, raw))
        out[key] = cast(clamped)
    signals.derived_thresholds = out
    return out


def apply_to_trainer_config(
    trainer_config: Any,
    derived: Dict[str, Any],
) -> None:
    """Write derived values into a `_dnn_core.TrainerConfig` instance.

    Walks the dotted field path so nested configs (LayerManagerConfig,
    RewardPenaltyConfig, BatchConfig) get the correct attribute set.
    """
    for path, value in derived.items():
        parts = path.split(".")
        target = trainer_config
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)
