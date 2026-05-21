import math

import pytest

from elasticneuralnetwork import (
    AdaptiveLRConfig,
    AdaptiveLRController,
    AdaptiveLRState,
    PlateauConfig,
    PlateauDetector,
    StabilityConfig,
    StabilityMonitor,
    StabilityState,
)


def test_plateau_detector_flat_signal_is_local_max():
    cfg = PlateauConfig()
    cfg.window = 10
    cfg.warmup_epochs = 5
    cfg.cooldown_epochs = 0
    cfg.slope_epsilon = 1e-3
    d = PlateauDetector(cfg)
    for _ in range(20):
        d.tick(0.5)
    assert d.is_at_local_max()
    assert abs(d.slope()) < 1e-3


def test_plateau_detector_rising_signal_not_local_max():
    cfg = PlateauConfig()
    cfg.window = 10
    cfg.warmup_epochs = 5
    cfg.cooldown_epochs = 0
    cfg.slope_epsilon = 1e-3
    d = PlateauDetector(cfg)
    for i in range(20):
        d.tick(i * 0.1)
    assert not d.is_at_local_max()
    assert d.slope() > 0


def test_plateau_window_too_small_raises():
    cfg = PlateauConfig()
    cfg.window = 2
    with pytest.raises(Exception):
        PlateauDetector(cfg)


def test_stability_monitor_stable_initial():
    cfg = StabilityConfig()
    m = StabilityMonitor(3, 30, cfg)
    assert m.current_state() == StabilityState.Stable


def test_stability_monitor_pathological_growth_after_many_adds():
    cfg = StabilityConfig()
    cfg.history_window = 20
    m = StabilityMonitor(2, 10, cfg)
    for i in range(20):
        m.update_epoch(i)
        m.record_change(1, 5)
        m.update_topology(2 + i, 10 + 5 * i)
    s = m.current_state()
    assert s in (
        StabilityState.PathologicalGrowth,
        StabilityState.ExcessiveGrowthRisk,
    )


def test_stability_monitor_critical_when_layers_below_min():
    cfg = StabilityConfig()
    cfg.min_layers = 2
    m = StabilityMonitor(1, 4, cfg)
    assert m.current_state() == StabilityState.Critical


def test_adaptive_lr_controller_rewards_improvement():
    cfg = AdaptiveLRConfig()
    ctl = AdaptiveLRController(cfg)
    cost_hist = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    util_hist = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    new_lr, action = ctl.step(cfg.baseline_learning_rate, cost_hist,
                                 util_hist)
    assert action in ("reward", "none")


def test_adaptive_lr_state_rates_sum_to_one():
    s = AdaptiveLRState()
    s.improvement_signals = 3
    s.regression_signals = 7
    assert abs(s.improvement_rate() + s.regression_rate() - 1.0) < 1e-9
