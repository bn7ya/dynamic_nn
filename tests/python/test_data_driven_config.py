import math

import torch

from elasticneuralnetwork import (
    compute_dataset_statistics,
    derive_adaptive_lr_config,
    derive_phase_schedule,
    derive_plateau_config,
    derive_pruning_config,
    derive_stability_config,
)


def test_dataset_statistics_gaussian():
    torch.manual_seed(0)
    X = torch.randn(256, 16)
    y = torch.randint(0, 4, (256,)).long()
    s = compute_dataset_statistics(X, y)
    assert s.num_samples == 256
    assert s.num_features == 16
    assert s.num_classes == 4
    assert not s.is_regression
    assert 0.0 <= s.label_entropy <= 1.0
    assert s.effective_rank > 0


def test_dataset_statistics_regression():
    torch.manual_seed(0)
    X = torch.randn(128, 8)
    y = torch.randn(128, 1)
    s = compute_dataset_statistics(X, y)
    assert s.is_regression
    assert s.fisher_separability == -1.0


def test_derived_configs_are_finite_and_positive():
    torch.manual_seed(0)
    X = torch.randn(200, 12)
    y = torch.randint(0, 3, (200,)).long()
    s = compute_dataset_statistics(X, y)
    lr = derive_adaptive_lr_config(s)
    assert lr.baseline_learning_rate > 0
    assert lr.min_learning_rate < lr.max_learning_rate
    assert 0 < lr.critical_imbalance_threshold < 1
    st = derive_stability_config(s)
    assert 0 < st.growth_anomaly_threshold < 1
    assert 0 < st.capacity_loss_threshold < 1
    pr = derive_pruning_config(s)
    assert 0 < pr.prune_utilization_threshold < 1
    assert 0 < pr.growth_utilization_threshold < 1
    ps = derive_phase_schedule(s)
    assert ps.topology_discovery_epochs > 0
    assert ps.adaptive_training_epochs > 0
    pl = derive_plateau_config(s)
    assert pl.window >= 3


def test_derived_phase_lr_schedule_monotonic_decreasing():
    torch.manual_seed(0)
    X = torch.randn(128, 8)
    y = torch.randint(0, 2, (128,)).long()
    s = compute_dataset_statistics(X, y)
    ps = derive_phase_schedule(s)
    assert ps.convergence_estimation_lr <= ps.topology_discovery_lr
    assert ps.adaptive_training_lr_floor <= (
        ps.adaptive_training_lr_initial)
    assert ps.frozen_finetune_lr <= ps.adaptive_training_lr_initial
