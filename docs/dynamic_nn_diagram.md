# elasticneuralnetwork — PyTorch extension architecture

## High-level overview

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                        elasticneuralnetwork — PyTorch C++ extension                    │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   Python                                              C++ (libtorch)                   │
│   ─────────────────────────────                       ─────────────────────────────    │
│                                                                                        │
│   ReversibleNetwork  ◀──── pybind11 ────▶  torch::nn::Module                           │
│   ReversibleLinear   ◀──── pybind11 ────▶    ReversibleNetworkImpl                     │
│   AdaptiveConv2d     ◀──── pybind11 ────▶    ReversibleLinearImpl                      │
│   PhaseController    ◀──── pybind11 ────▶    AdaptiveConv2dImpl                        │
│                                              PhaseController                           │
│                                                                                        │
│   torch.optim.Adam ────────────▶ net.parameters() ◀────── register_parameter(...)      │
│   torch.utils.data.DataLoader                                                          │
│                                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Built via `torch.utils.cpp_extension.CUDAExtension` (CPU fallback
`CppExtension`); the C++ TUs link into a single `_enn_core.so` that
ships alongside the Python package.

## Module hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ReversibleNetwork                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐ │
│   │   INPUT     │    │  ReversibleLinear │    │  ReversibleLinear │    │  OUTPUT    │ │
│   │   (X)       │───▶│  + ReLU           │───▶│  + ReLU           │───▶│  logits    │ │
│   │             │    │  active_mask: 1s  │    │  active_mask: 1s  │    │            │ │
│   │ fixed shape │    │  weight, bias     │    │  weight, bias     │    │ fixed shape│ │
│   └─────────────┘    └──────────────────┘    └──────────────────┘    └────────────┘ │
│                              │                        │                              │
│                              │  add_nodes / prune_nodes / compact                    │
│                              ▼                        ▼                              │
│                       ┌────────────────────────────────────┐                         │
│                       │            LayerManager            │                         │
│                       │  • PlateauDetector gate            │                         │
│                       │  • StabilityMonitor gate           │                         │
│                       │  • analyze_with_utilization → Action│                        │
│                       │  • execute(decision) → mutations    │                        │
│                       └────────────────────────────────────┘                         │
│                                       │                                              │
│                       ┌───────────────┴───────────────┐                              │
│                       ▼                               ▼                              │
│              ┌────────────────────┐         ┌────────────────────┐                  │
│              │  StabilityMonitor  │         │ AdaptiveLRController│                 │
│              │  growth_anomaly    │         │  improvement_signals │                 │
│              │  capacity_loss     │         │  regression_signals  │                 │
│              │  states:           │         │  step() →            │                 │
│              │   Stable           │         │   reward / penalty / │                 │
│              │   ExcessiveGrowth  │         │   reset              │                 │
│              │   PathologicalGrow │         └────────────────────┘                  │
│              │   ExcessivePrune   │                                                  │
│              │   PathologicalPrune│                                                  │
│              │   Critical         │                                                  │
│              └────────────────────┘                                                  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

The arrows above pointing at the inner box are the only direction
mutation flows. `StabilityMonitor` and `AdaptiveLRController` are
read-only observers; they consume scalar histories and emit
decisions, never reaching into the `torch::nn::Module` graph
themselves.

## Four-phase training schedule

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            PhaseController.fit(X, y)                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   compute_dataset_statistics(X, y)                                                   │
│        │                                                                             │
│        ▼                                                                             │
│   derive_phase_schedule / derive_adaptive_lr_config / derive_stability_config        │
│        │                                                                             │
│        ▼                                                                             │
│   ┌──────────────────────┐                                                           │
│   │  TopologyDiscovery   │  LR = schedule.topology_discovery_lr                      │
│   │                      │  optimizer = torch::optim::Adam                           │
│   │  plateau → grow      │  hook: PlateauDetector + LayerManager.auto_adjust()       │
│   └──────────┬───────────┘                                                           │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                           │
│   │ ConvergenceRateEst.  │  LR = schedule.convergence_estimation_lr                  │
│   │                      │  fixed-LR training to read off the cost trend             │
│   └──────────┬───────────┘                                                           │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                           │
│   │   AdaptiveTraining   │  LR = adapted by AdaptiveLRController per epoch           │
│   │                      │  reward / penalty / reset on cost+utilization trends      │
│   └──────────┬───────────┘                                                           │
│              ▼                                                                       │
│   ┌──────────────────────┐                                                           │
│   │ FrozenArchitecture-  │  LR = schedule.frozen_finetune_lr                         │
│   │ Finetuning           │  topology locked, weights move                            │
│   └──────────┬───────────┘                                                           │
│              ▼                                                                       │
│   net.compact()  ← only hard erasure of soft-pruned rows / layers                    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Every per-epoch loop inside a phase is the standard PyTorch shape:
`zero_grad` → `forward` → `cross_entropy` → `backward` → `step`. The
optimizer is `torch::optim::Adam`; the loss is
`torch::nn::functional::cross_entropy` (classification) or
`mse_loss` (regression).

## Data-driven thresholds

```
DatasetStatistics
├── feature_variance_mean        ┐
├── feature_variance_spread      │
├── signal_to_noise_ratio        │     derive_*  ─────▶  AdaptiveLRConfig
├── label_entropy                ├──── functions          StabilityConfig
├── effective_rank               │                        PruningConfig
└── fisher_separability          ┘                        PhaseScheduleConfig
                                                          PlateauConfig
```

Every derived threshold is either `mean + k·σ` (k ∈ {1, 2, 3}, σ-
coverage), a named percentile, `ceil(sqrt(n))` (window-size rule of
thumb), or `sigmoid(z, k=5)` (maps `z ≈ ±2` to ≈ 0.9). No bare
multiplicative constants live in code.

## Soft-delete invariant

```
   prune_nodes([3, 17])      add_nodes(2)              compact()
   ──────────────────▶       ──────────────▶           ──────────▶
   active_mask: [1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
                              ▲                       ▲
                              │                       │
                              │                       │
                  reactivates slots                hard removes
                  3 and 17                         remaining inactive
                  (weights preserved)              rows; weights gone
```

`topology_version()` increments on every call, including no-op
`compact()`.

## Where the C++/Python boundary lives

| Side | Lives in | Knows about |
|---|---|---|
| Python | `elasticneuralnetwork/*.py` | `torch.nn.Module`, `torch.optim`, `DataLoader`, numpy. Composes the bindings; provides `ElasticNetwork` convenience wrapper. |
| Bindings | `csrc/bindings/*.cpp` | pybind11. Exposes C++ classes as Python types preserving the `torch.nn.Module` shape (`m.parameters()`, `m.buffers()` work). Releases the GIL around `PhaseController::fit`. |
| C++ | `csrc/{modules,controllers,training}/*.cpp` | libtorch (`torch::Tensor`, `torch::nn::Module`, `torch::optim`, `torch::nn::functional::*`). Implements the reversible-topology semantics. |
