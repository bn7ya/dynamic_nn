import torch  # noqa: F401  -- preload libtorch so _enn_core can resolve

from elasticneuralnetwork import _enn_core

from elasticneuralnetwork.controllers import (
    AdaptiveLRConfig,
    AdaptiveLRController,
    AdaptiveLRState,
    LayerManagerConfig,
    PlateauConfig,
    PlateauDetector,
    StabilityConfig,
    StabilityMonitor,
    StabilityReport,
    StabilityState,
    TrainableConfig,
)
from elasticneuralnetwork.modules import (
    AdaptiveConv2d,
    ElasticNetwork,
    HiddenActivation,
    ReversibleLinear,
    ReversibleNetwork,
    ReversibleNetworkConfig,
)
from elasticneuralnetwork.training import (
    BatchConfig,
    BatchManager,
    CostFunction,
    DatasetStatistics,
    EarlyStopping,
    EarlyStoppingConfig,
    PhaseController,
    PhaseScheduleConfig,
    PruningConfig,
    TrainingResult,
    compute_dataset_statistics,
    derive_adaptive_lr_config,
    derive_phase_schedule,
    derive_plateau_config,
    derive_pruning_config,
    derive_stability_config,
)


__all__ = [
    "AdaptiveConv2d",
    "AdaptiveLRConfig",
    "AdaptiveLRController",
    "AdaptiveLRState",
    "BatchConfig",
    "BatchManager",
    "CostFunction",
    "DatasetStatistics",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ElasticNetwork",
    "HiddenActivation",
    "LayerManagerConfig",
    "PhaseController",
    "PhaseScheduleConfig",
    "PlateauConfig",
    "PlateauDetector",
    "PruningConfig",
    "ReversibleLinear",
    "ReversibleNetwork",
    "ReversibleNetworkConfig",
    "StabilityConfig",
    "StabilityMonitor",
    "StabilityReport",
    "StabilityState",
    "TrainableConfig",
    "TrainingResult",
    "compute_dataset_statistics",
    "derive_adaptive_lr_config",
    "derive_phase_schedule",
    "derive_plateau_config",
    "derive_pruning_config",
    "derive_stability_config",
]

__version__ = "0.1.0"
