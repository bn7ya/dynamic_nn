from __future__ import annotations

from typing import Optional

from elasticneuralnetwork._enn_core import controllers as _c


PlateauConfig = _c.PlateauConfig
PlateauDetector = _c.PlateauDetector

StabilityState = _c.StabilityState
StabilityConfig = _c.StabilityConfig
StabilityReport = _c.StabilityReport
StabilityMonitor = _c.StabilityMonitor
stability_state_name = _c.stability_state_name

AdaptiveLRConfig = _c.AdaptiveLRConfig
AdaptiveLRState = _c.AdaptiveLRState
step_adaptive_lr_controller = _c.step_adaptive_lr_controller
sigmoid_threshold = _c.sigmoid_threshold

LayerManagerConfig = _c.LayerManagerConfig
TrainableConfig = _c.TrainableConfig


class AdaptiveLRController:
    def __init__(self, config: Optional[AdaptiveLRConfig] = None) -> None:
        self.config = config if config is not None else AdaptiveLRConfig()
        self.state = AdaptiveLRState()

    def step(self, learning_rate, cost_history, utilization_history):
        return step_adaptive_lr_controller(
            learning_rate,
            list(cost_history),
            list(utilization_history),
            self.config,
            self.state,
        )
