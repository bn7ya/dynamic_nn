"""
Mixture-of-Experts (MoE) Model with Dynamic Neural Network Integration.

This module implements a MoE architecture that extends the DynamicNetwork framework,
leveraging efficiency tracking, health monitoring, and 4-phase training.
"""

from .moe_config import (
    ExpertConfig,
    GatingConfig,
    LoadBalanceConfig,
    MoEHealthConfig,
    MoETrainingPhaseConfig,
    DynamicMoEConfig,
)
from .expert import Expert
from .gating import GatingNetwork
from .moe_layer import MoELayer
from .dynamic_moe import DynamicMoE
from .tokenizer import SimpleTokenizer
from .data_loader import SlimPajamaLoader
from .trainer import MoETrainer
from .evaluator import MoEEvaluator

__all__ = [
    # Configs
    'ExpertConfig',
    'GatingConfig',
    'LoadBalanceConfig',
    'MoEHealthConfig',
    'MoETrainingPhaseConfig',
    'DynamicMoEConfig',
    # Core components
    'Expert',
    'GatingNetwork',
    'MoELayer',
    'DynamicMoE',
    # Data
    'SimpleTokenizer',
    'SlimPajamaLoader',
    # Training
    'MoETrainer',
    'MoEEvaluator',
]
