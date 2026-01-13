"""
Configuration dataclasses for the Dynamic MoE model.

Follows patterns from python/pydnn/network.py for consistency with the DynamicNetwork framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExpertConfig:
    """Configuration for individual expert networks."""
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = 'relu'
    dropout: float = 0.1

    # Efficiency tracking (like DynamicNetwork nodes)
    initial_efficiency: float = 0.5
    efficiency_decay: float = 0.9
    efficiency_update_scale: float = 0.1


@dataclass
class GatingConfig:
    """Configuration for the gating/router network."""
    num_experts: int = 4
    top_k: int = 2  # Number of experts to activate per token
    noise_std: float = 0.1  # Noise for load balancing exploration
    temperature: float = 1.0  # Softmax temperature

    # Exploration noise schedule
    exploration_noise_std: float = 0.5  # High noise during Phase 1
    main_noise_std: float = 0.1  # Lower noise during Phase 3
    finetune_noise_std: float = 0.0  # No noise during Phase 4


@dataclass
class LoadBalanceConfig:
    """Configuration for load balancing using efficiency metrics."""
    target_load: float = 0.25  # 1/num_experts for uniform (will be auto-computed)
    load_balance_weight: float = 0.01  # Auxiliary loss weight
    use_efficiency_weights: bool = True  # Use DNN-style efficiency tracking

    # Load balance thresholds
    min_expert_load: float = 0.05  # Minimum acceptable load per expert
    max_expert_load: float = 0.50  # Maximum acceptable load per expert


@dataclass
class MoEHealthConfig:
    """
    Health monitoring for MoE (extending DNN health concept).

    Cancer: One expert dominating (getting too much traffic)
    Alzheimer: Expert becoming inactive (not being used)
    """
    # Expert "cancer" - one expert dominating (overused)
    expert_cancer_threshold: float = 0.5  # If expert gets >50% of traffic

    # Expert "alzheimer" - expert becoming inactive
    expert_alzheimer_threshold: float = 0.05  # If expert gets <5% of traffic

    # Health state thresholds
    healthy_threshold: float = 0.3
    at_risk_threshold: float = 0.7

    # Intervention thresholds
    rebalance_threshold: float = 0.6  # Trigger rebalancing if health drops below


@dataclass
class MoETrainingPhaseConfig:
    """Configuration for 4-phase training approach (adapted for MoE)."""
    # Phase 1: Exploration - discover expert specializations
    exploration_epochs: int = 10
    exploration_learning_rate: float = 0.05
    exploration_batch_size: int = 64

    # Phase 2: Estimation - estimate training epochs
    estimation_epochs: int = 10
    estimation_learning_rate: float = 0.01

    # Phase 3: Main Training - reward/penalty adaptive LR
    main_learning_rate: float = 0.01
    main_batch_size: int = 32

    # Phase 4: Fine-tuning - frozen routing
    phase4_enabled: bool = True
    phase4_learning_rate: float = 0.005
    phase4_min_learning_rate: float = 0.0001
    phase4_lr_decay_rate: float = 0.95
    phase4_lr_decay_interval: int = 10
    phase4_batch_size: int = 64
    phase4_min_epochs: int = 20
    phase4_max_epochs: int = 100
    phase4_patience: int = 15

    # Epoch estimation
    target_efficiency: float = 0.8
    min_estimated_epochs: int = 30
    max_estimated_epochs: int = 200


@dataclass
class RewardPenaltyConfig:
    """Configuration for the reward/penalty system (from DynamicNetwork)."""
    cost_improvement_threshold: float = 0.005  # 0.5%
    efficiency_improvement_threshold: float = 0.02  # 2%
    load_balance_improvement_threshold: float = 0.01  # 1% improvement in balance

    min_learning_rate: float = 1e-5
    max_learning_rate: float = 0.5
    baseline_learning_rate: float = 0.1

    max_adjustment_factor: float = 1.5
    min_adjustment_factor: float = 0.7

    extreme_threshold: float = 0.7
    moderate_threshold: float = 0.6
    window_size: int = 15


@dataclass
class DynamicMoEConfig:
    """Full configuration for Dynamic MoE model."""
    # Model dimensions
    vocab_size: int = 10000
    embed_dim: int = 256
    max_seq_len: int = 128

    # Random seed
    seed: int = 42

    # Sub-configurations
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    load_balance: LoadBalanceConfig = field(default_factory=LoadBalanceConfig)
    health: MoEHealthConfig = field(default_factory=MoEHealthConfig)
    training_phase: MoETrainingPhaseConfig = field(default_factory=MoETrainingPhaseConfig)
    reward_penalty: RewardPenaltyConfig = field(default_factory=RewardPenaltyConfig)

    # Gradient clipping
    max_grad_norm: float = 1.0

    def __post_init__(self):
        """Auto-compute dependent values."""
        # Target load should be 1/num_experts for uniform distribution
        self.load_balance.target_load = 1.0 / self.gating.num_experts


@dataclass
class MoETrainingResult:
    """Result of MoE training with comprehensive diagnostics."""
    success: bool
    epochs_completed: int
    final_cost: float
    final_perplexity: float
    best_cost: float
    stopping_reason: str

    # History
    cost_history: List[float] = field(default_factory=list)
    perplexity_history: List[float] = field(default_factory=list)
    efficiency_history: List[float] = field(default_factory=list)

    # Expert metrics
    expert_load_history: List[List[float]] = field(default_factory=list)
    expert_efficiency_history: List[List[float]] = field(default_factory=list)

    # Health metrics
    cancer_score_history: List[float] = field(default_factory=list)
    alzheimer_score_history: List[float] = field(default_factory=list)

    # Emotional state (from DNN)
    total_rewards: int = 0
    total_penalties: int = 0
    learning_rate_history: List[float] = field(default_factory=list)

    # Phase metrics
    phase_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoEHealthReport:
    """Health report for MoE model."""
    state: str  # 'healthy', 'at_risk', 'critical'
    cancer_score: float  # Expert domination score
    alzheimer_score: float  # Expert death score
    overall_health: float

    # Per-expert health
    expert_loads: List[float] = field(default_factory=list)
    expert_efficiencies: List[float] = field(default_factory=list)

    # Diagnosis
    diagnosis: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Emotional state
    depression_ratio: float = 0.0
    excitement_ratio: float = 0.0
