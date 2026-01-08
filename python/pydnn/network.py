"""
High-level Python interface for Dynamic Neural Network.
"""

from typing import Tuple, List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed. Install with 'pip install tqdm' for progress bars.")


@dataclass
class TrainingResult:
    """Result of training with comprehensive diagnostics."""
    success: bool
    epochs_completed: int
    final_cost: float
    final_efficiency: float
    best_cost: float
    best_efficiency: float
    stopping_reason: str
    cost_history: List[float]
    efficiency_history: List[float]
    training_time_ms: int
    # New diagnostic fields
    nodes_added: int = 0
    nodes_removed: int = 0
    layers_added: int = 0
    layers_removed: int = 0
    cancer_score_history: List[float] = field(default_factory=list)
    alzheimer_score_history: List[float] = field(default_factory=list)
    architecture_history: List[Tuple[int, int]] = field(default_factory=list)
    perturbations_applied: int = 0
    phase_metrics: Dict[str, Any] = field(default_factory=dict)
    # Emotional state fields
    total_rewards: int = 0
    total_penalties: int = 0
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0
    learning_rate_history: List[float] = field(default_factory=list)
    emotional_state_history: List[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Network health report."""
    state: str
    cancer_score: float
    alzheimer_score: float
    overall_health: float
    diagnosis: str
    recommendations: List[str]
    current_layers: int
    current_nodes: int
    depression_ratio: float = 0.0
    excitement_ratio: float = 0.0
    emotional_state: str = "neutral"


@dataclass
class EmotionalState:
    """Tracks the network's emotional state during training."""
    total_rewards: int = 0
    total_penalties: int = 0
    reward_history: List[float] = field(default_factory=list)
    penalty_history: List[float] = field(default_factory=list)
    depression_history: List[float] = field(default_factory=list)
    excitement_history: List[float] = field(default_factory=list)
    lr_reset_count: int = 0

    @property
    def total_adjustments(self) -> int:
        return self.total_rewards + self.total_penalties

    @property
    def depression_ratio(self) -> float:
        """Ratio of penalties to total adjustments."""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_penalties / self.total_adjustments

    @property
    def excitement_ratio(self) -> float:
        """Ratio of rewards to total adjustments."""
        if self.total_adjustments == 0:
            return 0.0
        return self.total_rewards / self.total_adjustments


@dataclass
class RewardPenaltyConfig:
    """Configuration for the reward/penalty system."""
    cost_improvement_threshold: float = 0.001
    efficiency_improvement_threshold: float = 0.01
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1.0
    baseline_learning_rate: float = 0.1
    max_adjustment_factor: float = 2.0   # Max LR increase per penalty
    min_adjustment_factor: float = 0.5   # Max LR decrease per reward
    extreme_threshold: float = 0.8       # Depression/excitement threshold
    window_size: int = 10                # Window for trend analysis


class DynamicNetwork:
    """
    Dynamic Neural Network with automatic architecture adaptation.

    The network automatically adjusts its structure during training:
    - Adds layers/nodes when capacity is saturated
    - Removes layers/nodes when they're inefficient
    - Monitors for pathological growth (cancer) or pruning (alzheimer)
    - Gradually reduces trainable nodes from 100% to 1%
    - Uses adaptive batch sizing (small -> large)

    Only required hyperparameters:
    - seed: Random seed for reproducibility
    - input_shape: Shape of input data
    - output_size: Number of output neurons
    - cost_function: Loss function to use

    Example:
        >>> network = DynamicNetwork(
        ...     input_shape=(784,),
        ...     output_size=10,
        ...     seed=42,
        ...     cost_function="CrossEntropy"
        ... )
        >>> result = network.fit(X_train, y_train)
        >>> predictions = network.predict(X_test)
    """

    # Available cost functions
    COST_FUNCTIONS = {
        "MSE": "Mean Squared Error",
        "MAE": "Mean Absolute Error",
        "CrossEntropy": "Cross-Entropy (multi-class)",
        "BinaryCrossEntropy": "Binary Cross-Entropy",
        "Huber": "Huber Loss",
        "LogCosh": "Log-Cosh Loss",
        "KLDivergence": "KL Divergence",
        "CosineSimilarity": "Cosine Similarity"
    }

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 output_size: int,
                 seed: int,
                 cost_function: str = "MSE"):
        """
        Initialize a Dynamic Neural Network.

        Args:
            input_shape: Shape of input data (excluding batch dimension)
            output_size: Number of output neurons
            seed: Random seed - the ONLY required hyperparameter
            cost_function: One of the available cost functions
        """
        if cost_function not in self.COST_FUNCTIONS:
            raise ValueError(f"Unknown cost function: {cost_function}. "
                           f"Available: {list(self.COST_FUNCTIONS.keys())}")

        self.input_shape = input_shape
        self.output_size = output_size
        self.seed = seed
        self.cost_function = cost_function

        # Internal state
        self._trained = False
        self._training_result: Optional[TrainingResult] = None
        self._layers: List[Dict] = []

        # Try to use C++ backend
        try:
            from . import _dnn_core
            self._use_cpp = True
            self._init_cpp_network()
        except ImportError:
            self._use_cpp = False
            self._init_python_network()

    def _init_cpp_network(self):
        """Initialize using C++ backend."""
        from . import _dnn_core

        config = _dnn_core.NetworkConfig()
        config.seed = self.seed
        config.input_shape = list(self.input_shape)
        config.output_size = self.output_size
        config.cost_function = getattr(_dnn_core.CostFunction, self.cost_function)

        self._network = _dnn_core.Network(config)

    def _init_python_network(self):
        """Initialize using pure Python fallback."""
        # Pure Python implementation for when C++ is not available
        self._network = None
        print("Note: Using pure Python fallback. For best performance, build C++ extensions.")

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            callback: Optional[Callable[[int, float, float], None]] = None,
            verbose: bool = True) -> TrainingResult:
        """
        Train the network with automatic parameter adjustment.

        All training parameters are determined automatically from the seed:
        - Learning rate and schedule
        - Batch size progression
        - Early stopping criteria
        - Layer/node adjustments

        Args:
            X: Input data array of shape (n_samples, *input_shape)
            y: Target data array of shape (n_samples, output_size)
            callback: Optional callback(epoch, cost, efficiency)
            verbose: Print training progress

        Returns:
            TrainingResult with training history
        """
        if verbose:
            print(f"Training Dynamic Neural Network")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output size: {self.output_size}")
            print(f"  Seed: {self.seed}")
            print(f"  Cost function: {self.cost_function}")
            print(f"  Samples: {len(X)}")
            print()

        # Validate input shapes
        expected_input = (len(X),) + self.input_shape
        if X.shape != expected_input:
            # Try to reshape if total size matches
            if np.prod(X.shape[1:]) == np.prod(self.input_shape):
                X = X.reshape(expected_input)
            else:
                raise ValueError(f"Expected input shape {expected_input}, got {X.shape}")

        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        if self._use_cpp:
            result = self._fit_cpp(X, y, callback, verbose)
        else:
            result = self._fit_python(X, y, callback, verbose)

        self._trained = True
        self._training_result = result

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Epochs: {result.epochs_completed}")
            print(f"  Final cost: {result.final_cost:.6f}")
            print(f"  Final efficiency: {result.final_efficiency:.4f}")
            print(f"  Stopping reason: {result.stopping_reason}")

        return result

    def _fit_cpp(self, X, y, callback, verbose) -> TrainingResult:
        """Train using C++ backend."""
        from . import _dnn_core

        # Convert to C++ tensors
        inputs = [_dnn_core.Tensor(x) for x in X]
        targets = [_dnn_core.Tensor(t) for t in y]

        # Create trainer
        trainer = _dnn_core.Trainer(self._network)

        if callback:
            trainer.set_epoch_callback(callback)

        # Train
        cpp_result = trainer.train(inputs, targets)

        return TrainingResult(
            success=cpp_result.success,
            epochs_completed=cpp_result.epochs_completed,
            final_cost=cpp_result.final_cost,
            final_efficiency=cpp_result.final_efficiency,
            best_cost=cpp_result.best_cost,
            best_efficiency=cpp_result.best_efficiency,
            stopping_reason=cpp_result.stopping_reason,
            cost_history=list(cpp_result.cost_history),
            efficiency_history=list(cpp_result.efficiency_history),
            training_time_ms=cpp_result.training_time.count()
        )

    def _fit_python(self, X, y, callback, verbose) -> TrainingResult:
        """Train using pure Python fallback with 3-phase training."""
        import time
        start_time = time.time()

        np.random.seed(self.seed)

        # Initialize network architecture
        self._init_architecture(X.shape[1])

        # Tracking variables
        cost_history = []
        efficiency_history = []
        cancer_score_history = []
        alzheimer_score_history = []
        architecture_history = []
        nodes_added = 0
        nodes_removed = 0
        layers_added = 0
        layers_removed = 0
        perturbations_applied = 0
        best_cost = float('inf')
        best_architecture = None

        # Emotional state tracking (for Phase 3 reward/penalty system)
        emotional_state = EmotionalState()
        rp_config = RewardPenaltyConfig()
        learning_rate_history = []
        emotional_state_actions = []

        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # ============ PHASE 1: EXPLORATION (10 epochs) ============
        phase1_start = time.time()
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 1: EXPLORATION (10 epochs)")
            print("=" * 60)

        exploration_costs = []
        learning_rate = 0.5  # High LR for wide exploration
        batch_size = 64

        phase1_iter = range(10)
        if verbose and TQDM_AVAILABLE:
            phase1_iter = tqdm(phase1_iter, desc="Phase 1: Exploration", unit="epoch")

        for epoch in phase1_iter:
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            exploration_costs.append(epoch_cost)
            cost_history.append(epoch_cost)

            # Aggressive architecture exploration
            added, removed, layer_change = self._aggressive_layer_adjustment(
                saturation_threshold=0.3,
                efficiency_threshold=0.3
            )
            nodes_added += added
            nodes_removed += removed
            if layer_change > 0:
                layers_added += layer_change
            elif layer_change < 0:
                layers_removed += abs(layer_change)

            # Track best architecture
            if epoch_cost < best_cost:
                best_cost = epoch_cost
                best_architecture = self._save_architecture()

            # Compute metrics
            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, epoch + 1)
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            if verbose and TQDM_AVAILABLE:
                phase1_iter.set_postfix({
                    'cost': f'{epoch_cost:.4f}',
                    'layers': len(self._layers),
                    'nodes': self._count_nodes(),
                    'lr': f'{learning_rate:.4f}'
                })
            elif verbose and epoch % 2 == 0:
                print(f"  Epoch {epoch}: cost={epoch_cost:.4f}, layers={len(self._layers)}, nodes={self._count_nodes()}")

            if callback:
                callback(epoch, epoch_cost, efficiency)

        phase1_time = time.time() - phase1_start

        # ============ PHASE 2: ESTIMATION (10 epochs) ============
        phase2_start = time.time()
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: ESTIMATION (10 epochs)")
            print("=" * 60)

        estimation_costs = []
        learning_rate = 0.1  # Medium LR

        phase2_iter = range(10)
        if verbose and TQDM_AVAILABLE:
            phase2_iter = tqdm(phase2_iter, desc="Phase 2: Estimation", unit="epoch")

        for epoch in phase2_iter:
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            estimation_costs.append(epoch_cost)
            cost_history.append(epoch_cost)

            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, len(cost_history))
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            if callback:
                callback(10 + epoch, epoch_cost, efficiency)

        # Estimate epochs needed to reach 90% efficiency
        avg_improvement = (estimation_costs[0] - estimation_costs[-1]) / 10 if estimation_costs else 0.001
        current_efficiency = efficiency_history[-1] if efficiency_history else 0.5
        efficiency_gap = 0.9 - current_efficiency
        estimated_epochs = int(efficiency_gap / (avg_improvement * 0.1 + 1e-8))
        estimated_epochs = max(10, min(500, estimated_epochs))

        phase2_time = time.time() - phase2_start
        if verbose:
            print(f"  Estimated epochs for 90% efficiency: {estimated_epochs}")

        # ============ PHASE 3: MAIN TRAINING ============
        phase3_start = time.time()
        if verbose:
            print("\n" + "=" * 60)
            print(f"PHASE 3: MAIN TRAINING ({estimated_epochs} epochs)")
            print("=" * 60)

        perturbation_cutoff = int(estimated_epochs * 0.2)  # First 20%
        learning_rate = 0.1
        batch_size = 32

        phase3_iter = range(estimated_epochs)
        if verbose and TQDM_AVAILABLE:
            phase3_iter = tqdm(phase3_iter, desc="Phase 3: Training", unit="epoch")

        for epoch in phase3_iter:
            total_epoch = 20 + epoch  # Account for phases 1 and 2

            # Apply random perturbation in first 20% to avoid overfitting
            if epoch < perturbation_cutoff:
                self._apply_random_perturbation(fraction=0.005)
                perturbations_applied += 1

            # Compute adaptive saturation threshold
            efficiency = self._compute_efficiency(cost_history)
            saturation_threshold = self._sigmoid_threshold(efficiency)

            # Train one epoch
            epoch_cost = self._train_epoch(X, y, learning_rate, batch_size)
            cost_history.append(epoch_cost)

            # Track best cost
            if epoch_cost < best_cost:
                best_cost = epoch_cost

            # Layer adjustment with adaptive threshold
            added, removed, layer_change = self._layer_adjustment(
                saturation_threshold=saturation_threshold,
                efficiency_threshold=0.5
            )
            nodes_added += added
            nodes_removed += removed
            if layer_change > 0:
                layers_added += layer_change
            elif layer_change < 0:
                layers_removed += abs(layer_change)

            # Compute metrics
            efficiency = self._compute_efficiency(cost_history)
            efficiency_history.append(efficiency)
            cancer, alzheimer = self._compute_health_scores(nodes_added, nodes_removed, layers_added, layers_removed, len(cost_history))
            cancer_score_history.append(cancer)
            alzheimer_score_history.append(alzheimer)
            architecture_history.append((len(self._layers), self._count_nodes()))

            # Apply reward/penalty system (replaces fixed LR decay)
            learning_rate, action = self._apply_reward_penalty_system(
                learning_rate, cost_history, efficiency_history, rp_config, emotional_state
            )
            learning_rate_history.append(learning_rate)
            emotional_state_actions.append(action)

            # Adaptive batch size growth
            if epoch > 0 and epoch % 25 == 0:
                batch_size = min(batch_size * 2, 256)

            # Update progress bar
            if verbose and TQDM_AVAILABLE:
                phase3_iter.set_postfix({
                    'cost': f'{epoch_cost:.4f}',
                    'eff': f'{efficiency:.2%}',
                    'lr': f'{learning_rate:.4f}',
                    'dep': f'{emotional_state.depression_ratio:.0%}',
                    'exc': f'{emotional_state.excitement_ratio:.0%}',
                    'R/P': f'{emotional_state.total_rewards}/{emotional_state.total_penalties}'
                })
            elif verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: cost={epoch_cost:.4f}, eff={efficiency:.2%}, lr={learning_rate:.4f}, R/P={emotional_state.total_rewards}/{emotional_state.total_penalties}")

            if callback:
                callback(total_epoch, epoch_cost, efficiency)

            # Early stopping check
            if len(cost_history) > 20:
                recent_improvement = cost_history[-20] - cost_history[-1]
                if recent_improvement < 1e-6:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (no improvement)")
                    break

        phase3_time = time.time() - phase3_start

        # ============ TRAINING SUMMARY ============
        training_time_ms = int((time.time() - start_time) * 1000)

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"  Total Time: {training_time_ms / 1000:.2f}s")
            print(f"  Phases: Exploration({phase1_time:.1f}s) → Estimation({phase2_time:.1f}s) → Training({phase3_time:.1f}s)")
            initial_arch = architecture_history[0] if architecture_history else (2, 0)
            final_arch = architecture_history[-1] if architecture_history else (len(self._layers), self._count_nodes())
            print(f"  Architecture: {initial_arch[0]} layers → {final_arch[0]} layers, {initial_arch[1]} nodes → {final_arch[1]} nodes")
            print(f"  Nodes: +{nodes_added} added, -{nodes_removed} removed")
            print(f"  Layers: +{layers_added} added, -{layers_removed} removed")
            print(f"  Perturbations: {perturbations_applied}")
            print(f"  Health: Cancer={cancer_score_history[-1]:.1%}, Alzheimer={alzheimer_score_history[-1]:.1%}")
            print(f"  Emotional: {emotional_state.total_rewards} rewards, {emotional_state.total_penalties} penalties")
            print(f"  Depression ratio: {emotional_state.depression_ratio:.1%}, Excitement ratio: {emotional_state.excitement_ratio:.1%}")
            print(f"  LR resets: {emotional_state.lr_reset_count}")
            print("=" * 60)

        return TrainingResult(
            success=True,
            epochs_completed=len(cost_history),
            final_cost=cost_history[-1] if cost_history else 0.0,
            final_efficiency=efficiency_history[-1] if efficiency_history else 0.5,
            best_cost=best_cost,
            best_efficiency=max(efficiency_history) if efficiency_history else 0.5,
            stopping_reason="Training completed",
            cost_history=cost_history,
            efficiency_history=efficiency_history,
            training_time_ms=training_time_ms,
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            layers_added=layers_added,
            layers_removed=layers_removed,
            cancer_score_history=cancer_score_history,
            alzheimer_score_history=alzheimer_score_history,
            architecture_history=architecture_history,
            perturbations_applied=perturbations_applied,
            phase_metrics={
                'phase1_time': phase1_time,
                'phase2_time': phase2_time,
                'phase3_time': phase3_time,
                'estimated_epochs': estimated_epochs,
                'exploration_best_cost': min(exploration_costs) if exploration_costs else 0.0,
                'final_depression_ratio': emotional_state.depression_ratio,
                'final_excitement_ratio': emotional_state.excitement_ratio
            },
            # Emotional state fields
            total_rewards=emotional_state.total_rewards,
            total_penalties=emotional_state.total_penalties,
            depression_history=emotional_state.depression_history,
            excitement_history=emotional_state.excitement_history,
            lr_reset_count=emotional_state.lr_reset_count,
            learning_rate_history=learning_rate_history,
            emotional_state_history=emotional_state_actions
        )

    def _init_architecture(self, input_size: int) -> None:
        """Initialize network architecture."""
        # Compute initial hidden layer size (geometric mean)
        hidden_size = int(np.sqrt(input_size * self.output_size))
        hidden_size = max(hidden_size, 16)  # Minimum hidden size

        # Initialize weights with He initialization
        W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        W2 = np.random.randn(self.output_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros(self.output_size, dtype=np.float32)

        # Store layers
        self._layers = [
            {"W": W1, "b": b1, "efficiency": np.ones(hidden_size) * 0.5},
            {"W": W2, "b": b2, "efficiency": np.ones(self.output_size) * 0.5}
        ]

    def _count_nodes(self) -> int:
        """Count total nodes in network."""
        return sum(layer["W"].shape[0] for layer in self._layers)

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, learning_rate: float, batch_size: int) -> float:
        """Train for one epoch and return average cost."""
        epoch_cost = 0.0
        n_batches = max(1, len(X) // batch_size)
        perm = np.random.permutation(len(X))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            batch_indices = perm[start_idx:end_idx]

            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            m = len(batch_X)

            # Forward pass through all layers
            activations = [batch_X.reshape(m, -1)]
            z_values = []

            for i, layer in enumerate(self._layers):
                z = activations[-1] @ layer["W"].T + layer["b"]
                z_values.append(z)

                if i < len(self._layers) - 1:
                    # ReLU for hidden layers
                    a = np.maximum(0, z)
                else:
                    # Softmax for output layer
                    z_max = np.max(z, axis=1, keepdims=True)
                    exp_z = np.exp(z - z_max)
                    a = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
                activations.append(a)

            # Compute cost
            log_probs = np.log(activations[-1] + 1e-8)
            batch_cost = -np.mean(np.sum(batch_y * log_probs, axis=1))
            epoch_cost += batch_cost

            # Backward pass
            dz = (activations[-1] - batch_y) / m

            for i in range(len(self._layers) - 1, -1, -1):
                dW = dz.T @ activations[i]
                db = np.sum(dz, axis=0)

                # Gradient clipping
                clip_value = 5.0
                dW = np.clip(dW, -clip_value, clip_value)
                db = np.clip(db, -clip_value, clip_value)

                # Update weights
                self._layers[i]["W"] -= learning_rate * dW
                self._layers[i]["b"] -= learning_rate * db

                # Update efficiency based on gradient magnitude
                grad_magnitude = np.mean(np.abs(dW), axis=1)
                self._layers[i]["efficiency"] = 0.9 * self._layers[i]["efficiency"] + 0.1 * np.clip(grad_magnitude * 10, 0, 1)

                if i > 0:
                    da = dz @ self._layers[i]["W"]
                    dz = da * (z_values[i - 1] > 0).astype(np.float32)

        return epoch_cost / n_batches

    def _compute_efficiency(self, cost_history: List[float]) -> float:
        """Compute efficiency metric based on cost reduction."""
        if len(cost_history) < 2:
            return 0.5
        improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)
        return min(1.0, max(0.0, 0.5 + improvement * 10))

    def _compute_health_scores(self, nodes_added: int, nodes_removed: int,
                               layers_added: int, layers_removed: int, epoch: int) -> Tuple[float, float]:
        """Compute cancer and alzheimer scores."""
        # Cancer score: excessive growth
        growth_rate = (nodes_added + layers_added * 10) / (epoch + 1)
        cancer = min(1.0, growth_rate / 5.0)

        # Alzheimer score: excessive removal
        removal_rate = (nodes_removed + layers_removed * 10) / (epoch + 1)
        alzheimer = min(1.0, removal_rate / 5.0)

        return cancer, alzheimer

    def _sigmoid_threshold(self, efficiency: float, k: float = 5.0,
                          base: float = 0.3, range_val: float = 0.5) -> float:
        """Compute adaptive saturation threshold using sigmoid function."""
        sigmoid = 1.0 / (1.0 + np.exp(-k * (efficiency - 0.5)))
        return base + range_val * sigmoid

    # ============ REWARD/PENALTY SYSTEM METHODS ============

    def _compute_improvement_metrics(
        self,
        cost_history: List[float],
        efficiency_history: List[float],
        window: int = 5
    ) -> Dict[str, float]:
        """
        Compute improvement metrics over a sliding window.

        Returns:
            Dict with keys:
            - cost_improvement: (old - new) / old (positive = improvement)
            - efficiency_improvement: new - old (positive = improvement)
            - cost_trend: average rate of change (negative = decreasing = good)
            - efficiency_trend: average rate of change (positive = increasing = good)
        """
        if len(cost_history) < 2:
            return {
                "cost_improvement": 0.0,
                "efficiency_improvement": 0.0,
                "cost_trend": 0.0,
                "efficiency_trend": 0.0
            }

        # Use most recent window for trend analysis
        recent_costs = cost_history[-min(window, len(cost_history)):]
        recent_efficiency = efficiency_history[-min(window, len(efficiency_history)):]

        # Cost improvement (epoch-over-epoch)
        cost_improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)

        # Efficiency improvement (epoch-over-epoch)
        efficiency_improvement = efficiency_history[-1] - efficiency_history[-2] if len(efficiency_history) >= 2 else 0.0

        # Trend analysis over window
        if len(recent_costs) >= 2:
            cost_trend = (recent_costs[-1] - recent_costs[0]) / (len(recent_costs) * (recent_costs[0] + 1e-8))
        else:
            cost_trend = 0.0

        if len(recent_efficiency) >= 2:
            efficiency_trend = (recent_efficiency[-1] - recent_efficiency[0]) / len(recent_efficiency)
        else:
            efficiency_trend = 0.0

        return {
            "cost_improvement": cost_improvement,
            "efficiency_improvement": efficiency_improvement,
            "cost_trend": cost_trend,
            "efficiency_trend": efficiency_trend
        }

    def _should_reward(
        self,
        metrics: Dict[str, float],
        config: RewardPenaltyConfig
    ) -> Tuple[bool, float]:
        """
        Determine if current epoch deserves a reward.

        Returns:
            Tuple of (should_reward: bool, reward_magnitude: float)
        """
        # Reward conditions:
        # 1. Cost is decreasing
        # 2. Efficiency is good/improving
        # 3. Trend is positive overall
        cost_improving = metrics["cost_improvement"] > config.cost_improvement_threshold
        efficiency_good = metrics["efficiency_improvement"] >= 0 or metrics["efficiency_trend"] > 0
        trend_positive = metrics["cost_trend"] < 0  # Negative trend means cost decreasing

        should_reward = cost_improving and efficiency_good and trend_positive

        if should_reward:
            # Magnitude proportional to improvement
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)  # Cap at 1.0
        else:
            magnitude = 0.0

        return should_reward, magnitude

    def _should_penalize(
        self,
        metrics: Dict[str, float],
        config: RewardPenaltyConfig
    ) -> Tuple[bool, float]:
        """
        Determine if current epoch deserves a penalty.

        Returns:
            Tuple of (should_penalize: bool, penalty_magnitude: float)
        """
        # Penalty conditions:
        # 1. Cost is increasing
        # 2. Efficiency is declining
        # 3. Trend is negative overall
        cost_degrading = metrics["cost_improvement"] < -config.cost_improvement_threshold
        efficiency_bad = metrics["efficiency_improvement"] < -config.efficiency_improvement_threshold
        trend_negative = metrics["cost_trend"] > 0  # Positive trend means cost increasing

        should_penalize = cost_degrading or (efficiency_bad and trend_negative)

        if should_penalize:
            # Magnitude proportional to degradation
            magnitude = abs(metrics["cost_improvement"]) + abs(metrics["efficiency_improvement"]) * 0.5
            magnitude = min(magnitude, 1.0)  # Cap at 1.0
        else:
            magnitude = 0.0

        return should_penalize, magnitude

    def _apply_reward(
        self,
        learning_rate: float,
        magnitude: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> float:
        """
        Apply reward by decreasing learning rate proportionally.

        The intuition: good progress means we're on the right track,
        so we can take smaller steps to fine-tune.

        Returns:
            Adjusted learning rate
        """
        # Decrease factor proportional to magnitude
        # magnitude of 1.0 -> multiply by min_adjustment_factor (0.5)
        # magnitude of 0.0 -> no change
        decrease_factor = 1.0 - magnitude * (1.0 - config.min_adjustment_factor)

        new_lr = learning_rate * decrease_factor
        new_lr = max(new_lr, config.min_learning_rate)

        # Update emotional state
        emotional_state.total_rewards += 1
        emotional_state.reward_history.append(magnitude)

        return new_lr

    def _apply_penalty(
        self,
        learning_rate: float,
        magnitude: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> float:
        """
        Apply penalty by increasing learning rate proportionally.

        The intuition: poor progress means we might be stuck,
        so we need larger steps to escape.

        Returns:
            Adjusted learning rate
        """
        # Increase factor proportional to magnitude
        # magnitude of 1.0 -> multiply by max_adjustment_factor (2.0)
        # magnitude of 0.0 -> no change
        increase_factor = 1.0 + magnitude * (config.max_adjustment_factor - 1.0)

        new_lr = learning_rate * increase_factor
        new_lr = min(new_lr, config.max_learning_rate)

        # Update emotional state
        emotional_state.total_penalties += 1
        emotional_state.penalty_history.append(magnitude)

        return new_lr

    def _check_extreme_states(
        self,
        learning_rate: float,
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> Tuple[float, str]:
        """
        Check for extreme emotional states and reset LR if detected.

        Returns:
            Tuple of (adjusted_lr, state_description)
        """
        depression = emotional_state.depression_ratio
        excitement = emotional_state.excitement_ratio

        state = "neutral"

        # Check for extreme depression (too many penalties)
        if depression > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_depression"

        # Check for extreme excitement (too many rewards)
        elif excitement > config.extreme_threshold:
            learning_rate = config.baseline_learning_rate
            emotional_state.lr_reset_count += 1
            state = "extreme_excitement"

        elif depression > 0.5:
            state = "depressed"
        elif excitement > 0.5:
            state = "excited"

        # Record history
        emotional_state.depression_history.append(depression)
        emotional_state.excitement_history.append(excitement)

        return learning_rate, state

    def _apply_reward_penalty_system(
        self,
        learning_rate: float,
        cost_history: List[float],
        efficiency_history: List[float],
        config: RewardPenaltyConfig,
        emotional_state: EmotionalState
    ) -> Tuple[float, str]:
        """
        Apply the complete reward/penalty system for one epoch.

        Returns:
            Tuple of (new_learning_rate, action_taken)
            action_taken: "reward", "penalty", "neutral", "extreme_depression_reset", "extreme_excitement_reset"
        """
        # Compute metrics
        metrics = self._compute_improvement_metrics(cost_history, efficiency_history, config.window_size)

        action = "neutral"

        # Check for reward
        should_reward, reward_mag = self._should_reward(metrics, config)
        if should_reward:
            learning_rate = self._apply_reward(learning_rate, reward_mag, config, emotional_state)
            action = "reward"
        else:
            # Check for penalty
            should_penalize, penalty_mag = self._should_penalize(metrics, config)
            if should_penalize:
                learning_rate = self._apply_penalty(learning_rate, penalty_mag, config, emotional_state)
                action = "penalty"

        # Check for extreme states (may override previous adjustment)
        learning_rate, extreme_state = self._check_extreme_states(learning_rate, config, emotional_state)
        if extreme_state.startswith("extreme"):
            action = f"{extreme_state}_reset"

        return learning_rate, action

    # ============ END REWARD/PENALTY SYSTEM METHODS ============

    def _apply_random_perturbation(self, fraction: float = 0.005) -> None:
        """Apply random perturbation to fraction of nodes to avoid overfitting."""
        total_nodes = self._count_nodes()
        num_to_perturb = max(1, int(total_nodes * fraction))

        for _ in range(num_to_perturb):
            layer_idx = np.random.randint(len(self._layers))
            node_idx = np.random.randint(self._layers[layer_idx]["W"].shape[0])

            # Add small random perturbation (1% of weight magnitude)
            perturbation_scale = 0.01
            self._layers[layer_idx]["W"][node_idx] *= (1 + np.random.randn() * perturbation_scale)
            self._layers[layer_idx]["b"][node_idx] *= (1 + np.random.randn() * perturbation_scale)

    def _aggressive_layer_adjustment(self, saturation_threshold: float = 0.3,
                                     efficiency_threshold: float = 0.3) -> Tuple[int, int, int]:
        """Aggressive layer/node adjustment for exploration phase."""
        nodes_added = 0
        nodes_removed = 0
        layer_change = 0

        for i in range(len(self._layers) - 1):  # Don't modify output layer
            layer = self._layers[i]
            avg_efficiency = np.mean(layer["efficiency"])

            # Add nodes if layer is saturated
            if avg_efficiency > saturation_threshold and layer["W"].shape[0] < 2000:
                num_new = max(1, layer["W"].shape[0] // 4)
                self._add_nodes_to_layer(i, num_new)
                nodes_added += num_new

            # Remove inefficient nodes
            inefficient_mask = layer["efficiency"] < efficiency_threshold * 0.5
            if np.sum(inefficient_mask) > 0 and layer["W"].shape[0] > 8:
                num_to_remove = min(np.sum(inefficient_mask), layer["W"].shape[0] - 8)
                if num_to_remove > 0:
                    self._remove_nodes_from_layer(i, int(num_to_remove))
                    nodes_removed += int(num_to_remove)

        # Consider adding a layer if all hidden layers are saturated
        if len(self._layers) < 10:
            all_saturated = all(
                np.mean(self._layers[i]["efficiency"]) > saturation_threshold
                for i in range(len(self._layers) - 1)
            )
            if all_saturated:
                self._add_layer()
                layer_change = 1

        return nodes_added, nodes_removed, layer_change

    def _layer_adjustment(self, saturation_threshold: float,
                         efficiency_threshold: float) -> Tuple[int, int, int]:
        """Standard layer/node adjustment for main training phase."""
        nodes_added = 0
        nodes_removed = 0
        layer_change = 0

        for i in range(len(self._layers) - 1):
            layer = self._layers[i]
            avg_efficiency = np.mean(layer["efficiency"])

            # Add nodes if layer is saturated
            if avg_efficiency > saturation_threshold and layer["W"].shape[0] < 2000:
                num_new = max(1, layer["W"].shape[0] // 8)  # 12.5% growth (less aggressive)
                self._add_nodes_to_layer(i, num_new)
                nodes_added += num_new

        return nodes_added, nodes_removed, layer_change

    def _add_nodes_to_layer(self, layer_idx: int, num_nodes: int) -> None:
        """Add nodes to a layer."""
        layer = self._layers[layer_idx]
        input_size = layer["W"].shape[1]
        output_size = layer["W"].shape[0]

        # Initialize new weights
        new_W = np.random.randn(num_nodes, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        new_b = np.zeros(num_nodes, dtype=np.float32)
        new_eff = np.ones(num_nodes) * 0.5

        # Append to layer
        layer["W"] = np.vstack([layer["W"], new_W])
        layer["b"] = np.concatenate([layer["b"], new_b])
        layer["efficiency"] = np.concatenate([layer["efficiency"], new_eff])

        # Update next layer's input size
        if layer_idx < len(self._layers) - 1:
            next_layer = self._layers[layer_idx + 1]
            new_cols = np.random.randn(next_layer["W"].shape[0], num_nodes).astype(np.float32) * np.sqrt(2.0 / (output_size + num_nodes))
            next_layer["W"] = np.hstack([next_layer["W"], new_cols])

    def _remove_nodes_from_layer(self, layer_idx: int, num_nodes: int) -> None:
        """Remove least efficient nodes from a layer."""
        layer = self._layers[layer_idx]

        # Find indices of least efficient nodes
        indices_to_remove = np.argsort(layer["efficiency"])[:num_nodes]
        indices_to_keep = np.setdiff1d(np.arange(layer["W"].shape[0]), indices_to_remove)

        if len(indices_to_keep) < 4:
            return  # Keep minimum nodes

        # Remove from current layer
        layer["W"] = layer["W"][indices_to_keep]
        layer["b"] = layer["b"][indices_to_keep]
        layer["efficiency"] = layer["efficiency"][indices_to_keep]

        # Update next layer's input
        if layer_idx < len(self._layers) - 1:
            next_layer = self._layers[layer_idx + 1]
            next_layer["W"] = next_layer["W"][:, indices_to_keep]

    def _add_layer(self) -> None:
        """Add a new hidden layer."""
        if len(self._layers) >= 10:
            return

        # Insert before output layer
        insert_idx = len(self._layers) - 1
        prev_layer = self._layers[insert_idx - 1] if insert_idx > 0 else self._layers[0]
        next_layer = self._layers[insert_idx]

        # New layer size is geometric mean
        prev_size = prev_layer["W"].shape[0]
        next_input = next_layer["W"].shape[1]
        new_size = int(np.sqrt(prev_size * next_input))
        new_size = max(16, new_size)

        # Create new layer
        new_W = np.random.randn(new_size, prev_size).astype(np.float32) * np.sqrt(2.0 / prev_size)
        new_b = np.zeros(new_size, dtype=np.float32)
        new_eff = np.ones(new_size) * 0.5

        new_layer = {"W": new_W, "b": new_b, "efficiency": new_eff}

        # Update next layer's input
        next_layer["W"] = np.random.randn(next_layer["W"].shape[0], new_size).astype(np.float32) * np.sqrt(2.0 / new_size)

        # Insert new layer
        self._layers.insert(insert_idx, new_layer)

    def _save_architecture(self) -> Dict:
        """Save current architecture for restoration."""
        return {
            'layers': [
                {
                    'W': layer['W'].copy(),
                    'b': layer['b'].copy(),
                    'efficiency': layer['efficiency'].copy()
                }
                for layer in self._layers
            ]
        }

    def _restore_architecture(self, saved: Dict) -> None:
        """Restore architecture from saved state."""
        self._layers = [
            {
                'W': layer['W'].copy(),
                'b': layer['b'].copy(),
                'efficiency': layer['efficiency'].copy()
            }
            for layer in saved['layers']
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data array

        Returns:
            Predictions array
        """
        X = X.astype(np.float32)

        if self._use_cpp:
            from . import _dnn_core
            outputs = []
            for x in X:
                tensor = _dnn_core.Tensor(x.flatten())
                output = self._network.predict(tensor)
                outputs.append(output.numpy())
            return np.array(outputs)
        else:
            # Python fallback - works with multi-layer dynamic architecture
            a = X.reshape(len(X), -1)
            for i, layer in enumerate(self._layers):
                z = a @ layer["W"].T + layer["b"]
                if i < len(self._layers) - 1:
                    # ReLU for hidden layers
                    a = np.maximum(0, z)
                else:
                    # Softmax for output layer
                    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                    a = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)
            return a

    def health_status(self) -> HealthReport:
        """Get health status including emotional state (cancer/alzheimer/depression/excitement)."""
        # Compute emotional metrics from last training
        depression = 0.0
        excitement = 0.0
        emotional_state = "neutral"

        if self._training_result:
            total = self._training_result.total_rewards + self._training_result.total_penalties
            if total > 0:
                depression = self._training_result.total_penalties / total
                excitement = self._training_result.total_rewards / total

            if depression > 0.8:
                emotional_state = "extreme_depression"
            elif excitement > 0.8:
                emotional_state = "extreme_excitement"
            elif depression > 0.5:
                emotional_state = "depressed"
            elif excitement > 0.5:
                emotional_state = "excited"

        if self._use_cpp:
            from . import _dnn_core
            report = self._network.health_report()
            return HealthReport(
                state=str(report.state),
                cancer_score=report.cancer_score,
                alzheimer_score=report.alzheimer_score,
                overall_health=report.overall_health,
                diagnosis=report.diagnosis,
                recommendations=list(report.recommendations),
                current_layers=report.current_layers,
                current_nodes=report.current_nodes,
                depression_ratio=depression,
                excitement_ratio=excitement,
                emotional_state=emotional_state
            )
        else:
            return HealthReport(
                state="Healthy",
                cancer_score=0.0,
                alzheimer_score=0.0,
                overall_health=1.0,
                diagnosis="Network is healthy",
                recommendations=[],
                current_layers=len(self._layers),
                current_nodes=sum(l["W"].shape[0] for l in self._layers),
                depression_ratio=depression,
                excitement_ratio=excitement,
                emotional_state=emotional_state
            )

    def efficiency_report(self, include_nodes: bool = False) -> Dict[str, Any]:
        """Get efficiency report for network."""
        if self._use_cpp:
            from . import _dnn_core
            return self._network.efficiency_report(include_nodes)
        else:
            return {
                "avg_layer_efficiency": 0.5,
                "total_layers": len(self._layers),
                "total_nodes": sum(l["W"].shape[0] for l in self._layers),
                "total_parameters": sum(l["W"].size + l["b"].size for l in self._layers)
            }

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        if self._use_cpp:
            return self._network.num_layers()
        return len(self._layers)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        if self._use_cpp:
            return self._network.num_parameters()
        return sum(l["W"].size + l["b"].size for l in self._layers)

    def save(self, path: str, name: str = "model"):
        """
        Save model to files.

        Creates:
        - {name}.txt: Model architecture and metadata
        - code.py: Python code to load and use the model
        - requirements.txt: Required packages

        Args:
            path: Directory to save to
            name: Model name
        """
        import os
        os.makedirs(path, exist_ok=True)

        if self._use_cpp:
            from . import _dnn_core
            _dnn_core.save_model(self._network, path, name)
        else:
            # Python fallback - save as numpy files
            import json

            metadata = {
                "version": "1.0",
                "seed": self.seed,
                "input_shape": self.input_shape,
                "output_size": self.output_size,
                "cost_function": self.cost_function,
                "num_layers": len(self._layers)
            }

            with open(os.path.join(path, f"{name}.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            for i, layer in enumerate(self._layers):
                np.save(os.path.join(path, f"{name}_layer{i}_W.npy"), layer["W"])
                np.save(os.path.join(path, f"{name}_layer{i}_b.npy"), layer["b"])

    @classmethod
    def load(cls, path: str) -> "DynamicNetwork":
        """
        Load a saved model.

        Args:
            path: Path to model file

        Returns:
            Loaded DynamicNetwork
        """
        import os
        import json

        # Try to load JSON metadata
        json_path = path if path.endswith(".json") else path.replace(".txt", ".json")

        if os.path.exists(json_path):
            with open(json_path) as f:
                metadata = json.load(f)

            network = cls(
                input_shape=tuple(metadata["input_shape"]),
                output_size=metadata["output_size"],
                seed=metadata["seed"],
                cost_function=metadata["cost_function"]
            )

            # Load weights
            base_path = os.path.dirname(json_path)
            name = os.path.basename(json_path).replace(".json", "")

            network._layers = []
            for i in range(metadata["num_layers"]):
                W = np.load(os.path.join(base_path, f"{name}_layer{i}_W.npy"))
                b = np.load(os.path.join(base_path, f"{name}_layer{i}_b.npy"))
                network._layers.append({"W": W, "b": b})

            network._trained = True
            return network

        raise FileNotFoundError(f"Model file not found: {path}")
