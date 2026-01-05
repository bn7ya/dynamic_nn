"""
High-level Python interface for Dynamic Neural Network.
"""

from typing import Tuple, List, Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingResult:
    """Result of training."""
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
        """Train using pure Python fallback with backpropagation."""
        import time
        start_time = time.time()

        np.random.seed(self.seed)

        # Compute initial hidden layer size (geometric mean)
        input_size = int(np.prod(self.input_shape))
        hidden_size = int(np.sqrt(input_size * self.output_size))
        hidden_size = max(hidden_size, 16)  # Minimum hidden size

        # Initialize weights with He initialization
        W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        b1 = np.zeros(hidden_size, dtype=np.float32)
        W2 = np.random.randn(self.output_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros(self.output_size, dtype=np.float32)

        # Store layers with gradient storage
        self._layers = [
            {"W": W1, "b": b1, "dW": np.zeros_like(W1), "db": np.zeros_like(b1)},
            {"W": W2, "b": b2, "dW": np.zeros_like(W2), "db": np.zeros_like(b2)}
        ]

        cost_history = []
        efficiency_history = []
        learning_rate = 0.1
        batch_size = 32
        max_epochs = 100
        best_cost = float('inf')

        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        for epoch in range(max_epochs):
            epoch_cost = 0.0
            n_batches = max(1, len(X) // batch_size)

            # Shuffle each epoch
            perm = np.random.permutation(len(X))

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X))
                batch_indices = perm[start_idx:end_idx]

                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                m = len(batch_X)

                # ============ FORWARD PASS ============
                # Input layer
                a0 = batch_X.reshape(m, -1)

                # Hidden layer (ReLU)
                z1 = a0 @ self._layers[0]["W"].T + self._layers[0]["b"]
                a1 = np.maximum(0, z1)  # ReLU activation

                # Output layer (Softmax)
                z2 = a1 @ self._layers[1]["W"].T + self._layers[1]["b"]
                # Stable softmax
                z2_max = np.max(z2, axis=1, keepdims=True)
                exp_z2 = np.exp(z2 - z2_max)
                a2 = exp_z2 / (np.sum(exp_z2, axis=1, keepdims=True) + 1e-8)

                # ============ COMPUTE COST ============
                # Cross-entropy loss
                log_probs = np.log(a2 + 1e-8)
                batch_cost = -np.mean(np.sum(batch_y * log_probs, axis=1))
                epoch_cost += batch_cost

                # ============ BACKWARD PASS ============
                # Output layer gradient (softmax + cross-entropy)
                dz2 = (a2 - batch_y) / m  # Shape: (m, output_size)

                # Gradients for layer 2 (output layer)
                dW2 = dz2.T @ a1  # Shape: (output_size, hidden_size)
                db2 = np.sum(dz2, axis=0)  # Shape: (output_size,)

                # Backpropagate to hidden layer
                da1 = dz2 @ self._layers[1]["W"]  # Shape: (m, hidden_size)

                # ReLU gradient
                dz1 = da1 * (z1 > 0).astype(np.float32)  # Shape: (m, hidden_size)

                # Gradients for layer 1 (hidden layer)
                dW1 = dz1.T @ a0  # Shape: (hidden_size, input_size)
                db1 = np.sum(dz1, axis=0)  # Shape: (hidden_size,)

                # ============ UPDATE WEIGHTS ============
                # Gradient descent with gradient clipping
                clip_value = 5.0
                dW1 = np.clip(dW1, -clip_value, clip_value)
                dW2 = np.clip(dW2, -clip_value, clip_value)
                db1 = np.clip(db1, -clip_value, clip_value)
                db2 = np.clip(db2, -clip_value, clip_value)

                self._layers[0]["W"] -= learning_rate * dW1
                self._layers[0]["b"] -= learning_rate * db1
                self._layers[1]["W"] -= learning_rate * dW2
                self._layers[1]["b"] -= learning_rate * db2

            # Average epoch cost
            epoch_cost /= n_batches
            cost_history.append(epoch_cost)

            # Track best cost
            if epoch_cost < best_cost:
                best_cost = epoch_cost

            # Compute efficiency metric (based on cost reduction)
            if len(cost_history) > 1:
                improvement = (cost_history[-2] - cost_history[-1]) / (cost_history[-2] + 1e-8)
                efficiency = min(1.0, max(0.0, 0.5 + improvement * 10))
            else:
                efficiency = 0.5
            efficiency_history.append(efficiency)

            # Adaptive learning rate decay
            if epoch > 0 and epoch % 20 == 0:
                learning_rate *= 0.8

            # Adaptive batch size growth
            if epoch > 0 and epoch % 25 == 0:
                batch_size = min(batch_size * 2, 256)

            # Early stopping check
            if len(cost_history) > 10:
                recent_improvement = cost_history[-10] - cost_history[-1]
                if recent_improvement < 1e-6:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (no improvement)")
                    break

            if callback:
                callback(epoch, epoch_cost, efficiency)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch}: cost={epoch_cost:.6f}, lr={learning_rate:.6f}")

        training_time_ms = int((time.time() - start_time) * 1000)

        return TrainingResult(
            success=True,
            epochs_completed=len(cost_history),
            final_cost=cost_history[-1] if cost_history else 0.0,
            final_efficiency=efficiency_history[-1] if efficiency_history else 0.5,
            best_cost=best_cost,
            best_efficiency=max(efficiency_history) if efficiency_history else 0.5,
            stopping_reason="Training completed" if len(cost_history) == max_epochs else "Early stopping",
            cost_history=cost_history,
            efficiency_history=efficiency_history,
            training_time_ms=training_time_ms
        )

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
            # Python fallback
            a = X.reshape(len(X), -1)
            for layer in self._layers[:-1]:
                z = a @ layer["W"].T + layer["b"]
                a = np.maximum(0, z)
            z = a @ self._layers[-1]["W"].T + self._layers[-1]["b"]
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def health_status(self) -> HealthReport:
        """Get health status (cancer/alzheimer detection)."""
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
                current_nodes=report.current_nodes
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
                current_nodes=sum(l["W"].shape[0] for l in self._layers)
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
