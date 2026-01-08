"""
Standard Neural Network for Emotional Classification

A fixed-architecture feedforward neural network as baseline for comparison
with the Dynamic Neural Network.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time


@dataclass
class StandardNNConfig:
    """Configuration for Standard Neural Network."""
    input_size: int = 1000
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_size: int = 7  # 7 emotions
    dropout: float = 0.2
    activation: str = 'relu'  # 'relu', 'tanh', 'sigmoid'
    seed: int = 42


@dataclass
class StandardTrainingResult:
    """Training result for Standard NN."""
    success: bool
    epochs_completed: int
    final_cost: float
    final_accuracy: float
    best_cost: float
    best_accuracy: float
    cost_history: List[float]
    accuracy_history: List[float]
    val_cost_history: List[float]
    val_accuracy_history: List[float]
    training_time: float
    learning_rate_history: List[float]


class StandardNeuralNetwork:
    """
    Standard feedforward neural network with fixed architecture.

    Features:
    - Fixed layer structure
    - Standard backpropagation
    - Dropout regularization
    - Learning rate decay (fixed schedule)
    """

    def __init__(self, config: StandardNNConfig):
        """Initialize network with config."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize layers
        self._layers: List[Dict[str, np.ndarray]] = []
        self._init_weights()

        self._trained = False
        self.training_history: Optional[StandardTrainingResult] = None

    def _init_weights(self) -> None:
        """Initialize network weights using He initialization."""
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]

        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]

            # He initialization for ReLU
            std = np.sqrt(2.0 / fan_in)
            W = self.rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)

            self._layers.append({'W': W, 'b': b})

    def _activation(self, z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == 'relu':
            if derivative:
                return (z > 0).astype(np.float32)
            return np.maximum(0, z)
        elif self.config.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(z) ** 2
            return np.tanh(z)
        elif self.config.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            if derivative:
                return sig * (1 - sig)
            return sig
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-8)

    def _dropout_mask(self, shape: Tuple, training: bool) -> np.ndarray:
        """Generate dropout mask."""
        if training and self.config.dropout > 0:
            mask = (self.rng.random(shape) > self.config.dropout).astype(np.float32)
            mask /= (1 - self.config.dropout)  # Inverted dropout
            return mask
        return np.ones(shape, dtype=np.float32)

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Forward pass through network.

        Returns:
            Tuple of (output, cache) where cache contains intermediate values for backprop
        """
        cache = []
        a = X.astype(np.float32)

        for i, layer in enumerate(self._layers[:-1]):
            z = a @ layer['W'] + layer['b']
            a_pre = a.copy()

            a = self._activation(z)

            # Apply dropout
            mask = self._dropout_mask(a.shape, training)
            a = a * mask

            cache.append({
                'a_prev': a_pre,
                'z': z,
                'a': a,
                'mask': mask
            })

        # Output layer (no dropout)
        z_out = a @ self._layers[-1]['W'] + self._layers[-1]['b']
        output = self._softmax(z_out)

        cache.append({
            'a_prev': a,
            'z': z_out,
            'a': output,
            'mask': np.ones_like(output)
        })

        return output, cache

    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Convert to one-hot if needed
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_onehot

        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
        return loss

    def backward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        cache: List[Dict]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Backward pass to compute gradients.

        Returns:
            List of gradient dictionaries for each layer
        """
        batch_size = y_pred.shape[0]

        # Convert to one-hot
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_onehot

        gradients = []

        # Output layer gradient (softmax + cross-entropy)
        dz = (y_pred - y_true) / batch_size

        for i in range(len(self._layers) - 1, -1, -1):
            layer_cache = cache[i]
            a_prev = layer_cache['a_prev']

            dW = a_prev.T @ dz
            db = np.sum(dz, axis=0)

            gradients.insert(0, {'dW': dW, 'db': db})

            if i > 0:
                # Backprop to previous layer
                da = dz @ self._layers[i]['W'].T
                # Apply dropout mask
                da = da * cache[i-1]['mask']
                # Apply activation derivative
                dz = da * self._activation(cache[i-1]['z'], derivative=True)

        return gradients

    def _update_weights(
        self,
        gradients: List[Dict],
        learning_rate: float,
        momentum: float = 0.9,
        velocities: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Update weights using SGD with momentum."""
        if velocities is None:
            velocities = [
                {'vW': np.zeros_like(l['W']), 'vb': np.zeros_like(l['b'])}
                for l in self._layers
            ]

        for i, (layer, grad, vel) in enumerate(zip(self._layers, gradients, velocities)):
            # Update velocities
            vel['vW'] = momentum * vel['vW'] - learning_rate * grad['dW']
            vel['vb'] = momentum * vel['vb'] - learning_rate * grad['db']

            # Update weights
            layer['W'] += vel['vW']
            layer['b'] += vel['vb']

        return velocities

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        lr_decay: float = 0.95,
        lr_decay_epochs: int = 20,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> StandardTrainingResult:
        """
        Train the network.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            lr_decay: Learning rate decay factor
            lr_decay_epochs: Epochs between LR decay
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Print progress

        Returns:
            StandardTrainingResult with training metrics
        """
        start_time = time.time()

        cost_history = []
        accuracy_history = []
        val_cost_history = []
        val_accuracy_history = []
        learning_rate_history = []

        best_cost = float('inf')
        best_accuracy = 0.0
        best_weights = None
        patience_counter = 0

        velocities = None
        n_samples = len(X)

        if verbose:
            print("=" * 60)
            print("STANDARD NEURAL NETWORK TRAINING")
            print("=" * 60)
            print(f"Architecture: {self.config.input_size} -> {self.config.hidden_sizes} -> {self.config.output_size}")
            print(f"Parameters: {self.count_parameters():,}")
            print(f"Training samples: {n_samples}")
            print("=" * 60)

        for epoch in range(epochs):
            # Shuffle data
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []

            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred, cache = self.forward(X_batch, training=True)

                # Compute loss
                loss = self._cross_entropy_loss(y_pred, y_batch)
                epoch_losses.append(loss)

                # Backward pass
                gradients = self.backward(y_pred, y_batch, cache)

                # Update weights
                velocities = self._update_weights(gradients, learning_rate, velocities=velocities)

            # Epoch metrics
            epoch_cost = np.mean(epoch_losses)
            cost_history.append(epoch_cost)

            # Training accuracy
            y_pred_train, _ = self.forward(X, training=False)
            train_acc = np.mean(np.argmax(y_pred_train, axis=-1) == y)
            accuracy_history.append(train_acc)

            # Validation metrics
            if X_val is not None and y_val is not None:
                y_pred_val, _ = self.forward(X_val, training=False)
                val_cost = self._cross_entropy_loss(y_pred_val, y_val)
                val_acc = np.mean(np.argmax(y_pred_val, axis=-1) == y_val)
                val_cost_history.append(val_cost)
                val_accuracy_history.append(val_acc)

                # Track best model
                if val_cost < best_cost:
                    best_cost = val_cost
                    best_accuracy = val_acc
                    best_weights = [{'W': l['W'].copy(), 'b': l['b'].copy()} for l in self._layers]
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if epoch_cost < best_cost:
                    best_cost = epoch_cost
                    best_accuracy = train_acc
                    best_weights = [{'W': l['W'].copy(), 'b': l['b'].copy()} for l in self._layers]
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Learning rate decay (fixed schedule)
            if epoch > 0 and epoch % lr_decay_epochs == 0:
                learning_rate *= lr_decay

            learning_rate_history.append(learning_rate)

            # Progress output
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}: cost={epoch_cost:.4f}, acc={train_acc:.2%}, lr={learning_rate:.6f}"
                if X_val is not None:
                    msg += f", val_cost={val_cost:.4f}, val_acc={val_acc:.2%}"
                print(msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

        # Restore best weights
        if best_weights:
            self._layers = best_weights

        training_time = time.time() - start_time
        self._trained = True

        if verbose:
            print("=" * 60)
            print(f"Training completed in {training_time:.2f}s")
            print(f"Best cost: {best_cost:.4f}")
            print(f"Best accuracy: {best_accuracy:.2%}")
            print("=" * 60)

        self.training_history = StandardTrainingResult(
            success=True,
            epochs_completed=epoch + 1,
            final_cost=cost_history[-1],
            final_accuracy=accuracy_history[-1],
            best_cost=best_cost,
            best_accuracy=best_accuracy,
            cost_history=cost_history,
            accuracy_history=accuracy_history,
            val_cost_history=val_cost_history,
            val_accuracy_history=val_accuracy_history,
            training_time=training_time,
            learning_rate_history=learning_rate_history
        )

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        y_pred, _ = self.forward(X, training=False)
        return np.argmax(y_pred, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        y_pred, _ = self.forward(X, training=False)
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        y_pred_proba, _ = self.forward(X, training=False)
        y_pred = np.argmax(y_pred_proba, axis=-1)

        accuracy = np.mean(y_pred == y)
        loss = self._cross_entropy_loss(y_pred_proba, y)

        # Top-k accuracy
        top3_acc = np.mean([y[i] in np.argsort(y_pred_proba[i])[-3:] for i in range(len(y))])
        top5_acc = np.mean([y[i] in np.argsort(y_pred_proba[i])[-5:] for i in range(len(y))])

        # Per-class accuracy
        class_acc = {}
        for c in range(self.config.output_size):
            mask = y == c
            if np.sum(mask) > 0:
                class_acc[c] = np.mean(y_pred[mask] == c)

        return {
            'accuracy': accuracy,
            'loss': loss,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'class_accuracy': class_acc
        }

    def count_parameters(self) -> int:
        """Count total parameters."""
        total = 0
        for layer in self._layers:
            total += layer['W'].size + layer['b'].size
        return total

    def get_architecture_summary(self) -> Dict:
        """Get architecture summary."""
        return {
            'input_size': self.config.input_size,
            'hidden_sizes': self.config.hidden_sizes,
            'output_size': self.config.output_size,
            'num_layers': len(self._layers),
            'total_parameters': self.count_parameters(),
            'activation': self.config.activation,
            'dropout': self.config.dropout
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            'config': {
                'input_size': self.config.input_size,
                'hidden_sizes': self.config.hidden_sizes,
                'output_size': self.config.output_size,
                'dropout': self.config.dropout,
                'activation': self.config.activation,
                'seed': self.config.seed
            }
        }

        # Save weights
        for i, layer in enumerate(self._layers):
            data[f'layer_{i}_W'] = layer['W']
            data[f'layer_{i}_b'] = layer['b']

        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> 'StandardNeuralNetwork':
        """Load model from file."""
        data = np.load(path, allow_pickle=True)

        config = StandardNNConfig(
            input_size=int(data['config'].item()['input_size']),
            hidden_sizes=list(data['config'].item()['hidden_sizes']),
            output_size=int(data['config'].item()['output_size']),
            dropout=float(data['config'].item()['dropout']),
            activation=str(data['config'].item()['activation']),
            seed=int(data['config'].item()['seed'])
        )

        model = cls(config)

        # Load weights
        for i in range(len(model._layers)):
            model._layers[i]['W'] = data[f'layer_{i}_W']
            model._layers[i]['b'] = data[f'layer_{i}_b']

        model._trained = True
        return model
