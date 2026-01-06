"""
Training module for Sentiment Analysis.
Handles model training using DynamicNetwork.
"""

import sys
import time
import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

# Add parent path for pydnn import
sys.path.insert(0, '../../python')

try:
    from pydnn import DynamicNetwork, TrainingResult
except ImportError:
    print("Warning: pydnn not found. Please ensure the path is correct.")
    DynamicNetwork = None
    TrainingResult = None


@dataclass
class TrainingMetrics:
    """Container for training metrics and timing information."""
    epochs_completed: int
    final_cost: float
    best_cost: float
    stopping_reason: str
    training_time: float
    samples_per_second: float
    time_per_epoch: float
    cost_history: list
    efficiency_history: list


class SentimentTrainer:
    """
    Trainer class for sentiment analysis models.
    Wraps DynamicNetwork with convenience methods.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        seed: int = 123,
        cost_function: str = "CrossEntropy"
    ):
        """
        Initialize the trainer.

        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            seed: Random seed for network initialization
            cost_function: Loss function to use
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.seed = seed
        self.cost_function = cost_function
        self.network: Optional[DynamicNetwork] = None
        self.training_metrics: Optional[TrainingMetrics] = None

    def build_network(self) -> 'SentimentTrainer':
        """
        Build the neural network.

        Returns:
            Self for method chaining
        """
        if DynamicNetwork is None:
            raise ImportError("pydnn is not available")

        self.network = DynamicNetwork(
            input_shape=(self.input_size,),
            output_size=self.num_classes,
            seed=self.seed,
            cost_function=self.cost_function
        )

        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            callback: Optional callback function
            verbose: Whether to print progress

        Returns:
            TrainingMetrics with training results
        """
        if self.network is None:
            self.build_network()

        if verbose:
            print(f"\nNetwork configuration:")
            print(f"  Input shape: {self.input_size:,} features")
            print(f"  Output size: {self.num_classes} classes")
            print(f"  Training samples: {X_train.shape[0]:,}")
            print(f"\nTraining started...")

        start_time = time.time()
        result = self.network.fit(X_train, y_train, callback=callback, verbose=verbose)
        training_time = time.time() - start_time

        epochs = max(result.epochs_completed, 1)
        samples_per_second = X_train.shape[0] * epochs / training_time

        self.training_metrics = TrainingMetrics(
            epochs_completed=result.epochs_completed,
            final_cost=result.final_cost,
            best_cost=result.best_cost,
            stopping_reason=result.stopping_reason,
            training_time=training_time,
            samples_per_second=samples_per_second,
            time_per_epoch=training_time / epochs,
            cost_history=result.cost_history,
            efficiency_history=result.efficiency_history
        )

        if verbose:
            print(f"\nTraining completed!")
            print(f"  Total training time: {training_time:.2f}s")
            print(f"  Epochs: {result.epochs_completed}")
            print(f"  Final cost: {result.final_cost:.6f}")
            print(f"  Best cost: {result.best_cost:.6f}")
            print(f"  Stopping reason: {result.stopping_reason}")
            print(f"  Time per epoch: {training_time / epochs:.3f}s")
            print(f"  Samples per second: {samples_per_second:,.0f}")

        return self.training_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if self.network is None:
            raise ValueError("Network not trained")

        return self.network.predict(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input features

        Returns:
            Predicted class indices
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)

    def get_health_status(self) -> dict:
        """
        Get network health status.

        Returns:
            Dictionary with health information
        """
        if self.network is None:
            return {"error": "Network not trained"}

        health = self.network.health_status()
        return {
            "state": health.state,
            "diagnosis": health.diagnosis,
            "overall_health": health.overall_health,
            "cancer_score": health.cancer_score,
            "alzheimer_score": health.alzheimer_score,
            "current_layers": health.current_layers,
            "current_nodes": health.current_nodes
        }

    def save(self, path: str, name: str) -> None:
        """
        Save the model.

        Args:
            path: Directory path
            name: Model name
        """
        if self.network is None:
            raise ValueError("Network not trained")

        self.network.save(path, name)
        print(f"Model saved to {path}/{name}")

    def load(self, filepath: str) -> 'SentimentTrainer':
        """
        Load a saved model.

        Args:
            filepath: Path to model JSON file

        Returns:
            Self for method chaining
        """
        if DynamicNetwork is None:
            raise ImportError("pydnn is not available")

        self.network = DynamicNetwork.load(filepath)
        return self

    def get_training_result(self) -> Optional[TrainingResult]:
        """
        Get the raw training result for visualization.

        Returns:
            TrainingResult object or None
        """
        if self.training_metrics is None:
            return None

        # Create a mock TrainingResult for compatibility with visualization
        @dataclass
        class MockResult:
            epochs_completed: int
            final_cost: float
            best_cost: float
            stopping_reason: str
            cost_history: list
            efficiency_history: list
            training_time_ms: int
            success: bool = True
            final_efficiency: float = 0.0
            best_efficiency: float = 0.0

        return MockResult(
            epochs_completed=self.training_metrics.epochs_completed,
            final_cost=self.training_metrics.final_cost,
            best_cost=self.training_metrics.best_cost,
            stopping_reason=self.training_metrics.stopping_reason,
            cost_history=self.training_metrics.cost_history,
            efficiency_history=self.training_metrics.efficiency_history,
            training_time_ms=int(self.training_metrics.training_time * 1000)
        )
